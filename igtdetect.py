#!/usr/bin/env python3
import statistics, abc, logging
from argparse import ArgumentParser, ArgumentTypeError
from copy import copy
from functools import partial
from collections import defaultdict, Counter
import glob, sys, math
from io import TextIOBase
from multiprocessing.pool import Pool
from tempfile import NamedTemporaryFile

from subprocess import Popen, PIPE

from env import *
from collections import Iterator
import re

# -------------------------------------------
# Set up logging
# -------------------------------------------
NORM_LEVEL = 1000
logging.addLevelName(NORM_LEVEL, 'NORMAL')
logging.basicConfig(level=logging.WARN)
LOG = logging.getLogger(name='IGT-Detect')

# -------------------------------------------
# CONSTANTS
# -------------------------------------------
TYPE_FREKI = 'freki'
TYPE_TEXT = 'text'

# -------------------------------------------
# Readers for text/Freki Docs
# -------------------------------------------
class DocReader(Iterator):
    @abc.abstractmethod
    def __init__(self, fh):
        self.fh = fh
        self.lineno = 0
        self.featdict = {}
        self.linedict = {}

        for line in self:
            self.featdict[self.lineno] = get_textfeats(line)
            self.linedict[self.lineno] = line
        self.seek(0)

    def seek(self, offset, whence=0):
        self.fh.seek(offset, whence)

    def get_line(self, lineno):
        return self.linedict[lineno]





# -------------------------------------------
#
# -------------------------------------------
class BBox(object):
    def __init__(self, tup):
        self.llx, self.lly, self.urx, self.ury = tup

    def __str__(self):
        return '{},{},{},{}'.format(self.llx, self.lly, self.urx, self.ury)


class FrekiBlock(list):
    # doc_id=601.tetml page=1 block_id=1-2 bbox=147.5,711.0,465.4,723.0 1 1
    def __init__(self, lines=None, doc_id=None, page=None, block_id=None, bbox=None, start=0, stop=0):
        if lines is None:
            lines = []
        self.doc_id = doc_id
        self.page = page
        self.block_id = block_id
        self.bbox = bbox
        self.start = start
        self.stop = stop
        super().__init__(lines)

    def __copy__(self):
        fb = FrekiBlock(doc_id=self.doc_id, page=self.page, block_id=self.block_id, bbox=self.bbox, start=self.start,
                        stop=self.stop)
        for line in self:
            fb.append(copy(line))
        return fb

    def set_line_label(self, lineno: int, label: str):
        for line in self:
            if line.lineno == lineno:
                line.label = label
                break

    def preamble(self):
        return 'doc_id={} page={} block_id={} bbox={} {} {}'.format(self.doc_id,
                                                                    self.page,
                                                                    self.block_id,
                                                                    self.bbox,
                                                                    self.start,
                                                                    self.stop)

    def __str__(self):
        ret_str = '{}\n'.format(self.preamble())

        # -------------------------------------------
        # Get max length of preambles
        # -------------------------------------------
        max_pre_len = max([len(l.preamble()) for l in self])

        # -------------------------------------------
        # Add the content of the lines in the block.
        # -------------------------------------------
        for line in self:
            ret_str += '{}:{}\n'.format(line.preamble(max_pre_len), line)
        return ret_str

# =============================================================================
# FrekiReader
#
# Structure for reading through Freki Files, that does two things primarily:
#
#    1) Loop through the file once first, processing the lines so that
#       previous and following line features can be included for the current line
#
#    2) Make it so that the files can be looped through
#
# =============================================================================

class FrekiReader(DocReader):
    def __init__(self, fh, lm=None):
        self.fh = fh
        self.lineno = 0

        self.block_id = None
        self.bbox = None
        self.startline = None
        self.stopline = None

        # -------------------------------------------
        # Dictionaries created by scanning the file.
        # -------------------------------------------
        self.featdict = {}
        self.linedict = {}
        self.llxs = {}
        self.widths = {}
        self.block_ids = {}
        self.block_dict = {}
        self.cur_block = None

        # -------------------------------------------
        # Font Counters
        # -------------------------------------------
        self.fonts = Counter()
        self.fontsizes = defaultdict(Counter)

        for line in self:
            self.llxs[self.lineno] = self.bbox.llx
            self.widths[self.lineno] = self.width
            self.featdict[self.lineno] = get_textfeats(line, lm)
            self.block_ids[self.lineno] = self.block_id
            self.linedict[self.lineno] = line

            # -------------------------------------------
            # Scan the line for its font info
            # -------------------------------------------
            for font, size in line.fonts:
                self.fonts.update([font])
                self.fontsizes[font].update([size])

        if len(self.llxs) > 0:
            try:
                self.left_indent = statistics.mode(self.llxs)
            except statistics.StatisticsError:
                self.left_indent = min(self.llxs)
            self.avg_width = statistics.mean(self.widths)
            if len(self.llxs) > 1:
                self.width_stdev = statistics.stdev(self.widths)

        add_frekifeats(self)
        self.seek(0)

    def most_common_font(self):
        return self.fonts.most_common(1)[0][0]

    def most_common_size(self, font):
        return self.fontsizes[font].most_common(1)[0][0]

    @property
    def width(self):
        return self.bbox.urx - self.bbox.llx

    def labels(self):
        return [line.label for lineno, line in sorted(self.linedict.items(), key=lambda x: x[0])]

    def line_numbers(self):
        return sorted(self.linedict.keys())

    def __len__(self):
        return len(self.linedict.keys())

    def __next__(self):
        data = self.fh.__next__()

        while data:
            if data.startswith('doc_id'):
                d = data.split()
                self.startline = int(d[-2])
                self.stopline = int(d[-1])
                nd = {k: v for k, v in (t.split('=') for t in d[:-2])}
                self.bbox = BBox(float(i) for i in nd['bbox'].split(','))
                self.block_id = nd['block_id']

                # Create a new FrekiBlock object...
                fb = FrekiBlock(doc_id=nd['doc_id'], page=int(nd['page']),
                                block_id=nd['block_id'], bbox=self.bbox,
                                start=int(d[-2]), stop=int(d[-1]))
                self.block_dict[nd['block_id']] = fb
                self.cur_block = fb
                data = self.fh.__next__()
                continue

            elif data.startswith('line='):
                try:
                    preamble, text = re.search('(line=.*?):(.*)$', data).groups()

                    pre_dict = {x[0]: x[1] for x in re.findall('(\S+)=(\S+)', preamble)}
                    lineno = int(pre_dict['line'])

                    # -------------------------------------------
                    # Format the tag and flags
                    # -------------------------------------------
                    tag='O'
                    flags=''

                    if 'tag' in pre_dict:
                        fulltag = pre_dict['tag']
                        flags = fulltag.split('+')[1:]
                        tag = fulltag.split('+')[0]

                        # -------------------------------------------
                        # Handle the tag according to the logic in the
                        # config file.
                        # -------------------------------------------

                        # If we are not allowing "multi" labels, only
                        # use the first label.
                        if not USE_MULTI_LABELS(conf):
                            tag = tag.split('-')[0]

                        # If we are using "B" and "I" labels, label the line
                        # according to whether or not a new span has started
                        # or not.
                        if USE_BI_LABELS(conf) and 'O' not in tag:
                            if __name__ == '__main__':
                                if __name__ == '__main__':
                                    if self.block_for_line(max(lineno - 1, 1)).block_id != self.cur_block.block_id:
                                        tag = 'B-{}'.format(tag)
                                    else:
                                        tag = 'I-{}'.format(tag)

                        # If we are not stripping the tags off, re-add them to
                        # the label
                        if not STRIP_FLAGS(conf):
                            tag = '+'.join([tag] + flags)

                    fonts = []
                    if 'fonts' in pre_dict:
                        for f in pre_dict['fonts'].split(','):
                            fonts.append(f.split('-'))

                    l = Line(text, int(lineno), fonts, label=tag, flags=flags)

                    self.lineno = int(lineno)

                    # Update the block dicts
                    self.cur_block.append(l)
                    self.block_ids[int(lineno)] = self.cur_block.block_id
                    return l

                # TODO: This exception never quite gets raised
                except Exception as e:
                    raise(e)


            else:
                data = self.fh.__next__()
                continue

        raise StopIteration()

    def block_for_line(self, lineno) -> FrekiBlock:
        return self.block_dict[self.block_ids[lineno]]


class Line(str):
    """
    This is a class to contain a line that is intended to be a classification target instance.
    """

    def __new__(cls, seq='', lineno: int = 0, fonts: tuple = None, label: str = None, flags: tuple = None):

        if fonts is None:
            fonts = tuple()
        if flags is None:
            flags = tuple()

        l = str.__new__(cls, seq)
        l.lineno = lineno
        l.fonts = fonts
        l.label = label
        l.flags = flags
        return l

    def __copy__(self):
        l = Line(seq=self, lineno=self.lineno, fonts=self.fonts, label=self.label, flags=self.flags)
        return l

    def search(self, pattern, flags=0):
        return re.search(pattern, self, flags=flags)

    def fulltag(self):
        return '+'.join(list(self.label) + list(self.flags))

    def preamble(self, length=0):
        fonts = ','.join(['{}-{}'.format(x[0], x[1]) for x in self.fonts])
        pre = 'line={} tag={} fonts={}'.format(self.lineno, self.fulltag(), fonts)

        return '{{:<{}}}'.format(length).format(pre)


class NgramDict(object):
    def __init__(self):
        self._trigrams = defaultdict(partial(defaultdict, (partial(defaultdict, int))))
        self._unigrams = defaultdict(int)
        self._bigrams = defaultdict(partial(defaultdict, int))
        self._total = 0

    def add(self, c1, c2, c3, n=1):
        self._unigrams[c1] += n
        self._bigrams[c1][c2] += n
        self._trigrams[c1][c2][c3] += n
        self._total += n

    def __getitem__(self, item):
        return self._trigrams[item]

    def size(self):
        return self._total

    def unigram_prob(self, k1):
        return 0 if self._total == 0 else self._unigrams[k1] / self._total

    def bigram_prob(self, k1, k2):
        num = self._bigrams[k1][k2]
        den = self._unigrams[k1]
        return 0 if den == 0 else num / den

    def trigram_prob(self, k1, k2, k3):
        num = self._trigrams[k1][k2][k3]
        den = self._bigrams[k1][k2]
        return 0 if den == 0 else num / den

    def logprob_word(self, word):
        s = 0
        for i, char in enumerate(word):
            pc = '#' if i - 1 < 0 else word[i - 1]
            nc = '#' if i >= len(word) - 1 else word[i + 1]
            prob = self.trigram_prob(pc, char, nc)
            if prob == 0:
                s += float('-inf')
            else:
                s += math.log(self.trigram_prob(pc, char, nc), 10)
        return s

    def mean_logprob_sent(self, sent):
        word_probs = []
        for word in split_words(sent):
            if word:
                word_probs.append(self.logprob_word(word))

        return statistics.mean(word_probs) if word_probs else 0

    def __contains__(self, item):
        return self._trigrams.__contains__(item)

    def __str__(self):
        ret_str = '{'
        for k1 in self._trigrams.keys():
            ret_str += '{} : {{'.format(k1)

            for k2 in self._trigrams[k1]:
                ret_str += '{} : {{'.format(k2)
                for k3 in self._trigrams[k1][k2]:
                    ret_str += '{} : {}'.format(k3, self._trigrams[k1][k2][k3])
                ret_str += '}, '
            ret_str += '}, '

        return ret_str


class TextReader(DocReader):
    def __next__(self):
        data = self.fh.__next__()
        self.lineno += 1
        return data


def get_textfeats(line: Line, lm: NgramDict) -> dict:
    """
    Given a line as input, return the text-based features
    available for that line.
    """

    # Quick local function to check if a
    # feature is enabled in the config
    # and add it to the feature dict if so.
    feats = {}

    def checkfeat(name, func):
        nonlocal feats
        if name in ENABLED_TEXT_FEATS(conf):
            feats[name] = func(line)

    # Quick function to add featuers for words
    # in the line.
    def basic_words(line):
        nonlocal feats
        for word in split_words(line):
            if word:
                feats['word_{}'.format(word)] = 1

    checkfeat(T_BASIC, basic_words)
    checkfeat(T_HAS_LANGNAME, has_langname)
    checkfeat(T_HAS_GRAMS, has_grams)
    checkfeat(T_HAS_PARENTHETICAL, has_parenthetical)
    checkfeat(T_HAS_CITATION, has_citation)
    checkfeat(T_HAS_ASTERISK, has_asterisk)
    checkfeat(T_HAS_UNDERSCORE, has_underscore)
    checkfeat(T_HAS_BRACKETING, has_bracketing)
    checkfeat(T_HAS_QUOTATION, has_quotation)
    checkfeat(T_HAS_NUMBERING, has_numbering)
    checkfeat(T_HAS_LEADING_WHITESPACE, has_leading_whitespace)
    checkfeat(T_HIGH_OOV_RATE, high_en_oov_rate)
    checkfeat(T_MED_OOV_RATE, med_en_oov_rate)
    checkfeat(T_HAS_JPN, has_japanese)
    checkfeat(T_HAS_GRK, has_greek)
    checkfeat(T_HAS_KOR, has_korean)
    checkfeat(T_HAS_ACC, has_accented_latin)
    checkfeat(T_HAS_CYR, has_cyrillic)
    checkfeat(T_HAS_DIA, has_diacritic)
    checkfeat(T_HAS_UNI, has_unicode)
    checkfeat(T_HAS_YEAR, has_year)
    checkfeat(T_HIGH_GLS_OOV_RATE, high_gls_oov_rate)

    return feats


def add_frekifeats(r: DocReader):
    for lineno in sorted(r.featdict.keys()):

        feats = {}

        # Use this function to check the
        # feature constant name against the
        # list of enabled features, and trigger
        # the appropriate function if it's enabled.
        def checkfeat(name, func):
            nonlocal feats
            if name in ENABLED_FREKI_FEATS(conf):
                feats[name] = func(r, lineno)

        # Apply each feature if it is enabled
        checkfeat(F_IS_INDENTED, isindented)
        checkfeat(F_IS_FIRST_PAGE, is_first_page)
        checkfeat(F_PREV_LINE_SAME_BLOCK, prev_line_same_block)
        checkfeat(F_NEXT_LINE_SAME_BLOCK, next_line_same_block)
        checkfeat(F_HAS_NONSTANDARD_FONT, has_nondefault_font)
        checkfeat(F_HAS_SMALLER_FONT, has_smaller_font)
        checkfeat(F_HAS_LARGER_FONT, has_larger_font)

        r.featdict[lineno].update(feats)


def get_all_line_feats(featdict, lineno) -> dict:
    """
    Given a dictionary mapping lines to features, get
    a new feature dict that includes features for the
    current line, as well as n-1 and n-2 lines, and n+1.
    """

    # Always include the features for the current line.
    cur_feats = featdict[lineno]
    all_feats = copy(cur_feats)

    # Use the features for the line before the previous one (n-2)
    if USE_PREV_PREV_LINE(conf):
        prev_prev_feats = featdict.get(lineno - 2, {})
        for prev_key in prev_prev_feats.keys():
            all_feats['prev_prev_' + prev_key] = prev_prev_feats[prev_key]

    # Use the features for the previous line (n-1)
    if USE_PREV_LINE(conf):
        prev_feats = featdict.get(lineno - 1, {})
        for prev_key in prev_feats.keys():
            all_feats['prev_' + prev_key] = prev_feats[prev_key]

    # Use the features for the next line (n+1)
    if USE_NEXT_LINE(conf):
        next_feats = featdict.get(lineno + 1, {})
        for next_key in next_feats.keys():
            all_feats['next_' + next_key] = next_feats[next_key]

    return all_feats


def _path_rename(path, ext):
    return os.path.splitext(os.path.basename(path))[0] + ext


def get_feat_path(path):
    return os.path.join(FEAT_DIR(conf), _path_rename(path, '_feats.txt'))

def get_classifications_path(path):
    return os.path.join(os.path.join(DEBUG_DIR(conf), 'raw_classifications'), _path_rename(path, '_classifications.txt'))

classified_suffix = '_classified.txt'

def get_classified_path(path):
    return os.path.join(OUT_DIR(conf), _path_rename(path, classified_suffix))

def get_gold_for_classified(path):
    return os.path.join(GOLD_DIR(conf), os.path.basename(path).replace(classified_suffix, '.txt'))

def get_weight_path(path):
    return os.path.join(DEBUG_DIR(conf), _path_rename(path, '_weights.txt'))


# -------------------------------------------
# Perform feature extraction.
# -------------------------------------------
def extract_feats(filelist, filetype, overwrite=False, skip_noisy=False):
    """
    Perform feature extraction over a list of files.

    Call extract_feat_for_path() in parallel for a speed boost.
    """

    p = Pool()

    for path in filelist:
        # p.apply_async(extract_feat_for_path, args=[path, overwrite, skip_noisy])
        extract_feat_for_path(path, overwrite, skip_noisy)

    p.close()
    p.join()


def extract_feat_for_path(path, overwrite=False, skip_noisy=False):
    """
    Perform feature extraction for a single file.

    The output files are in svmlight format, namely:

        LABEL   feature_1:value_1   feature_2:value_2 ...etc

    The "skip_noisy" parameter is intended for training data that
    was created automatically, and for which the labels were mapped,
    but seem unlikely to be correct. Such noisy labels are preceded by
    an asterisk.
    """
    feat_path = get_feat_path(path)

    path_rel = os.path.abspath(path)
    feat_rel = os.path.abspath(feat_path)

    # -------------------------------------------
    # Skip generating the text feature for this path
    # if it's already been generated and the user
    # has not asked to overwrite them.
    # -------------------------------------------
    if os.path.exists(feat_path) and (not overwrite):
        LOG.log(NORM_LEVEL, 'File "{}" already generated, not regenerating (use -f to force)...'.format(feat_path))
        return

    LOG.log(NORM_LEVEL, 'Opening file "{}" for feature extraction to file "{}"...'.format(path_rel, feat_rel))

    os.makedirs(os.path.dirname(feat_path), exist_ok=True)
    with open(feat_path, 'w', encoding='utf-8') as train_f:

        with open(path, 'r', encoding='utf-8') as f:
            # r = FrekiReader(f) if filetype == TYPE_FREKI else TextReader(f)
            r = FrekiReader(f)

            # Now, let's iterate through again and extract features.
            for lineno in sorted(r.featdict.keys()):
                label = r.linedict[lineno].label

                # If the label contains an asterisk, that means
                # it is very noisy. Don't use it for either "O" or any
                # in-IGT label.
                if label.startswith('*'):
                    if skip_noisy:
                        continue
                    else:
                        label = label.replace('*', '')

                all_feats = get_all_line_feats(r.featdict, lineno)

                write_training_vector(all_feats, label, train_f)


def write_training_vector(featdict, label, out: TextIOBase = sys.stdout):
    """

    :param featdict:
    :param label:
    :param out:
    :return:
    """
    out.write('{:s}'.format(label))
    for feat in sorted(featdict.keys()):
        val = featdict[feat]
        val_str = 1 if val else 0
        if val_str:
            out.write('\t{}:{}'.format(feat, val_str))
    out.write('\n')


# =============================================================================
# FEATURES
# =============================================================================
def isindented(r: FrekiReader, lineno: int):
    return r.left_indent and (r.llxs[lineno] > r.left_indent)


def has_smaller_font(r: FrekiReader, lineno: int):
    line = r.linedict[lineno]
    for font, size in line.fonts:
        if size < r.most_common_size(font):
            return True
    return False


def has_larger_font(r: FrekiReader, lineno: int):
    line = r.linedict[lineno]
    for font, size in line.fonts:
        if size > r.most_common_size(font):
            return True
    return False


def has_nondefault_font(r: FrekiReader, lineno: int):
    line = r.linedict[lineno]
    mcf = r.most_common_font()
    for font, size in line.fonts:
        if font != mcf:
            return True
    return False


def thinner_than_usual(r: FrekiReader):
    """
    Return true if the width of the current line's block
    is more than one std. dev. away from the mean.

    :rtype: bool
    """
    return r.width and (r.width <= (r.avg_width - r.width_stdev))


def has_grams(line: str):
    return bool(line.search('|'.join(GRAM_LIST), flags=re.I) or line.search('|'.join(CASED_GRAM_LIST)))


def has_parenthetical(line: str):
    return bool(line.search('\(.*\)'))


# Cover four-digit numbers from 1800--2019
year_str = '(?:1[8-9][0-9][0-9]|20[0-1][0-9])'


def has_citation(line: str):
    return bool(line.search('\([^,]+, {}\)'.format(year_str)))


def has_year(line: str):
    return bool(line.search(year_str))


def has_asterisk(line: Line):
    return '*' in line


def has_underscore(line: Line):
    return '_' in line


def has_bracketing(line: Line):
    return bool(line.search('\[.*\]'))


def has_numbering(line: Line):
    return bool(line.search('^\s*\(?[0-9a-z]+[\)\.]'))


def has_leading_whitespace(line: Line):
    return bool(line.search('^\s+'))

# -------------------------------------------
# Various Unicode Ranges
# -------------------------------------------

def has_cyrillic(line: Line):
    return bool(line.search('[\u0400-\u04FF]', flags=re.UNICODE))


def has_diacritic(line: Line):
    return bool(line.search('[\u0300–\u036F]|[\u1AB0-\u1AFF]|[\u1DC0-\u1DFF]|[\u20D0-\u20FF]|[\uFE20-\uFE2F]',
                            flags=re.UNICODE))


def has_greek(line: Line):
    return bool(line.search('[\u0370-\u03FF]|[\u1F00-\u1FFF]', flags=re.UNICODE))


def has_japanese(line: Line):
    has_kanji = bool(line.search('[\u4E00-\u9FBF]', flags=re.U))
    has_hiragana = bool(line.search('[\u3040-\u309F]', flags=re.U))
    has_katakana = bool(line.search('[\u30A0-\u30FF]', flags=re.U))
    return has_kanji or has_hiragana or has_katakana


def has_accented_latin(line: Line):
    return bool(line.search('[\u00C0-\u00FF]', flags=re.U))


def has_korean(line: Line):
    return bool(line.search('[\uAC00-\uD7A3]', flags=re.U))

def has_unicode(line: Line):
    cyr = has_cyrillic(line)
    dia = has_diacritic(line)
    grk = has_greek(line)
    jpn = has_japanese(line)
    kor = has_korean(line)
    acc = has_accented_latin(line)
    return cyr or dia or grk or jpn or acc or kor

# -------------------------------------------

word_re = re.compile('(\w+)', flags=re.UNICODE)


def clean_word(s):
    w_match = word_re.findall(s)
    return w_match

# -------------------------------------------
# OOV Rate Functions
#
# Use a set threshold to decide at what
# ratio of OOV words to In-Vocabulary words
# constitutes being too dissimilar.
# -------------------------------------------

def med_en_oov_rate(line: Line):
    return HIGH_OOV_THRESH(conf) > oov_rate(en_wl, line) > MED_OOV_THRESH(conf)


def high_en_oov_rate(line: Line):
    return oov_rate(en_wl, line) >= HIGH_OOV_THRESH(conf)


def high_gls_oov_rate(line: Line):
    return oov_rate(gls_wl, line) > HIGH_OOV_THRESH(conf)


def oov_rate(wl: WordlistFile, line: Line):
    if not wl:
        return 0.0
    else:

        words = []
        for word in split_words(line):
            words.extend(clean_word(word.lower()))

        if len(words) <= 2:
            return 0.0

        oov_words = Counter([w in en_wl for w in words])
        c_total = sum([v for v in oov_words.values()])

        if not c_total:
            return 0.0
        else:
            oov_rate = oov_words[False] / c_total
            return oov_rate

langs = set([])
def init_langnames():
    global langs
    if len(langs) == 0:
        with open(LNG_NAMES(conf), 'r', encoding='utf-8') as f:
            for line in f:
                last_col = ' '.join(line.split()[3:])
                for langname in last_col.split(','):
                    langname = langname.replace('[', '')
                    if len(langname) >= 5:
                        langs.add(langname.lower())

lang_re = re.compile('({})'.format('|'.join(langs), flags=re.I))


def has_langname(line: Line):
    init_langnames()
    return bool(line.search(lang_re))


def has_quotation(line: Line):
    """ Return true if the line in question surrounds more than one word in quotes """
    return bool(line.search('[\'\"‘`“]\S+\s+.+[\'\"’”]'))


def is_first_page(r: FrekiReader, lineno: int):
    block = r.block_for_line(lineno)
    return block.page == 1


def prev_line_same_block(r: FrekiReader, lineno: int):
    return r.block_ids.get(lineno - 1) == r.block_ids.get(lineno)


def next_line_same_block(r: FrekiReader, lineno: int):
    return r.block_ids.get(lineno + 1) == r.block_ids.get(lineno)


# -------------------------------------------
# TRAIN THE CLASSIFIER
# -------------------------------------------


def label_sort(l):
    order = ['O', 'B', 'I', 'L', 'L-T', 'G', 'T', 'M']
    if l in order:
        return order.index(l)
    else:
        return float('inf')


class ClassifierInfo():
    """
    This is a class for storing information about
    the mallet classifier, such as which features are
    the most informative, and what labels are among
    those that are expected to be seen.
    """

    def __init__(self):
        self.featdict = defaultdict(partial(defaultdict, float))
        self.labels = set([])

    def add_feat(self, label, feat, weight):
        self.featdict[feat][label] = float(weight)
        self.labels.add(label)

    def write_features(self, out=sys.stdout, limit=30):


        vals = []
        defaults = []
        for feat in self.featdict.keys():
            for label, val in self.featdict[feat].items():
                if feat == '<default>':
                    defaults.append((feat, label, val))
                else:
                    vals.append((feat, label, val))

        defaults = sorted(defaults, key=lambda x: label_sort(x[1]))

        # If limit is None, set it to dump all features.
        if limit is None:
            limit = len(vals)

        vals = sorted(vals, key=lambda x: abs(x[2]), reverse=True)[:limit]

        longest_featname = max([len(x[0]) for x in vals])
        longest_label = max([len(x[1]) for x in vals] + [5])

        format_str = '{{:{}}}\t{{:{}}}\t{{:<5.6}}\n'.format(longest_featname, longest_label)

        out.write(format_str.format('feature', 'label', 'weight'))
        linesep = '-' * (longest_featname + longest_label + 10)+'\n'
        out.write(linesep)
        for d in defaults:
            out.write(format_str.format(*d))
        out.write(linesep)
        for val in vals:
            out.write(format_str.format(*val))


def get_classifier_info(classpath, limit=30):
    """
    List the top $limit most informative features for the
     given classifier.

    :param classpath: Path to the classifier
    :param limit: Number of features to list
    """
    p = Popen([INFO_BIN(conf),
               '--classifier', classpath], stdout=PIPE)

    ci = ClassifierInfo()

    label = None
    for line in p.stdout:
        line = line.decode('utf-8')
        if line.startswith('FEATURES FOR CLASS'):
            label = line.split()[-1]

        else:
            feat, weight = line.split()
            ci.add_feat(label, feat, weight)

    return ci


def combine_feat_files(pathlist, out_path=None):
    """

    :param pathlist:
    :param out_path:
    :return:
    """
    # Create the training file.

    if out_path is None:
        combined_f = NamedTemporaryFile(mode='w', encoding='utf-8', delete=False)
        out_path = combined_f.name
    else:
        combined_f = open(out_path, 'w', encoding='utf-8')

    # -------------------------------------------
    # 1) Combine all the instances in the files...
    # -------------------------------------------
    for path in pathlist:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                combined_f.write(line)
    combined_f.close()
    return out_path


# =============================================================================
# Train the classifier given a list of files
# =============================================================================
def train_classifier(filelist, classifier_path, config):
    """
    Train the classifier based on the input files in filelist.

    :param filelist: List of files to train the classifier based upon.
    :param classifier_path:
    :return:
    """
    # Get feature paths for all the input files.
    feat_paths = [get_feat_path(p) for p in filelist]

    if not feat_paths:
        LOG.critical("No text vector files were found.")

    # Now, create an output feature file...
    combined_feat_path = combine_feat_files(feat_paths)

    # And convert it to a vector.
    train_vector_path = convert_to_vectors(combined_feat_path)

    args = JAVA_ARGS(config) + ['cc.mallet.classify.tui.Vectors2Classify',
                                   '--trainer', 'MaxEntTrainer',
                                   '--input', train_vector_path,
                                   '--output-classifier', classifier_path]

    LOG.warn(' '.join(args))

    p = Popen(args)
    p.wait()

    ci = get_classifier_info(classifier_path, config)
    if DEBUG_ON(config):
        classifier_debug_dir = os.path.join(DEBUG_DIR(config), 'classifier_info')
        os.makedirs(classifier_debug_dir, exist_ok=True)
        c_name = os.path.splitext(os.path.basename(classifier_path))[0]+'_feat_weights.txt'
        classinfo_path = os.path.join(classifier_debug_dir, c_name)

        with open(classinfo_path, 'w', encoding='utf-8') as f:
            ci.write_features(out=f, limit=None)

    os.unlink(combined_feat_path)


def convert_to_vectors(path, prev_pipe_path=None, out_path=None):
    """
    Take an svm-light format file and return a temporary file
    converted to the mallet format binary vectors.
    """

    # If a desired output path is not specified, then
    # create a temporary file.
    if not out_path:
        vector_f = NamedTemporaryFile('w', delete=False)
        vector_f.close()
        vector_path = vector_f.name
    else:
        vector_path = out_path
        os.makedirs(os.path.dirname(out_path), exist_ok=True)

    args = [MALLET_BIN(conf), 'import-svmlight',
            '--input', path,
            '--output', vector_path]
    if prev_pipe_path:
        args += ['--use-pipe-from', prev_pipe_path]
    p = Popen(args)
    p.wait()

    return vector_path


# -------------------------------------------
# DO the classification
# -------------------------------------------
def classify_doc(path, classifier, classifier_info):

    # -------------------------------------------
    # We don't care about the labels that are currently
    # on the document. However, due to what I'd describe
    # as a bug in mallet, labels on the testing instances
    # that have not been seen before will cause mallet to
    # crash. So, as a workaround, write out a temporary
    # file and replace the labels in it with a random
    # known label from the classifier.
    # -------------------------------------------
    label_to_use = classifier_info.labels.pop()

    temp_in = NamedTemporaryFile('w', encoding='utf-8', delete=False)
    with open(path, 'r') as f:
        for line in f:
            data = line.split()
            data[0] = label_to_use
            temp_in.write('{}\n'.format('\t'.join(data)))
    temp_in.close()

    # -------------------------------------------
    # Next, run mallet on the temporary file,
    # piping the output to a pipe.
    # -------------------------------------------
    classifications = []
    p = Popen([MALLET_BIN(conf), 'classify-svmlight',
               '--classifier', classifier,
               '--input', temp_in.name,
               '--output', '-'], stdout=PIPE)

    # -------------------------------------------
    # Read the output, which looks like:
    #      array:171	O	0.9958349677234593	L	0.0012096458407225073 ... LABEL    WEIGHT
    #
    # So, find all of the Label___weight pairs, parse the weight, and
    # reverse sort by weight.
    # -------------------------------------------
    for line in p.stdout:
        scores = []
        for label, score in re.findall('(\S+)\s+([0-9\-\.E]+)', line.decode('utf-8')):
            scores.append((label, float(score)))

        classification = tuple(sorted(scores, key=lambda x: x[1], reverse=True))
        classifications.append(classification)

    if p.returncode:
        raise Exception("There was an error running the classifier.")

    os.unlink(temp_in.name) # Delete the temporary output file with the replaced labels

    return classifications


class SpanCounter(object):
    """
    This is a utility class that helps calculate
    performance over spans of IGT lines, rather
    than the per-line accuracies, which are in
    some ways less helpful.
    """

    def __init__(self):
        self.last_gold = 'O'
        self.last_guess = 'O'

        self.guess_spans = set()
        self.gold_spans = set()

        self.cur_guess_span = []
        self.cur_gold_span = []

        self._matrix = defaultdict(partial(defaultdict, int))

    def exact_matches(self):
        return len(self.guess_spans & self.gold_spans)

    def span_prf(self, exact=True):
        return (self.span_precision(exact), self.span_recall(exact), self.span_fmeasure(exact))

    def span_precision(self, exact=True):
        if not exact:
            return self.partial_precision()
        else:
            den = len(self.guess_spans)
            return self.exact_matches() / den if den > 0 else 0

    def span_recall(self, exact=True):
        if not exact:
            return self.partial_recall()
        else:
            den = len(self.gold_spans)
            return self.exact_matches() / den if den > 0 else 0

    def span_fmeasure(self, exact=True):
        denom = (self.span_precision(exact) + self.span_recall(exact))
        if denom == 0:
            return 0
        else:
            return 2 * (self.span_precision(exact) * self.span_recall(exact)) / denom

    def partial_recall(self):
        """
        The partial span recall is calculated by the number of gold spans for which a match
        is found among the system outputs.

        (There could be multiple system spans which overlap with the gold span,
        we simply care that one of them overlaps).
        :return:
        """
        matches = 0
        for gold_start, gold_stop in [(s[0], s[-1]) for s in self.gold_spans]:
            for sys_start, sys_stop in [(s[0], s[-1]) for s in self.guess_spans]:
                if (sys_stop >= gold_start >= sys_start) or (sys_stop >= gold_stop >= sys_start):
                    matches += 1
                    break
        return matches / len(self.gold_spans) if self.gold_spans else 0

    def partial_precision(self):
        """
        The partial span precision is calculated by the number of system spans which overlap
        in some way with a system span.
        :return:
        """
        matches = 0

        for sys_start, sys_stop in [(s[0], s[-1]) for s in self.guess_spans]:
            for gold_start, gold_stop in [(s[0], s[-1]) for s in self.gold_spans]:

                # We define a partial match by whether either the start or stop index of
                # the system span occurs within the [start,stop] range of at least one gold span.
                if (gold_stop >= sys_start >= gold_start) or (gold_stop >= sys_stop >= gold_start):
                    matches += 1
                    break

        return matches / len(self.guess_spans) if self.guess_spans else 0

    def add_line(self, lineno, gold, guess):
        """
        For a given line number, catalog it.
        """
        if guess != 'O':
            self.cur_guess_span.append(lineno)
        elif guess == 'O' and self.last_guess != 'O':
            self.guess_spans.add(tuple(self.cur_guess_span))
            self.cur_guess_span = []

        if gold != 'O':
            self.cur_gold_span.append(lineno)
        elif gold == 'O' and self.last_gold != 'O':
            self.gold_spans.add(tuple(self.cur_gold_span))
            self.cur_gold_span = []

        self._matrix[gold][guess] += 1

        self.last_guess = guess
        self.last_gold = gold

    def _matches(self, exclude=list()):
        return [self._matrix[gold][gold] for gold in self._labels() if gold not in exclude]

    def _gold_sums(self, exclude=list()):
        gold_totals = defaultdict(int)
        for gold in self._matrix.keys():
            if gold in exclude:
                continue
            for guess in self._matrix[gold]:
                gold_totals[gold] += self._matrix[gold][guess]
        return [gold_totals[l] for l in self._labels()]

    def _guess_sums(self, exclude=list()):
        guess_totals = defaultdict(int)
        for gold in self._matrix.keys():
            for guess in self._matrix[gold]:
                if guess in exclude:
                    continue
                guess_totals[guess] += self._matrix[gold][guess]

        return [guess_totals[l] for l in self._labels()]

    def _recalls(self):
        return [matches / sums if sums > 0 else 0 for matches, sums in zip(self._matches(), self._gold_sums())]

    def _labels(self):
        return sorted(set(self._matrix.keys()) | set(
            [inner_key for outer_key in self._matrix.keys() for inner_key in self._matrix[outer_key].keys()]),
                      key=label_sort)

    # -------------------------------------------
    # Functions for calculate per-label
    # precision, recall, and f-measure, optionally
    # excluding certain labels.
    # -------------------------------------------

    def recall(self, exclude=list()):
        num = sum(self._matches(exclude))
        den = sum(self._gold_sums(exclude))
        return num / den if den > 0 else 0

    def precision(self, exclude=list()):
        """
        Calculate label precision
        """
        num = sum(self._matches(exclude))
        den = sum(self._guess_sums(exclude))
        return num / den if den > 0 else 0

    def prf(self, exclude=list()):
        return (self.precision(exclude), self.recall(exclude), self.f_measure(exclude))

    def f_measure(self, exclude=list()):
        denom = self.precision(exclude) + self.recall(exclude)
        if denom == 0:
            return 0
        else:
            return 2 * (self.precision(exclude) * self.recall(exclude)) / denom

    def _vals(self):
        return [[self._matrix[gold][label] for gold in self._labels()] for label in self._labels()]

    def matrix(self, csv=False):
        # Switch the delimiter from tab to comma
        # if using a csv format.
        delimiter = '\t'
        if csv:
            delimiter = ','

        ret_str = '{} COLS: Gold --- ROWS: Predicted\n'.format(delimiter)
        ret_str += delimiter.join([''] + ['{:4}'.format(l) for l in self._labels()]) + '\n'
        for label in self._labels():
            vals = [self._matrix[gold][label] for gold in self._labels()]
            matches = self._matrix[label][label]
            compares = sum(vals)
            precision = matches / compares if compares > 0 else 0
            ret_str += delimiter.join([label] + ['{:4}'.format(v) for v in vals] + ['{:.2f}'.format(precision)]) + '\n'

        ret_str += delimiter.join([''] + ['{:4.2f}'.format(r) for r in self._recalls()]) + '\n'
        return ret_str


# =============================================================================
# Testing (Apply Classifier to new Documents)
# =============================================================================

def classify_docs(filelist, class_path, conf_file):
    feat_paths = [get_feat_path(p) for p in filelist]
    if not feat_paths:
        LOG.critical("No text vector files were found.")
        sys.exit()

    sc = SpanCounter()

    for path, feat_path in zip(filelist, feat_paths):

        # -------------------------------------------
        # Open the output classification path
        # -------------------------------------------
        classified_path = get_classified_path(path)
        os.makedirs(OUT_DIR(conf), exist_ok=True)
        out_f = open(classified_path, 'w', encoding='utf-8')

        # Grab the info on the previously trained classifier
        # so that we can add the "dummy" labels to the test
        # file so that mallet doesn't crash when it sees
        # a previously unseen label.
        LOG.info(NORM_LEVEL, "Getting classifier info to avoid unknown label crashes...")
        ci = get_classifier_info(class_path)

        # -------------------------------------------
        # Classify the individual docs
        # -------------------------------------------
        LOG.log(NORM_LEVEL, 'Classifying file "{}"'.format(path))
        classifications = classify_doc(feat_path, class_path, ci)
        fr = FrekiReader(open(path, 'r', encoding='utf-8'))

        f_len = len(list(fr.featdict.keys()))
        c_len = len(classifications)

        if f_len != c_len:
            LOG.critical("The number of lines ({}) does not match the number of classifications ({}). Skipping file {}".format(f_len, c_len, path))
            continue

        # -------------------------------------------
        # We want to get not only classification accuracy,
        # but also count how well non-O "spans" get
        # labeled.
        # -------------------------------------------

        working_block = None  # Keep track of which block the last block is so we can write out

        # This file will contain the raw labelings from the classifier.
        if DEBUG_ON(conf):
            os.makedirs(os.path.dirname(get_classifications_path(path)), exist_ok=True)
            LOG.log(NORM_LEVEL, 'Writing out raw classifications "{}"'.format(get_classifications_path(path)))
            classification_f = open(get_classifications_path(path), 'w')

        # -------------------------------------------
        # Iterate through the returned classifications
        # and assign them to the lines in the test file.
        #
        # Optionally, write out the raw classification distribution.
        # -------------------------------------------
        for lineno, classification in zip(sorted(fr.featdict.keys()), classifications):

            # Write the line number and classification probabilities to the debug file.
            if DEBUG_ON(conf):
                classification_f.write('{}:'.format(lineno))
                for label, weight in classification:
                    classification_f.write('\t{}  {:.3e}'.format(label, weight))
                classification_f.write('\n')
                classification_f.flush()

            # Get the most probable result from
            # the list distribution of labels returned
            # by the classifier.
            prediction = classification[0][0]

            # Get the block that contains this line.
            cur_block = fr.block_for_line(lineno)

            # If this is our first block (the previous block
            # was None), then don't try to write it out,
            # but do set the last seen block to the current one.
            if working_block is None:
                working_block = copy(cur_block)

            # Set the label for the line in the working block
            # before potentially writing it out.
            working_block.set_line_label(lineno, prediction)

            # If we have moved to a new block, write
            # out the previous one.
            #
            # (NB: I write out the blocks as a chunk, rather than
            #      the lines one at a time because it helps keep
            #      regular spacing between lines)
            if cur_block.block_id != working_block.block_id:
                out_f.write('{}\n'.format(working_block))
                working_block = copy(cur_block)

        # Write out the final block in the file.
        out_f.write('{}\n'.format(working_block))
        out_f.close()

        if DEBUG_ON(conf):
            classification_f.close()


def eval_files(filelist, out_path, csv):
    """
    Given a list of freki files, evaluate them against
    the files given in the gold dir.

    If the gold dir does not exist, or does not contain
    the specified file, make sure to log an error.
    """
    # Set up the output stream
    if out_path is None:
        out_f = sys.stdout
    else:
        out_f = open(out_path, 'w')

    if not os.path.exists(GOLD_DIR(conf)):
        LOG.critical('The gold file directory "{}" is missing or is unavailable.'.format(GOLD_DIR(conf)))
        sys.exit(2)
    elif os.path.isfile(GOLD_DIR(conf)):
        LOG.error('The gold file directory "{}" appears to be a file, not a directory.'.format(GOLD_DIR(conf)))
        sys.exit(2)

    # Create the counter to iterate over all the files.
    sc = SpanCounter()
    for eval_path in filelist:
        gold_path = get_gold_for_classified(eval_path)
        if not os.path.exists(gold_path):
            LOG.warning('No corresponding gold file was found for the evaluation file "{}"'.format(eval_path))
        else:
            eval_file(eval_path, gold_path, sc=sc)

    # Now, write out the sc results.
    delimiter = '\t'
    if csv:
        delimiter = ','
    out_f.write(sc.matrix()+'\n')

    out_f.write('----- Labels -----\n')
    out_f.write(' Classifiation Acc: {:.2f}\n'.format(sc.precision()))
    out_f.write('       Non-O P/R/F: {}\n\n'.format(delimiter.join(['{:.2f}'.format(x) for x in sc.prf(['O'])])))
    out_f.write('----- Spans ------\n')
    out_f.write('  Exact-span P/R/F: {}\n'.format(delimiter.join(['{:.2f}'.format(x) for x in sc.span_prf(exact=True)])))
    out_f.write('Partial-span P/R/F: {}\n'.format(delimiter.join(['{:.2f}'.format(x) for x in sc.span_prf(exact=False)])))
    
    out_f.close()

def eval_file(eval_path, gold_path, sc=None, outstream=sys.stdout):
    """
    Look for the filename that matches the specified file
    """
    eval_fr = FrekiReader(open(eval_path, 'r'))
    gold_fr = FrekiReader(open(gold_path, 'r'))

    if len(eval_fr) != len(gold_fr):
        LOG.error('The evaluation file "{}" and the gold file "{}" appear to have a different number of lines. Evaluation aborted.'.format(eval_path, gold_path))
    else:
        if sc is None:
            sc = SpanCounter()

        for lineno in eval_fr.line_numbers():
            eval_label = eval_fr.get_line(lineno).label
            gold_label = gold_fr.get_line(lineno).label
            sc.add_line(lineno, gold_label, eval_label)

        return sc
        print(sc.matrix())
        print()
        print(' Classifiation Acc: {:.2f}'.format(sc.precision()))
        print('       Non-O P/R/F: {}'.format(','.join(['{:.2f}'.format(x) for x in sc.prf(['O'])])))
        print('  Exact-span P/R/F: {}'.format(','.join(['{:.2f}'.format(x) for x in sc.span_prf(exact=True)])))
        print('Partial-span P/R/F: {}'.format(','.join(['{:.2f}'.format(x) for x in sc.span_prf(exact=False)])))


def flatten(seq):
    flat = []
    if not (isinstance(seq, list) or isinstance(seq, tuple)):
        return [seq]
    else:
        for elt in seq:
            flat.extend(flatten(elt))
        return flat


# -------------------------------------------
# ARG TYPES
# -------------------------------------------

def globfiles(pathname):
    g = glob.glob(pathname)
    if not g:
        raise ArgumentTypeError('No files found matching pattern "{}".\nCheck that the path is valid and that containing directories exist.'.format(pathname))
    else:
        return g
# -------------------------------------------

#
def split_words(sent):
    return [re.sub('[#:]', '', w.lower()) for w in re.split('[\.\-\s]', sent)]


if __name__ == '__main__':
    p = ArgumentParser()

    subparsers = p.add_subparsers(help='Valid subcommands', dest='subcommand')
    subparsers.required = True


    # -------------------------------------------
    # TRAINING
    # -------------------------------------------
    train_p = subparsers.add_parser('train')
    train_p.add_argument('files', nargs='+', type=globfiles)
    train_p.add_argument('--type', choices=[TYPE_FREKI, TYPE_TEXT], default=TYPE_FREKI)
    train_p.add_argument('-f', '--overwrite', action='store_true', help='Overwrite text vectors.')
    train_p.add_argument('-o', '--out', required=True, help='Output path for the classifier.')
    train_p.add_argument('-c', '--config', help='Alternate config file.')

    # -------------------------------------------
    # TESTING
    # -------------------------------------------
    test_p = subparsers.add_parser('test')

    test_p.add_argument('--type', choices=[TYPE_FREKI, TYPE_TEXT], default=TYPE_FREKI)
    test_p.add_argument('-f', '--overwrite', action='store_true', help='Overwrite text vectors')
    test_p.add_argument('--classifier', required=True)
    test_p.add_argument('files', nargs='+', type=globfiles, help='Files to apply classifier against')
    test_p.add_argument('-c', '--config', help='Alternate config file')
    # -------------------------------------------


    # -------------------------------------------
    # EVAL
    # -------------------------------------------
    eval_p = subparsers.add_parser('eval')

    eval_p.add_argument('files', nargs='+', help='Path expression (wildcards accepted) to files to evaluate.', type=globfiles)
    eval_p.add_argument('-o', '--output', help='Write the evaluation result to a file. If not specified, stdout is used.')
    eval_p.add_argument('--csv', help='Format the output as CSV')
    eval_p.add_argument('-c', '--config', help='Alternate config file')
    # -------------------------------------------

    args = p.parse_args()

    # -------------------------------------------
    # Read in the config file, if provided. Otherwise
    # the default config parser class will use the provided
    # defaults.
    # -------------------------------------------
    global conf
    conf = DefaultConfigParser()
    if args.config:
        if not os.path.exists(args.config):
            LOG.critical('The config file "{}" could not be found.'.format(args.config))
            sys.exit(2)
        else:
            conf.read(args.config)

    # -------------------------------------------
    # Debug
    # -------------------------------------------
    if DEBUG_ON(conf):
        os.makedirs(DEBUG_DIR(conf), exist_ok=True)

    # -------------------------------------------
    # Load wordlist files for performance if testing or training
    # -------------------------------------------
    global en_wl, gls_wl
    if args.subcommand in ['test', 'train', 'eval']:
        en_wl = EN_WL(conf)
        gls_wl = GL_WL(conf)


    filelist = []
    if hasattr(args, 'files'):
        filelist = flatten(args.files)

    if args.subcommand == 'test':
        LOG.log(NORM_LEVEL, "Beginning feature extraction...")
        extract_feats(filelist, args.type, args.overwrite, skip_noisy=False)
        LOG.log(NORM_LEVEL, "Feature extraction finished, beginning classification...")
        classify_docs(filelist, args.classifier, conf)
        LOG.log(NORM_LEVEL, "Classification complete.")
    elif args.subcommand == 'train':
        extract_feats(filelist, args.type, args.overwrite, skip_noisy=True)
        train_classifier(filelist, args.out, conf)
    elif args.subcommand == 'eval':
        eval_files(filelist, args.output, args.csv)