#!/usr/bin/env python3
import os
import statistics
import abc
import logging
from argparse import ArgumentParser, ArgumentError
from copy import copy
from functools import partial
from collections import defaultdict, Counter
import glob
import sys
import pickle
import math
from io import TextIOBase
from multiprocessing.pool import Pool
from tempfile import NamedTemporaryFile

from subprocess import Popen, PIPE

import itertools

from config import *
from collections import Iterator
import re

# -------------------------------------------
# Set up logging
# -------------------------------------------
NORM_LEVEL = 1000
logging.addLevelName(NORM_LEVEL, 'NORMAL')
logging.basicConfig(level=NORM_LEVEL)
LOG = logging.getLogger()

# -------------------------------------------
# CONSTANTS
# -------------------------------------------
TYPE_FREKI = 'freki'
TYPE_TEXT  = 'text'


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

# -------------------------------------------
# Load the Wordlist if it is defined in the config.
# -------------------------------------------
class WordlistFile(set):
    def __init__(self, path):
        super().__init__()
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                self.add(line.strip())

WLF = WordlistFile(WORDLIST) if os.path.exists(WORDLIST) else None
# -------------------------------------------

class BBox(object):
    def __init__(self, tup):
        self.llx, self.lly, self.urx, self.ury = tup

class FrekiBlock(list):
    #doc_id=601.tetml page=1 block_id=1-2 bbox=147.5,711.0,465.4,723.0 1 1
    def __init__(self, lines=None, doc_id = None, page = None, block_id = None, bbox = None, start = 0, stop = 0):
        if lines is None:
            lines = []
        self.doc_id = doc_id
        self.page = page
        self.block_id = block_id
        self.bbox = bbox
        self.start = start
        self.stop = stop
        super().__init__(lines)




class FrekiReader(DocReader):
    def __init__(self, fh, lm=None):
        self.fh = fh
        self.lineno = 0


        self.block_id  = None
        self.bbox      = None
        self.startline = None
        self.stopline  = None

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
            self.avg_width   = statistics.mean(self.widths)
            if len(self.llxs) > 1:
                self.width_stdev   = statistics.stdev(self.widths)

        add_frekifeats(self)
        self.seek(0)

    def get_label(self, lineno):
        line = self.linedict[lineno]
        return line.label if line.label else 'O'

    def most_common_font(self):
        return self.fonts.most_common(1)[0][0]

    def most_common_size(self, font):
        return self.fontsizes[font].most_common(1)[0][0]

    @property
    def width(self):
        return self.bbox.urx - self.bbox.llx

    def __next__(self):
        data = self.fh.__next__()

        while data:
            if data.startswith('doc_id'):
                d = data.split()
                self.startline = int(d[-2])
                self.stopline  = int(d[-1])
                nd = {k:v for k, v in (t.split('=') for t in d[:-2])}
                self.bbox = BBox(float(i) for i in nd['bbox'].split(','))
                self.block_id = nd['block_id']

                # Create a new FrekiBlock object...
                fb = FrekiBlock(doc_id = nd['doc_id'], page=int(nd['page']),
                                block_id=nd['block_id'], bbox=self.bbox,
                                start=int(d[-2]), stop=int(d[-1]))
                self.block_dict[nd['block_id']] = fb
                self.cur_block = fb

            elif data.startswith('line='):
                preamble, text = re.search('(line=.*?):(.*)\n', data).groups()
                if 'tag=' in preamble:
                    lineno, tag, fonts = [l.split('=')[1] for l in preamble.split()]
                else:
                    lineno, fonts = [l.split('=')[1] for l in preamble.split()]
                    tag = None

                fonts = (tuple((font, float(size)) for font, size in  [f.split('-') for f in fonts.split(',')]))

                l = Line(text, int(lineno), fonts, label=tag)

                self.lineno = int(lineno)

                # Update the block dicts
                self.cur_block.append(l)
                self.block_ids[int(lineno)] = self.cur_block.block_id
                return l

            data = self.fh.__next__()

    def block_for_line(self, lineno):
        return self.block_dict[self.block_ids[lineno]]

class Line(str):
    def __new__(cls, seq='', lineno: int = 0, fonts: tuple=None, label: str = None):
        l = str.__new__(cls, seq)
        l.lineno = lineno
        l.fonts = fonts
        l.label = label
        return l

    def search(self, pattern, flags=0):
        return re.search(pattern, self, flags=flags)


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
        return 0 if self._total == 0 else self._unigrams[k1]/self._total

    def bigram_prob(self, k1, k2):
        num = self._bigrams[k1][k2]
        den = self._unigrams[k1]
        return 0 if den == 0 else num/den

    def trigram_prob(self, k1, k2, k3):
        num = self._trigrams[k1][k2][k3]
        den = self._bigrams[k1][k2]
        return 0 if den == 0 else num/den

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


def get_textfeats(line: Line, lm : NgramDict) -> dict:
    """
    Given a line as input, return the text-based features
    available for that line.
    """


    feats = {
        'has_langname': has_langname(line)
        ,'has_grams': has_grams(line)
        ,'has_parenthetical': has_parenthetical(line)
        ,'has_citation': has_citation(line)
        ,'has_asterisk': has_asterisk(line)
        ,'has_underscore' : has_underscore(line)
        ,'has_bracketing': has_bracketing(line)
        ,'has_quotation': has_quotation(line)
        ,'has_numbering': has_numbering(line)
        ,'has_leading_whitespace' : has_leading_whitespace(line)
        ,'high_oov_rate' : high_oov_rate(line)
        ,'med_oov_rate' : med_oov_rate(line)
        # Various language / unicode features
        ,'has_jpn' : has_japanese(line)
        ,'has_grk' : has_greek(line)
        ,'has_kor' : has_korean(line)
        ,'has_acc' : has_accented_latin(line)
        ,'has_cyr' : has_cyrillic(line)
        ,'has_dia' : has_diacritic(line)
        ,'has_unicode': has_unicode(line)
        # ,'has_year': has_year(line)
    }

    # if lm is not None:
    #     feats['looks_english'] = looks_english(line, lm)

    for word in split_words(line):
        if word:
            feats['word_{}'.format(word)] = 1
    return feats

def add_frekifeats(r: DocReader):

    for lineno in sorted(r.featdict.keys()):
        feats = {
            'is_indented': isindented(r, lineno)
            ,'is_first_page': is_first_page(r, lineno)
            ,'prev_line_same_block': prev_line_same_block(r, lineno)
            ,'next_line_same_block': next_line_same_block(r, lineno)
            ,'has_nonstandard_font' : has_nondefault_font(r, lineno)
            ,'has_smaller_font' : has_smaller_font(r, lineno)
            ,'has_larger_font' : has_larger_font(r, lineno)
        }

        r.featdict[lineno].update(feats)

def get_all_line_feats(featdict, lineno) -> dict:
    """
    Given a dictionary mapping lines to features, get
    a new feature dict that includes features for the
    current line, as well as n-1 and n-2 lines, and n+1.
    """

    cur_feats = featdict[lineno]
    prev_prev_feats = featdict.get(lineno - 2, {})
    prev_feats = featdict.get(lineno - 1, {})
    next_feats = featdict.get(lineno + 1, {})

    all_feats = copy(cur_feats)

    for prev_key in prev_prev_feats.keys():
        all_feats['prev_prev_'+prev_key] = prev_prev_feats[prev_key]
    for prev_key in prev_feats.keys():
        all_feats['prev_' + prev_key] = prev_feats[prev_key]
    for next_key in next_feats.keys():
        all_feats['next_' + next_key] = next_feats[next_key]

    return all_feats

def match_file_for_path(path):
    return os.path.join(MATCH_DIR, os.path.splitext(os.path.basename(path))[0] + '.txt')

def feat_file_for_path(path):
    return os.path.join(FEAT_DIR, os.path.splitext(os.path.basename(path))[0] + '.feats')

# -------------------------------------------
# Perform feature extraction.
# -------------------------------------------
def extract_feats(filelist, filetype, overwrite=False, skip_noisy=False):

    p = Pool()

    for path in filelist:
        p.apply_async(extract_feat_for_path, args=[path, overwrite, skip_noisy])

    p.close()
    p.join()

def extract_feat_for_path(path, overwrite=False, skip_noisy=False):
    feat_path = feat_file_for_path(path)

    path_rel = os.path.relpath(path, __file__)
    feat_rel = os.path.relpath(feat_path, __file__)

    # -------------------------------------------
    # Skip generating the text feature for this path
    # if it's already been generated and the user
    # has not asked to overwrite them.
    # -------------------------------------------
    if os.path.exists(feat_path) and (not overwrite):
        LOG.log(NORM_LEVEL, 'File "{}" already generated, skipping...'.format(feat_path))
        return

    LOG.log(NORM_LEVEL, 'Opening file "{}" for feature extraction to file "{}"...'.format(path_rel, feat_rel))

    os.makedirs(os.path.dirname(feat_path), exist_ok=True)
    with open(feat_path, 'w', encoding='utf-8') as train_f:

        with open(path, 'r', encoding='utf-8') as f:
            # r = FrekiReader(f) if filetype == TYPE_FREKI else TextReader(f)
            r = FrekiReader(f)

            # Now, let's iterate through again and extract features.
            for lineno in sorted(r.featdict.keys()):
                label = r.linedict[lineno].label.split('+')[0]

                # If the line is not labeled, assume it is "O"
                if label is None:
                    label = 'O'

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

def write_training_vector(featdict, label, out: TextIOBase=sys.stdout):
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



# -------------------------------------------
# FEATURES
# -------------------------------------------
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

def has_nondefault_font(r : FrekiReader, lineno: int):
    line = r.linedict[lineno]
    mcf  = r.most_common_font()
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
    return bool(line.search('[\u0300–\u036F]|[\u1AB0-\u1AFF]|[\u1DC0-\u1DFF]|[\u20D0-\u20FF]|[\uFE20-\uFE2F]', flags=re.UNICODE))

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

def med_oov_rate(line: Line):
    return 0.5 > oov_rate(line) > 0.2

def high_oov_rate(line: Line):
    return oov_rate(line) >= 0.5

def oov_rate(line: Line):
    if not WLF:
        return 0.0
    else:

        words = []
        for word in split_words(line):
            words.extend(clean_word(word.lower()))

        if len(words) <= 2:
            return 0.0

        oov_words = Counter([w in WLF for w in words])
        c_total = sum([v for v in oov_words.values()])

        if not c_total:
            return 0.0
        else:
            oov_rate = oov_words[False] / c_total
            return oov_rate



def looks_english(line: Line, lm : NgramDict):
    lp = lm.mean_logprob_sent(line)
    return lp > -5

langs = set([])
with open('langs.txt', 'r', encoding='utf-8') as f:
    for line in f:
        last_col = ' '.join(line.split()[3:])
        for langname in last_col.split(','):
            langname = langname.replace('[', '')
            if len(langname) >= 5:
                langs.add(langname.lower())

lang_re = re.compile('({})'.format('|'.join(langs), flags=re.I))

def has_langname(line: Line):
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
MALLET_BIN = os.path.join(MALLET_DIR, 'bin/mallet')
INFO_BIN   = os.path.join(MALLET_DIR, 'bin/classifier2info')


def label_sort(l):
    order = ['O', 'L', 'L-T', 'G', 'T', 'M']
    if l in order:
        return order.index(l)
    else:
        return float('inf')

def get_class_feats(classpath, limit=30):
    """
    List the top $limit most informative features for the
     given classifier.

    :param classpath: Path to the classifier
    :param limit: Number of features to list
    """
    p = Popen([INFO_BIN,
               '--classifier', classpath], stdout=PIPE)

    featdict = defaultdict(partial(defaultdict, float))

    label = None
    for line in p.stdout:
        line = line.decode('utf-8')
        if line.startswith('FEATURES FOR CLASS'):
            label = line.split()[-1]

        else:
            feat, weight = line.split()
            featdict[feat][label] = float(weight)

    vals = []
    defaults = []
    for feat in featdict.keys():
        for label, val in featdict[feat].items():
            if feat == '<default>':
                defaults.append((feat, label, val))
            else:
                vals.append((feat, label, val))


    defaults = sorted(defaults, key=lambda x: label_sort(x[1]))
    vals = sorted(vals, key=lambda x: abs(x[2]), reverse=True)[:limit]

    longest_featname = max([len(x[0]) for x in vals])
    longest_label    = max([len(x[1]) for x in vals]+[5])

    format_str = '{{:{}}}\t{{:{}}}\t{{:<5.6}}'.format(longest_featname, longest_label)

    print(format_str.format('feature', 'label', 'weight'))
    linesep = '-'*(longest_featname+longest_label+10)
    print(linesep)
    for d in defaults:
        print(format_str.format(*d))
    print(linesep)
    for val in vals:
        print(format_str.format(*val))

def combine_vectors(pathlist):
    # Create the training file.

    combined_f = NamedTemporaryFile(mode='w', encoding='utf-8', delete=False)

    # -------------------------------------------
    # 1) Combine all the instances in the files...
    # -------------------------------------------
    for path in pathlist:
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                combined_f.write(line)
    combined_f.close()
    return combined_f.name

def train_classifier(filelist, class_path):

    # First start by converting the filelist of
    feat_paths  = [feat_file_for_path(p) for p in filelist]


    if not feat_paths:
        LOG.critical("No text vector files were found.")

    combined_vector_path = combine_vectors(feat_paths)


    # -------------------------------------------
    # 2) Create the training vectors for mallet.
    # -------------------------------------------
    vector_path = convert_to_vectors(combined_vector_path)

    p = Popen([MALLET_BIN, 'train-classifier',
               '--trainer', 'MaxEntTrainer',
               '--input', vector_path,
               '--output-classifier', class_path])
    p.wait()

    get_class_feats(class_path)

    os.unlink(combined_vector_path)


def convert_to_vectors(path, prev_pipe_path=None, out_path = None):
    """
    Take an svm-light format file and return a temporary file
    converted to vectors.
    """
    if not out_path:
        vector_f = NamedTemporaryFile('w', delete=False)
        vector_f.close()
        vector_path = vector_f.name
    else:
        vector_path = out_path

    args = [MALLET_BIN, 'import-svmlight',
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
def classify_doc(path, classifier):
    classifications = []
    p = Popen([MALLET_BIN, 'classify-svmlight',
               '--classifier', classifier,
               '--input', path,
               '--output', '-'], stdout=PIPE)

    for line in p.stdout:
        scores = []
        for label, score in re.findall('(\S+)\s+([0-9\-\.E]+)', line.decode('utf-8')):
            scores.append((label, float(score)))

        classification = tuple(sorted(scores, key=lambda x: x[1], reverse=True))
        classifications.append(classification)

    return classifications

class SpanCounter(object):
    def __init__(self):
        self.last_gold = 'O'
        self.last_guess = 'O'

        self.guess_spans = set()
        self.gold_spans  = set()

        self.cur_guess_span = []
        self.cur_gold_span  = []


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
        return 2*(self.span_precision(exact)*self.span_recall(exact))/(self.span_precision(exact)+self.span_recall(exact))


    def partial_recall(self):
        """
        The partial span recall is calculated by the number of gold spans for which a match
        is found. (There could be multiple system spans which overlap with the gold span,
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
        The partial span precision is calculated by the number of system spans which do in
        fact overlap with a gold span.
        :return:
        """
        matches = 0

        for sys_start, sys_stop in [(s[0], s[-1]) for s in self.guess_spans]:
            for gold_start, gold_stop in [(s[0], s[-1]) for s in self.gold_spans]:
                if (gold_stop >= sys_start >= gold_start) or (gold_stop >= sys_stop >= gold_start):
                    matches += 1
                    break

        return matches / len(self.guess_spans) if self.guess_spans else 0


    def add_line(self, lineno, gold, guess):
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
        self.last_gold  = gold

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
        return [matches/sums if sums > 0 else 0 for matches, sums in zip(self._matches(), self._gold_sums())]

    def _labels(self):
        return sorted(set(self._matrix.keys()) | set([inner_key for outer_key in self._matrix.keys() for inner_key in self._matrix[outer_key].keys()]), key=label_sort)

    def recall(self, exclude=list()):
        num = sum(self._matches(exclude))
        den = sum(self._gold_sums(exclude))
        return num / den if den > 0 else 0

    def precision(self, exclude=list()):
        num = sum(self._matches(exclude))
        den = sum(self._guess_sums(exclude))
        return num / den if den > 0 else 0

    def prf(self, exclude=list()):
        return (self.precision(exclude), self.recall(exclude), self.f_measure(exclude))

    def f_measure(self, exclude=list()):
        return 2*(self.precision(exclude) * self.recall(exclude))/(self.precision(exclude)+self.recall(exclude))

    def _vals(self):
        return [[self._matrix[gold][label] for gold in self._labels()] for label in self._labels()]


    def matrix(self):
        print('\t COLS: Gold --- ROWS: Predicted')
        print('\t'.join(['']+['{:4}'.format(l) for l in self._labels()]))
        for label in self._labels():
            vals = [self._matrix[gold][label] for gold in self._labels()]
            matches = self._matrix[label][label]
            compares = sum(vals)
            precision = matches / compares if compares > 0 else 0
            print('\t'.join([label]+['{:4}'.format(v) for v in vals]+['{:.2f}'.format(precision)]))

        print('\t'.join(['']+['{:4.2f}'.format(r) for r in self._recalls()]))






def classify_docs(filelist, class_path):

    feat_paths = [feat_file_for_path(p) for p in filelist]
    if not feat_paths:
        LOG.critical("No text vector files were found.")
        sys.exit()

    sc = SpanCounter()

    for path, feat_path in zip(filelist, feat_paths):
        LOG.log(NORM_LEVEL, 'Classifying file "{}"'.format(path))
        classifications = classify_doc(feat_path, class_path)
        fr = FrekiReader(open(path, 'r', encoding='utf-8'))

        assert len(list(fr.featdict.keys())) == len(classifications)

        # -------------------------------------------
        # We want to get not only classification accuracy,
        # but also count how well non-O "spans" get
        # labeled.
        # -------------------------------------------

        for lineno, classification in zip(sorted(fr.featdict.keys()), classifications):
            line = fr.linedict[lineno]

            prediction = classification[0][0]

            raw_label = line.label
            if raw_label.startswith('*'):
                raw_label = raw_label.replace('*', '')

            gold = raw_label.split('+')[0]

            sc.add_line(lineno, gold, prediction)


    sc.matrix()
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
        raise ArgumentError("No Files found matching pattern.")
    else:
        return g

def lmpath(pathname):
    if not pathname.endswith('.lm'):
        raise ArgumentError
    else:
        return pathname

# -------------------------------------------

def split_words(sent):
    return [re.sub('[#:]', '', w.lower()) for w in re.split('[\.\-\s]', sent)]

def build_lm(filelist, filetype, outpath, label_dir):

    ngd = NgramDict()

    for path in filelist:

        with open(path, 'r', encoding='utf-8') as f:

            r = FrekiReader(f)
            for lineno in r.linedict.keys():

                # We're only interested in NON-IGT data...
                if lf.get(lineno) is None:
                    line = r.linedict[lineno]
                    for word in split_words(line):
                        if word:
                            for i, char in enumerate(word.lower()):
                                pc = '#' if i-1 < 0 else word[i-1]
                                nc = '#' if i >= len(word)-1 else word[i+1]
                                ngd.add(pc, char, nc)

    LOG.log(NORM_LEVEL, 'Writing out language model..')
    with open(outpath, 'wb') as f:
        pickle.dump(ngd, f)




if __name__ == '__main__':
    p = ArgumentParser()

    subparsers = p.add_subparsers(help='Valid subcommands', dest='subcommand')
    subparsers.required = True

    # -------------------------------------------
    # FEATURE EXTRACTION
    # -------------------------------------------
    extract_p = subparsers.add_parser('extract')

    extract_p.add_argument('--type', choices=[TYPE_FREKI, TYPE_TEXT], default=TYPE_FREKI)
    extract_p.add_argument('--lm', default=None)
    extract_p.add_argument('files', nargs='+', type=globfiles)

    # -------------------------------------------
    # TRAINING
    # -------------------------------------------
    train_p = subparsers.add_parser('train')
    train_p.add_argument('files', nargs='+', type=globfiles)
    train_p.add_argument('--type', choices=[TYPE_FREKI, TYPE_TEXT], default=TYPE_FREKI)
    train_p.add_argument('-f', '--overwrite', action='store_true', help='Overwrite text vectors.')
    train_p.add_argument('-o', '--out', required=True, help='Output path for the classifier.')

    # -------------------------------------------
    # TESTING
    # -------------------------------------------

    test_p = subparsers.add_parser('test')

    test_p.add_argument('--type', choices=[TYPE_FREKI, TYPE_TEXT], default=TYPE_FREKI)
    test_p.add_argument('-f','--overwrite', action='store_true', help='Overwrite text vectors')
    test_p.add_argument('--classifier', required=True)
    test_p.add_argument('files', nargs='+', type=globfiles)

    # -------------------------------------------
    # Build LM
    # -------------------------------------------
    lm_p = subparsers.add_parser('lm')

    lm_p.add_argument('--type', choices=[TYPE_FREKI, TYPE_TEXT], default=TYPE_FREKI)
    lm_p.add_argument('--labels', required=True)
    lm_p.add_argument('--out', required=True, type=lmpath)
    lm_p.add_argument('files', nargs='+', type=globfiles)

    # -------------------------------------------

    args = p.parse_args()

    filelist = flatten(args.files)

    if args.subcommand == 'test':
        extract_feats(filelist, args.type, args.overwrite, skip_noisy=False)
        classify_docs(filelist, args.classifier)
    elif args.subcommand == 'lm':
        build_lm(filelist, args.type, args.out, args.labels)
    elif args.subcommand == 'train':
        extract_feats(filelist, args.type, args.overwrite, skip_noisy=True)
        train_classifier(filelist, args.out)