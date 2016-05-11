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
from tempfile import NamedTemporaryFile

from subprocess import Popen, PIPE

from config import *
from collections import Iterator
import re

# -------------------------------------------
# Set up logging
# -------------------------------------------
NORM_LEVEL = 1000
logging.addLevelName(NORM_LEVEL, 'NORMAL')
logging.basicConfig(level=logging.WARN)
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
        self.lines_to_blocks = {}

        # -------------------------------------------
        # Font Counters
        # -------------------------------------------
        self.fonts = Counter()
        self.fontsizes = defaultdict(Counter)



        for line in self:
            self.llxs[self.lineno] = self.bbox.llx
            self.widths[self.lineno] = self.width
            self.featdict[self.lineno] = get_textfeats(line, lm)
            self.lines_to_blocks[self.lineno] = self.block_id
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

            elif data.startswith('line='):
                lineno, fonts, text = re.search('line=([0-9]+) fonts=(\S+?):(.*)\n', data).groups()

                fonts = (tuple((font, float(size)) for font, size in  [f.split('-') for f in fonts.split(',')]))

                l = Line(text, int(lineno), fonts)
                self.lineno = int(lineno)
                return l

            data = self.fh.__next__()

class Line(object):
    def __init__(self, text: str='', lineno: int = 0, fonts: tuple=None):
        if fonts is None:
            fonts = tuple()

        self.text = text
        self.fonts = fonts
        self.lineno = lineno

    def search(self, pattern, flags=0):
        return re.search(pattern, self.text, flags=flags)

    def __iter__(self):
        return self.text.__iter__()


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


class LabelFile(object):
    """
    Class to parse the files containing the label supervision.
    """
    def __init__(self, path):
        self.labeldict = {}
        with open(path, 'r') as f:
            for line in f:
                lineno, label = line.strip().split(':')
                label = re.split('[\-\+]', label)[0]
                if label == 'B':
                    continue
                self.labeldict[int(lineno)] = label

    def get(self, lineno, default=None):
        return self.labeldict.get(lineno, default)

    def __getitem__(self, lineno):
        return self.get(lineno)

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
        ,'has_unicode' : has_unicode(line)
        ,'high_oov_rate' : high_oov_rate(line)
        ,'med_oov_rate' : med_oov_rate(line)
        # ,'has_year': has_year(line)
    }

    # if lm is not None:
    #     feats['looks_english'] = looks_english(line, lm)

    # for word in split_words(line):
    #     if word:
    #         feats['word_{}'.format(word)] = 1
    return feats

def add_frekifeats(r: DocReader):

    for lineno in sorted(r.featdict.keys()):
        feats = {
            'is_indented': isindented(r, lineno)
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
    return os.path.join(MATCH_DIR, os.path.splitext(os.path.basename(path))[0] + '.matches')

def feat_file_for_path(path):
    return os.path.join(FEAT_DIR, os.path.splitext(os.path.basename(path))[0] + '.feats')

# -------------------------------------------
# Perform feature extraction.
# -------------------------------------------
def extract_feats(filelist, filetype, lm=None):

    LOG.log(NORM_LEVEL, "Extracting features for training...")

    # -------------------------------------------
    # Load the lm if it exists...
    # -------------------------------------------
    if lm is not None:
        with open(lm, 'rb') as f:
            lm = pickle.load(f)

    for path in filelist:
        match_path = match_file_for_path(path)
        feat_path  = feat_file_for_path(path)

        if not os.path.exists(match_path):
            continue
            LOG.warn('No label file found for "{}", skipping'.format(path))

        LOG.log(NORM_LEVEL, 'Opening file "{}" for training...'.format(path))
        lf = LabelFile(match_path)

        os.makedirs(os.path.dirname(feat_path), exist_ok=True)
        with open(feat_path, 'w', encoding='utf-8') as train_f:

            with open(path, 'r', encoding='utf-8') as f:
                # r = FrekiReader(f) if filetype == TYPE_FREKI else TextReader(f)
                r = FrekiReader(f, lm)

                # Now, let's iterate through again and extract features.
                for lineno in sorted(r.featdict.keys()):
                    label = lf.get(lineno)


                    # If the line is not labeled, assume it is "O"
                    if label is None:
                        label = 'O'

                    # If the label contains an asterisk, that means
                    # it is very noisy. Don't use it for either "O" or any
                    # in-IGT label.
                    if label.startswith('*'):
                        continue

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

# These grams will be searched for case-insensitive.
GRAM_LIST = ['1SG', '1PL', '1SM',
             '2SG', '2P', '2SM',
             '3SG', '3REFL', '3SGP', '3SM', '3P']

# These grams will be searched for case sensitive.
CASED_GRAM_LIST = ['POSS',
                   'ACC','NOM', 'DAT', 'ERG', 'AOR', 'ABS', 'OBL', 'DUAL', 'REFL',
                   'NEG', 'TOP',
                   'FUT', 'PROG', 'PRES', 'PASS']

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

def has_unicode(line: Line):
    return bool(line.search('[\u00a2-\uFFFF]', flags=re.UNICODE))

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
        for word in split_words(line.text):
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

def prev_line_same_block(r: FrekiReader, lineno: int):
    return r.lines_to_blocks.get(lineno - 1) == r.lines_to_blocks.get(lineno)

def next_line_same_block(r: FrekiReader, lineno: int):
    return r.lines_to_blocks.get(lineno + 1) == r.lines_to_blocks.get(lineno)

# -------------------------------------------
# TRAIN THE CLASSIFIER
# -------------------------------------------
MALLET_BIN = os.path.join(MALLET_DIR, 'bin/mallet')
INFO_BIN   = os.path.join(MALLET_DIR, 'bin/classifier2info')

def get_class_feats(classpath, limit=25):
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
    for feat in featdict.keys():
        if feat == '<default>':
            continue
        for label, val in featdict[feat].items():
            vals.append((feat, label, val))

    vals = sorted(vals, key=lambda x: abs(x[2]), reverse=True)[:limit]
    for val in vals:
        print(val)



def train_classifier(filelist, class_out):
    # Create the training file.
    combined_f = NamedTemporaryFile(mode='w', encoding='utf-8', delete=False)

    # -------------------------------------------
    # 1) Combine all the instances in the files...
    # -------------------------------------------
    for path in filelist:
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                combined_f.write(line)
    combined_f.close()

    # -------------------------------------------
    # 2) Create the training vectors for mallet.
    # -------------------------------------------
    vector_path = convert_to_vectors(path, strip_labels=True)

    p = Popen([MALLET_BIN, 'train-classifier',
               '--trainer', 'MaxEntTrainer',
               '--input', vector_path,
               '--output-classifier', class_out])
    p.wait()

    get_class_feats(class_out)

    os.unlink(combined_f.name)
    os.unlink(vector_path)


def convert_to_vectors(path, strip_labels = False):
    """
    Take an svm-light format file and return a temporary file
    converted to vectors.
    """
    vector_f = NamedTemporaryFile(mode='w', encoding='utf-8', delete=False)
    vector_f.close()
    p = Popen([MALLET_BIN, 'import-svmlight',
               '--input', path,
               '--output', vector_f.name])
    p.wait()

    return vector_f.name

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




def classify_docs(filelist, filetype, classifier):

    for path in filelist:
        # vector_path = convert_to_vectors(path, strip_labels=True)
        # print(vector_path)

        classifications = classify_doc(path, classifier)
        match_path = match_file_for_path(path)
        lf = LabelFile(match_path)

        compares = 0
        matches  = 0

        non_o_golds = 0
        non_o_matches = 0

        for i, classification in enumerate(classifications):
            prediction = classification[0][0]
            gold       = lf.get(i+1, 'O')

            if gold != 'O':
                non_o_golds += 1

            if prediction == gold:
                matches += 1
                if prediction != 'O':
                    non_o_matches += 1
            compares += 1

        print(non_o_matches / non_o_golds)
        # os.unlink(vector_path)


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
    return glob.glob(pathname)

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
            match_path = match_file_for_path(label_dir, path)
            if not os.path.exists(match_path):
                continue
                LOG.warn('No label file found for "{}", skipping'.format(path))
            else:
                lf = LabelFile(match_path)
                LOG.log(NORM_LEVEL, "Opening file {} for lm building...".format(path))

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
    train_p.add_argument('--out', required=True, help='Output path for the classifier.')

    # -------------------------------------------
    # TESTING
    # -------------------------------------------

    test_p = subparsers.add_parser('test')

    test_p.add_argument('--type', choices=[TYPE_FREKI, TYPE_TEXT], default=TYPE_FREKI)
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

    if args.subcommand == 'extract':
        extract_feats(filelist, args.type, args.lm)
    elif args.subcommand == 'test':
        classify_docs(filelist, args.type, args.classifier)
    elif args.subcommand == 'lm':
        build_lm(filelist, args.type, args.out, args.labels)
    elif args.subcommand == 'train':
        train_classifier(filelist, args.out)