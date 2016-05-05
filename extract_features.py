#!/usr/bin/env python3
import os
import statistics
from functools import partial
import logging
from argparse import ArgumentParser, ArgumentError
import abc
from copy import copy
from collections import defaultdict
import glob

# -------------------------------------------
# Set up logging
# -------------------------------------------
import pickle

import sys

import math

NORM_LEVEL = 1000
logging.addLevelName(NORM_LEVEL, 'NORMAL')
logging.basicConfig(level=logging.WARN)
LOG = logging.getLogger()

from nltk.classify import maxent

from collections import Iterator

import re

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

class BBox(object):
    def __init__(self, tup):
        self.llx, self.lly, self.urx, self.ury = tup

class FrekiReader(DocReader):
    def __init__(self, fh, lm=None):
        self.fh = fh
        self.lineno = 0
        self.featdict = {}
        self.linedict = {}

        self.block_id  = None
        self.bbox      = None
        self.startline = None
        self.stopline  = None

        self.llxs = {}
        self.widths = {}
        self.lines_to_blocks = {}

        for line in self:
            self.llxs[self.lineno] = self.bbox.llx
            self.widths[self.lineno] = self.width
            self.featdict[self.lineno] = get_textfeats(line, lm)
            self.lines_to_blocks[self.lineno] = self.block_id
            self.linedict[self.lineno] = line

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
                lineno, fonts, line = re.search('line=([0-9]+) fonts=(\S+?):(.*)\n', data).groups()
                self.lineno = int(lineno)
                return line

            data = self.fh.__next__()


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
    def __init__(self, path):
        self.labeldict = {}
        with open(path, 'r') as f:
            for line in f:
                lineno, label = line.strip().split(':')
                label = label.split('+')[0]
                if label == 'B':
                    continue
                self.labeldict[int(lineno)] = label

    def get(self, lineno, default=None):
        return self.labeldict.get(lineno, default)

    def __getitem__(self, lineno):
        return self.get(lineno)

def get_textfeats(line: str, lm : NgramDict):
    feats = {
        'has_langname': has_langname(line)
        ,'has_grams': has_grams(line)
        ,'has_parenthetical': has_parenthetical(line)
        ,'has_citation': has_citation(line)
        ,'has_asterisk': has_asterisk(line)
        ,'has_bracketing': has_bracketing(line)
        ,'has_quotation': has_quotation(line)
        ,'has_numbering': has_numbering(line)
        ,'has_leading_whitespace' : has_leading_whitespace(line)
        ,'has_unicode' : has_unicode(line)
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
                 'is_indented': isindented(r, lineno),
                 'prev_line_same_block': prev_line_same_block(r, lineno),
                 'next_line_same_block': next_line_same_block(r, lineno),
                 }

    r.featdict[lineno].update(feats)

def get_all_line_feats(featdict, lineno):

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

def match_file_for_path(label_dir, path):
    return os.path.join(label_dir, os.path.splitext(os.path.basename(path))[0] + '.matches')

# -------------------------------------------
def train_classifier(filelist, filetype, outpath, label_dir, lm=None):

    training_toks = []

    LOG.log(NORM_LEVEL, "Extracting features for training...")

    train_f = open(outpath, 'w')

    # -------------------------------------------
    # Load the lm if it exists...
    # -------------------------------------------
    if lm is not None:
        with open(lm, 'rb') as f:
            lm = pickle.load(f)

    for path in filelist:
        match_path = match_file_for_path(label_dir, path)
        if not os.path.exists(match_path):
            continue
            LOG.warn('No label file found for "{}", skipping'.format(path))

        LOG.log(NORM_LEVEL, 'Opening file "{}" for training...'.format(path))
        lf = LabelFile(match_path)

        with open(path, 'r', encoding='utf-8') as f:
            # r = FrekiReader(f) if filetype == TYPE_FREKI else TextReader(f)
            r = FrekiReader(f, lm)

            # Now, let's iterate through again and extract features.
            for lineno in sorted(r.featdict.keys()):
                label = lf.get(lineno)
                if label is None:
                    label = 'O'

                all_feats = get_all_line_feats(r.featdict, lineno)
                write_training_vector(all_feats, label, train_f)

                training_toks.append((all_feats, label))

    train_f.close()

    # LOG.log(NORM_LEVEL, 'Finished extracting features, beginning training.')
    #
    # mec = maxent.MaxentClassifier.train(training_toks)
    #
    # mec.show_most_informative_features(n=100)
    #
    # LOG.log(NORM_LEVEL, "Writing out model...")
    # with open(outpath, 'wb') as f:
    #     pickle.dump(mec, f)


def write_training_vector(featdict, label, out=sys.stdout):
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
    return bool(re.search('|'.join(GRAM_LIST), line, flags=re.I) or re.search('|'.join(CASED_GRAM_LIST), line))

def has_parenthetical(line: str):
    return bool(re.search('\(.*\)', line))

# Cover four-digit numbers from 1800--2019
year_str = '(?:1[8-9][0-9][0-9]|20[0-1][0-9])'

def has_citation(line: str):
    return bool(re.search('\([^,]+, {}\)'.format(year_str), line))

def has_year(line: str):
    return bool(re.search(year_str, line))

def has_asterisk(line: str):
    return '*' in line

def has_bracketing(line: str):
    return bool(re.search('\[.*\]', line))

def has_numbering(line: str):
    return bool(re.search('^\s*\(?[0-9a-z]+[\)\.]', line))

def has_leading_whitespace(line: str):
    return bool(re.search('^\s+', line))

def has_unicode(line: str):
    return bool(re.search('[\u00a2-\uFFFF]', line, flags=re.UNICODE))

def looks_english(line: str, lm : NgramDict):
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

def has_langname(line: str):
    return bool(re.search(lang_re, line))

def has_quotation(line: str):
    """ Return true if the line in question surrounds more than one word in quotes """
    return bool(re.search('[\'\"‘`“]\S+\s+.+[\'\"’”]', line))

def prev_line_same_block(r: FrekiReader, lineno: int):
    return r.lines_to_blocks.get(lineno - 1) == r.lines_to_blocks.get(lineno)

def next_line_same_block(r: FrekiReader, lineno: int):
    return r.lines_to_blocks.get(lineno + 1) == r.lines_to_blocks.get(lineno)

# -------------------------------------------
# DO the classification
# -------------------------------------------

def classify_docs(filelist, filetype, classifier):
    with open(classifier, 'rb') as f:
        mec = pickle.load(f)
        assert  isinstance(mec, maxent.MaxentClassifier)

        for path in filelist:
            with open(path, 'r', encoding='utf-8') as fp:
                r = FrekiReader(fp)

                for lineno in r.featdict:
                    feats = get_all_line_feats(r.featdict, lineno)
                    label = mec.classify(feats)
                    if label != 'O':
                        print(label, r.linedict[lineno])

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
    # TRAINING
    # -------------------------------------------
    train_p = subparsers.add_parser('train')

    train_p.add_argument('--type', choices=[TYPE_FREKI, TYPE_TEXT], default=TYPE_FREKI)
    train_p.add_argument('--labels', default=None, required=True)
    train_p.add_argument('--lm', default=None)
    train_p.add_argument('--out', default=None, required=True)
    train_p.add_argument('files', nargs='+', type=globfiles)

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

    if args.subcommand == 'train':
        train_classifier(filelist, args.type, args.out, args.labels, args.lm)
    elif args.subcommand == 'test':
        classify_docs(filelist, args.type, args.classifier)
    elif args.subcommand == 'lm':
        build_lm(filelist, args.type, args.out, args.labels)