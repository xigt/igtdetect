#!/usr/bin/env python3
import statistics, abc, logging
from argparse import ArgumentParser, ArgumentTypeError
from bz2 import BZ2File
from copy import copy, deepcopy
from functools import partial
from collections import defaultdict, Counter, namedtuple
import glob, sys
from io import TextIOBase
from logging import StreamHandler
from multiprocessing.pool import Pool
from tempfile import NamedTemporaryFile
import pickle

# -------------------------------------------
# Import scikit-learn modules
# -------------------------------------------
import numpy as np
from multiprocessing import Lock

import time

from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import LogisticRegression

from subprocess import Popen, PIPE

from env import *
import re

# -------------------------------------------
# Set up logging
# -------------------------------------------
from freki.serialize import FrekiDoc, FrekiLine, FrekiBlock, FrekiFont

NORM_LEVEL = 1000
logging.addLevelName(NORM_LEVEL, 'NORMAL')
ERR_LOG = logging.getLogger(name='ERRORS')
STD_LOG = logging.getLogger(name='STD')

stdhandler = StreamHandler(sys.stdout)
errhandler = StreamHandler(sys.stderr)

stdhandler.setLevel(NORM_LEVEL)
errhandler.setLevel(NORM_LEVEL)

STD_LOG.addHandler(stdhandler)
ERR_LOG.addHandler(errhandler)

# -------------------------------------------
# CONSTANTS
# -------------------------------------------
TYPE_FREKI = 'freki'
TYPE_TEXT = 'text'




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

def safe_mode(iterable):
    """
    Like taking the mode of the most common item
    in a sequence, but pick between one of the two
    most frequent if there is no unique mode.

    :param iterable:
    :return:
    """
    return sorted(Counter(iterable).items(),
                  reverse=True,
                  key=lambda x: x[1])[0][0]

class FrekiInfo(object):
    """
    Store a few document-wide pieces of info for
    FrekiDocs, so that they don't need to be
    recalculated each time.
    """
    def __init__(self, fonts=None, llxs=None):
        """
        :type font: FrekiFont
        :type llx: float
        """
        self.def_font = safe_mode(fonts)
        self.llx = safe_mode(llxs)

class FrekiAnalysis(object):
    """
    Wrap the features, labels, and
    full document in an object to output
    from the feature extraction code.
    """
    def __init__(self,
                 measurements: list,
                 labels: list,
                 doc: FrekiDoc):
        self.doc = doc
        self.measurements = measurements
        self.labels = labels



def get_textfeats(line: FrekiLine) -> dict:
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
                feats['word_{}'.format(word)] = True

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
    checkfeat(T_HIGH_MET_OOV_RATE, high_met_oov_rate)

    return feats


def get_frekifeats(line: FrekiLine, fi: FrekiInfo):
    feats = {}

    # Use this function to check the
    # feature constant name against the
    # list of enabled features, and trigger
    # the appropriate function if it's enabled.
    def checkfeat(name, func):
        nonlocal feats
        if name in ENABLED_FREKI_FEATS(conf):
            feats[name] = func(line, fi)

    # Apply each feature if it is enabled
    checkfeat(F_IS_INDENTED, isindented)
    checkfeat(F_IS_FIRST_PAGE, is_first_page)
    checkfeat(F_PREV_LINE_SAME_BLOCK, prev_line_same_block)
    checkfeat(F_NEXT_LINE_SAME_BLOCK, next_line_same_block)
    checkfeat(F_HAS_NONSTANDARD_FONT, has_nondefault_font)
    checkfeat(F_HAS_SMALLER_FONT, has_smaller_font)
    checkfeat(F_HAS_LARGER_FONT, has_larger_font)

    return feats


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
    return os.path.join(FEAT_DIR(args), _path_rename(path, '_feats.txt'))


def get_raw_classification_path(path):
    return os.path.join(os.path.join(DEBUG_DIR(args), 'raw_classifications'),
                        _path_rename(path, '_classifications.txt'))


classified_suffix = '_classified.txt'
detected_suffix = '_detected.txt'


def get_classified_path(path, classified_dir):
    return os.path.join(classified_dir, _path_rename(path, classified_suffix))


def get_detected_path(path, detected_dir):
    return os.path.join(detected_dir, _path_rename(path, detected_suffix))


def get_gold_for_classified(path):
    return os.path.join(GOLD_DIR(args), os.path.basename(path).replace(classified_suffix, '.txt'))


def get_weight_path(path):
    return os.path.join(DEBUG_DIR(args), _path_rename(path, '_weights.txt'))


class ClassifierWrapper(object):
    """
    This class serves as a wrapper for saving the classifier.
    This consists of the learner itself ("c"), the object
    responsible for keeping track of the feature-to-index mappings,
    and which features to use (via the "feat_selector")
    """

    def __init__(self):
        self.c = LogisticRegression()
        self.dv = DictVectorizer(dtype=bool)
        self.feat_selector = SelectKBest(chi2, 'all')
        self.labels = []


# -------------------------------------------
# Perform feature extraction.
# -------------------------------------------
def extract_feats(filelist, cw, overwrite=False, skip_noisy=True):
    """
    Perform feature extraction over a list of files.

    Call extract_feat_for_path() in parallel for a speed boost.
    """

    # -------------------------------------------
    # Build a list of measurements from the files.
    # This will be a list of dicts, where each list item
    # represents a line, and each dictionary entry represents
    # a feature:value pair.
    # -------------------------------------------
    measurements = []
    labels = []

    p = Pool()
    l = Lock()

    def callback(result):
        l.acquire()
        measurements.extend(result.measurements)
        labels.extend(result.labels)
        l.release()

    for path in filelist:
        # p.apply_async(extract_feats_for_path, args=[path, overwrite, skip_noisy], callback=callback)
        callback(extract_feats_for_path(path, overwrite=overwrite, skip_noisy=skip_noisy))

    p.close()
    p.join()

    # -------------------------------------------
    # Remove the "B/I" from labels if that is disabled.
    # -------------------------------------------
    if not USE_BI_LABELS(conf):
        for i, label in enumerate(labels):
            if label[0:2] in ['B-', 'I-']:
                labels[i] = label[2:]

    # -------------------------------------------
    # Turn the extracted feature dict into vectors for sklearn.
    # -------------------------------------------
    return measurements, labels


def load_feats(path):
    """
    Load features from a saved svm-lite like file

    :param path:
    :return:
    """
    labels, measures = [], []
    with open(path, 'r') as feat_f:
        for line in feat_f:
            line_measures = {}
            data = line.split()
            label = data[0]
            for feat, value in [pair.split(':') for pair in data[1:]]:
                line_measures[feat] = bool(value)

            labels.append(label)
            measures.append(line_measures)
    return labels, measures


def extract_feats_for_path(path, overwrite=False, skip_noisy=True):
    """
    Perform feature extraction for a single file.

    The output files are in svmlight format, namely:

        LABEL   feature_1:value_1   feature_2:value_2 ...etc

    The "skip_noisy" parameter is intended for training data that
    was created automatically, and for which the labels were mapped,
    but seem unlikely to be correct. Such noisy labels are preceded by
    an asterisk.

    :rtype: FrekiAnalysis
    """
    feat_path = get_feat_path(path)

    path_rel = os.path.abspath(path)
    feat_rel = os.path.abspath(feat_path)

    # -------------------------------------------
    # Create a list of measurements, and associated labels.
    # -------------------------------------------
    measurements = []
    labels = []

    # Read in the freki document, whether or
    # not the features need to be reprocessed.
    fd = FrekiDoc.read(path)


    # -------------------------------------------
    # Skip generating the text feature for this path
    # if it's already been generated and the user
    # has not asked to overwrite them.
    # -------------------------------------------
    if os.path.exists(feat_path) and (not overwrite):
        ERR_LOG.warn('File "{}" already generated, not regenerating (use -f to force)...'.format(feat_path))

        path_labels, path_measurements = load_feats(feat_path)
        labels.extend(path_labels)
        measurements.extend(path_measurements)

    else:

        #     return

        ERR_LOG.info('Opening file "{}" for feature extraction to file "{}"...'.format(path_rel, feat_rel))

        os.makedirs(os.path.dirname(feat_path), exist_ok=True)
        with open(feat_path, 'w', encoding='utf-8') as train_f:

            fi = FrekiInfo(fonts=fd.fonts(),
                           llxs=fd.llxs())

            # 1) Start by getting the features for this
            #    particular line...
            feat_dict = {}
            for line in fd.lines():
                feat_dict[line.lineno] = get_textfeats(line)
                feat_dict[line.lineno].update(get_frekifeats(line, fi))

            # 2) Now, add the prev/next line data as necessary
            for line in fd.lines():
                # Skip noisy (preceded with '*') tagged lines
                label = line.tag
                if label.startswith('*'):
                    if skip_noisy: continue
                    else: label = label.replace('*', '')

                # Strip flags and multiple tags if
                # needed
                label = fix_label_flags_multi(label)

                if 'O' not in label:
                    if line.span_id == line.doc.get_line(line.lineno-1).span_id:
                        bi_status = 'I'
                    else:
                        bi_status = 'B'

                    label = '{}-{}'.format(bi_status, label)


                all_feats = get_all_line_feats(feat_dict, line.lineno)
                measurements.append(all_feats)
                labels.append(label)

                write_training_vector(all_feats, label, train_f)

    return FrekiAnalysis(measurements, labels, fd)


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
def isindented(line: FrekiLine, fi: FrekiInfo):
    # Is the line's indenting greater than that
    # for the overall document.
    return line.block.llx > fi.llx

def has_smaller_font(line: FrekiLine, fi: FrekiInfo):
    for font in line.fonts:
        if font.f_size < fi.def_font.f_size:
            return True
    return False

def has_larger_font(line: FrekiLine, fi: FrekiInfo):
    for font in line.fonts:
        if font.f_size > fi.def_font.f_size:
            return True
    return False

def has_nondefault_font(line: FrekiLine, fi: FrekiInfo):
    # Get the "default" font
    return bool(set(line.fonts) - set([fi.def_font]))


def has_grams(line: str):
    return bool(line.search('|'.join(GRAM_LIST), flags=re.I) or line.search('|'.join(CASED_GRAM_LIST)))


def has_parenthetical(line: str):
    return bool(line.search('\(.*\)'))


# Cover four-digit numbers from 1800--2019
year_str = '(?:1[8-9][0-9][0-9]|20[0-1][0-9])'


def has_citation(line: FrekiLine):
    return bool(line.search('\([^,]+, {}\)'.format(year_str)))


def has_year(line: FrekiLine):
    return bool(line.search(year_str))


def has_asterisk(line: FrekiLine):
    return '*' in line


def has_underscore(line: FrekiLine):
    return '_' in line


def has_bracketing(line: FrekiLine):
    return bool(line.search('\[.*\]'))


def has_numbering(line: FrekiLine):
    return bool(line.search('^\s*\(?[0-9a-z]+[\)\.]'))


def has_leading_whitespace(line: FrekiLine):
    return bool(line.search('^\s+'))


# -------------------------------------------
# Various Unicode Ranges
# -------------------------------------------

def has_cyrillic(line: FrekiLine):
    return bool(line.search('[\u0400-\u04FF]', flags=re.UNICODE))


def has_diacritic(line: FrekiLine):
    return bool(line.search('[\u0300–\u036F]|[\u1AB0-\u1AFF]|[\u1DC0-\u1DFF]|[\u20D0-\u20FF]|[\uFE20-\uFE2F]',
                            flags=re.UNICODE))


def has_greek(line: FrekiLine):
    return bool(line.search('[\u0370-\u03FF]|[\u1F00-\u1FFF]', flags=re.UNICODE))


def has_japanese(line: FrekiLine):
    has_kanji = bool(line.search('[\u4E00-\u9FBF]', flags=re.U))
    has_hiragana = bool(line.search('[\u3040-\u309F]', flags=re.U))
    has_katakana = bool(line.search('[\u30A0-\u30FF]', flags=re.U))
    return has_kanji or has_hiragana or has_katakana


def has_accented_latin(line: FrekiLine):
    return bool(line.search('[\u00C0-\u00FF]', flags=re.U))


def has_korean(line: FrekiLine):
    return bool(line.search('[\uAC00-\uD7A3]', flags=re.U))


def has_unicode(line: FrekiLine):
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

def med_en_oov_rate(line: FrekiLine):
    return HIGH_OOV_THRESH(conf) > oov_rate(en_wl, line) > MED_OOV_THRESH(conf)


def high_en_oov_rate(line: FrekiLine):
    return oov_rate(en_wl, line) >= HIGH_OOV_THRESH(conf)


def high_gls_oov_rate(line: FrekiLine):
    return oov_rate(gls_wl, line) > HIGH_OOV_THRESH(conf)


def high_met_oov_rate(line: FrekiLine):
    return oov_rate(gls_wl, line) > HIGH_OOV_THRESH(conf)


def oov_rate(wl: WordlistFile, line: FrekiLine):
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


def has_langname(line: FrekiLine):
    init_langnames()
    return bool(line.search(lang_re))


def has_quotation(line: FrekiLine):
    """ Return true if the line in question surrounds more than one word in quotes """
    return bool(line.search('[\'\"‘`“]\S+\s+.+[\'\"’”]'))


def is_first_page(line: FrekiLine, *args):
    return line.block.page == 1

def same_block(cur_line, other_line):
    if other_line is None:
        return False
    else:
        return cur_line.block.block_id == other_line.block.block_id

def prev_line_same_block(line: FrekiLine, *args):
    prev_line = line.doc.get_line(line.lineno-1)
    return same_block(line, prev_line)

def next_line_same_block(line: FrekiLine, *args):
    next_line = line.doc.get_line(line.lineno+1)
    return same_block(line, next_line)

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
        linesep = '-' * (longest_featname + longest_label + 10) + '\n'
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
def train_classifier(cw: ClassifierWrapper, measurements, labels, classifier_path=None, debug_on=False,
                     max_features=None, **kwargs):
    """
    Train the classifier based on the input files in filelist.

    :param filelist: List of files to train the classifier based upon.
    :param classifier_path:
    :return:
    """

    if max_features is not None:
        max_features = int(max_features)
    else:
        max_features = -1

    # Transform the measurements (list of dicts)
    # to a feature vector.
    ERR_LOG.info("Converting features from dictionary to matrix...")
    vec = cw.dv.fit_transform(measurements)

    # Perform feature selection to limit to the top
    # 1,000 features.

    feat_select = cw.feat_selector
    if max_features > 0:
        n_features = min(vec.shape[-1], max_features)
        ERR_LOG.info('Selecting features to the top "{}" best'.format(n_features))
        feat_select = SelectKBest(chi2, n_features)

    cw.feat_selector = feat_select

    feat_select.fit(vec, labels)

    # Print out the pruned features...
    selected_feats = feat_select.get_support()  # Boolean array of the selected features.
    feat_names = np.array(cw.dv.get_feature_names())
    ERR_LOG.info('Reduced feature count from {} to {}'.format(vec.shape[-1], np.count_nonzero(selected_feats)))

    # Reduce the [sample, feature] matrix down to the selected features...
    X = feat_select.transform(vec)

    # Start the training time...
    start_training = time.time()

    # -------------------------------------------
    # Train the classifier...
    # -------------------------------------------
    ERR_LOG.log(NORM_LEVEL, "Training classifier...")
    if debug_on:
        cw.c.set_params(verbose=True)

    cw.c.set_params(n_jobs=8)  # Use all cpu cores
    cw.c.fit(X, labels)
    ERR_LOG.log(NORM_LEVEL, 'Training finished in "{}" seconds.'.format(int(time.time() - start_training)))

    # -------------------------------------------
    # Print out the feature weights...
    # -------------------------------------------
    debug_dir = kwargs.get('debug_dir', './debug')
    if debug_on:
        classinfo_dir = os.path.join(debug_dir, 'classifier_info')
        os.makedirs(classinfo_dir, exist_ok=True)
        classinfo_path = os.path.join(classinfo_dir,
                                      os.path.splitext(os.path.basename(classifier_path))[0] + '_weights.txt')
        classinfo_f = open(classinfo_path, 'w', encoding='utf-8')

        # Get the default weights for each class...
        defaults = {c: cw.c.intercept_[i] for i, c in enumerate(cw.labels)}

        # Next, get the weights for each feature.
        weights = {(f, c): cw.c.coef_[i][j] for j, f in enumerate(feat_names[selected_feats]) for i, c in
                   enumerate(cw.labels)}
        sorted_weights = sorted(weights.items(), reverse=True, key=lambda x: x[1])

        format_str = '{:<40}{:<10.6}{:>10}\n'
        classinfo_f.write(format_str.format('feat_name', 'weight', 'cls_name'))

        for clsname in sorted(cw.labels):
            classinfo_f.write(format_str.format('DEFAULT', defaults[clsname], clsname))
        classinfo_f.write('- ' * 30 + '\n')

        for w_tup, weight in sorted_weights:
            feat_name, clsname = w_tup
            classinfo_f.write(format_str.format(feat_name, weight, clsname))

        classinfo_f.close()

    # -------------------------------------------
    # Add the labels to the classifier wrapper
    # -------------------------------------------
    cw.labels = np.unique(labels)

    ERR_LOG.log(NORM_LEVEL, 'Writing classifier out to "{}"'.format(classifier_path))
    with BZ2File(classifier_path, 'wb') as f:
        pickle.dump(cw, f)

def assign_spans(fd):
    """
    Assign span IDs to a document without them,
    assuming only that a span is a contiguous
    block of non-'O' labels.

    :param fd: Document to assign span_ids to
    :type fd: FrekiDoc
    """
    num_spans = 0
    last_tag = 'O'
    for line in fd.lines():
        if 'O' not in line.tag:

            # Increment if the last tag
            # was 'O'
            if 'O' in last_tag:
                num_spans += 1

            line.span_id = 's{}'.format(num_spans)

        last_tag = line.tag

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

    os.unlink(temp_in.name)  # Delete the temporary output file with the replaced labels

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

def classify_docs(filelist, classifier_path=None, overwrite=None, debug_on=False,
                  classified_dir=None, detected_dir=None,
                  **kwargs):
    feat_paths = [get_feat_path(p) for p in filelist]

    if not feat_paths:
        ERR_LOG.critical("No text vector files were found.")
        sys.exit()

    # Load the saved classifier...
    STD_LOG.log(NORM_LEVEL, "Loading saved classifier...")
    cf = BZ2File(classifier_path, 'r')
    cw = pickle.load(cf)
    cf.close()
    assert isinstance(cw, ClassifierWrapper), type(cw)
    STD_LOG.log(NORM_LEVEL, "Classifier Loaded.")

    for path, feat_path in zip(filelist, feat_paths):

        # -------------------------------------------
        # Open the output classification path
        # -------------------------------------------
        fa = extract_feats_for_path(path, overwrite=overwrite, skip_noisy=True)

        if not fa.measurements:
            continue

        # Transform the feature vector for use with the classifier...
        vec = cw.dv.transform(fa.measurements)

        # And reduce the features according to the selected features...
        selected_vec = cw.feat_selector.transform(vec)

        # Now retrieve the classification probability distribution
        classifications = cw.c.predict_proba(selected_vec)

        # Ensure that the number of lines in the feature file
        # matches the number of lines returned by the classifier
        num_lines = len(fa.measurements)
        num_classifications = len(classifications)

        if num_lines != num_classifications:
            ERR_LOG.critical(
                "The number of lines ({}) does not match the number of classifications ({}). Skipping file {}".format(
                    num_lines, num_classifications, path))
            continue

        # -------------------------------------------
        # Get ready to write the classified IGT instances out.
        # The "classified_dir" is for the full files, with "O"
        # lines, the "detected_dir" is only for contiguous, non-O lines.
        # -------------------------------------------
        if classified_dir:
            os.makedirs(classified_dir, exist_ok=True)
            classified_f = open(get_classified_path(path, classified_dir), 'w', encoding='utf-8')

        if detected_dir:
            os.makedirs(detected_dir, exist_ok=True)
            detected_f = open(get_detected_path(path, detected_dir), 'w', encoding='utf-8')

        # -------------------------------------------

        # This file will contain the raw labelings from the classifier.
        if debug_on:
            os.makedirs(os.path.dirname(get_raw_classification_path(path)), exist_ok=True)
            ERR_LOG.log(NORM_LEVEL, 'Writing out raw classifications "{}"'.format(get_raw_classification_path(path)))
            raw_classification_f = open(get_raw_classification_path(path), 'w')

        # -------------------------------------------
        # Iterate through the returned classifications
        # and assign them to the lines in the test file.
        #
        # Optionally, write out the raw classification distribution.
        # -------------------------------------------
        cur_span = []
        total_detected = 0

        old_lines = list(fa.doc.lines())

        for line, probs in zip(old_lines, classifications):

            classification = list(zip(cw.labels, probs))

            # Write the line number and classification probabilities to the debug file.
            if debug_on:
                raw_classification_f.write('{}:'.format(line.lineno))
                for label, weight in classification:
                    raw_classification_f.write('\t{}  {:.3e}'.format(label, weight))
                raw_classification_f.write('\n')
                raw_classification_f.flush()

            # -------------------------------------------
            # Get the best label
            # -------------------------------------------
            best_label = sorted(classification, key=lambda x: x[1], reverse=True)[0][0]

            # -------------------------------------------
            # If we are using B+I labels, make sure to
            # strip them off before writing out the file.
            # -------------------------------------------
            if best_label[0:2] in set(['I-', 'B-']):
                best_label = best_label[2:]

            # Set the label for the line in the working block
            # before potentially writing it out.
            fl = FrekiLine(line,
                           tag=best_label,
                           line=line.lineno)
            fl.fonts = line.fonts

            fa.doc.set_line(line.lineno, fl)


            if best_label == 'O' and detected_dir:
                if cur_span:
                    detected_f.write('\n'.join(cur_span))
                    detected_f.write('\n\n')
                    cur_span = []
                    total_detected += 1
            else:
                cur_span.append('{:<8}{}'.format(best_label, fl))

        # Write out the classified file.
        if classified_dir:
            assign_spans(fa.doc)
            classified_f.write(str(fa.doc))
            classified_f.close()

        if detected_dir:
            detected_f.close()
            if total_detected == 0:
                os.unlink(get_detected_path(path, detected_dir))

        if debug_on:
            raw_classification_f.close()


def eval_files(filelist, out_path, csv, gold_dir=None, **kwargs):
    """
    Given a list of target files, evaluate them against
    the files given in the gold dir.

    If the gold dir does not exist, or does not contain
    the specified file, make sure to log an error.
    """
    # Set up the output stream
    if out_path is None:
        out_f = sys.stdout
    else:
        out_f = open(out_path, 'w')

    if not os.path.exists(gold_dir):
        ERR_LOG.critical('The gold file directory "{}" is missing or is unavailable.'.format(GOLD_DIR(conf)))
        sys.exit(2)
    elif not os.path.isdir(gold_dir):
        ERR_LOG.error('The gold file directory "{}" appears to be a file, not a directory.'.format(GOLD_DIR(conf)))
        sys.exit(2)

    # Create the counter to iterate over all the files.
    sc = SpanCounter()
    for eval_path in filelist:
        gold_path = get_gold_for_classified(eval_path)
        if not os.path.exists(gold_path):
            ERR_LOG.warning('No corresponding gold file was found for the evaluation file "{}"'.format(eval_path))
        else:
            eval_file(eval_path, gold_path, sc=sc)

    # Now, write out the sc results.
    delimiter = '\t'
    if csv:
        delimiter = ','
    out_f.write(sc.matrix() + '\n')

    out_f.write('----- Labels -----\n')
    out_f.write(' Classifiation Acc: {:.2f}\n'.format(sc.precision()))
    out_f.write('       Non-O P/R/F: {}\n\n'.format(delimiter.join(['{:.2f}'.format(x) for x in sc.prf(['O'])])))
    out_f.write('----- Spans ------\n')
    out_f.write(
        '  Exact-span P/R/F: {}\n'.format(delimiter.join(['{:.2f}'.format(x) for x in sc.span_prf(exact=True)])))
    out_f.write(
        'Partial-span P/R/F: {}\n'.format(delimiter.join(['{:.2f}'.format(x) for x in sc.span_prf(exact=False)])))

    out_f.close()

def fix_label_flags_multi(label):
    if STRIP_FLAGS(conf):
        label = label.split('+')[0]

    if not USE_MULTI_LABELS(conf):
        label = label.split('-')[0]

    return label


def eval_file(eval_path, gold_path, sc=None, outstream=sys.stdout):
    """
    Look for the filename that matches the specified file
    """
    eval_fd = FrekiDoc.read(eval_path)
    gold_fd = FrekiDoc.read(gold_path)

    if len(eval_fd) != len(gold_fd):
        ERR_LOG.error(
            'The evaluation file "{}" and the gold file "{}" appear to have a different number of lines. Evaluation aborted.'.format(
                eval_path, gold_path))
    else:
        if sc is None:
            sc = SpanCounter()

        for line in eval_fd.lines():
            eval_label = fix_label_flags_multi(eval_fd.get_line(line.lineno).tag)
            gold_label = fix_label_flags_multi(gold_fd.get_line(line.lineno).tag)

            sc.add_line(line.lineno, gold_label, eval_label)

        return sc
        outstream.write(sc.matrix() + '\n\n')
        outstream.write(' Classifiation Acc: {:.2f}\n'.format(sc.precision()))
        outstream.write('       Non-O P/R/F: {}\n'.format(','.join(['{:.2f}'.format(x) for x in sc.prf(['O'])])))
        outstream.write(
            '  Exact-span P/R/F: {}\n'.format(','.join(['{:.2f}'.format(x) for x in sc.span_prf(exact=True)])))
        outstream.write(
            'Partial-span P/R/F: {}\n'.format(','.join(['{:.2f}'.format(x) for x in sc.span_prf(exact=False)])))


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
        raise ArgumentTypeError(
            'No files found matching pattern "{}".\nCheck that the path is valid and that containing directories exist.'.format(
                pathname))
    else:
        paths = []
        for path in g:
            if os.path.isdir(path):
                paths.extend([os.path.join(path, p) for p in os.listdir(path)])
            else:
                paths.append(path)
        return paths


# -------------------------------------------

#
def split_words(sent):
    return [re.sub('[#:]', '', w.lower()) for w in re.split('[\.\-\s]', sent)]


# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':

    # -------------------------------------------
    # Set up the main argument parser (for subcommands)
    # -------------------------------------------
    main_parser = ArgumentParser()

    # -------------------------------------------
    # Set up common parser options shared by
    # the subcommands
    # -------------------------------------------
    common_parser = ArgumentParser(add_help=False)
    common_parser.add_argument('-v', '--verbose', action='store_true', help='Enable verbosity.')
    common_parser.add_argument('-c', '--config', help='Alternate config file.')
    common_parser.add_argument('-f', '--force', dest='overwrite', action='store_true',
                               help='Overwrite previously generated feature files.')

    # -------------------------------------------
    # Load the default config file.
    # -------------------------------------------
    conf = None
    def_path = os.path.join(os.path.dirname(__file__), './defaults.ini')
    if os.path.exists(def_path):
        conf = PathRelativeConfigParser()
        conf.read(def_path)

    for sec in conf.sections():
        common_parser.set_defaults(**conf[sec])

    # Determine whether the classifier
    classifier_required = os.path.exists(def_path)

    # -------------------------------------------
    # Set up a common parser to inherit for the functions
    # that require the classifier to be specified
    # -------------------------------------------
    tt_parser = ArgumentParser(add_help=False)
    tt_parser.add_argument('--classifier-path', required=classifier_required,
                           help='Path to the saved classifier model.')
    tt_parser.add_argument('--in-files', help='Path to input files used for training or testing.', type=globfiles)

    # Parser for combining evaluation arguments.
    ev_parser = ArgumentParser(add_help=False)
    ev_parser.add_argument('-o', '--output', dest='out_path', help='Output path to write result. [Default: stdout]')
    ev_parser.add_argument('--csv', help='Format the output as CSV.')
    ev_parser.add_argument('--eval-files', help='Files to evaluate against', required=True, type=globfiles)
    ev_parser.add_argument('--gold-dir', default=conf.get('paths', 'gold_dir'))

    # -------------------------------------------
    # Set up the subcommands
    # -------------------------------------------

    subparsers = main_parser.add_subparsers(help='Valid subcommands', dest='subcommand')
    subparsers.required = True

    # -------------------------------------------
    # TRAINING
    # -------------------------------------------
    train_p = subparsers.add_parser('train', parents=[common_parser, tt_parser])
    train_p.add_argument('--use-bi-labels', type=int, default=1)
    train_p.add_argument('--max-features', type=int, default=-1)

    # -------------------------------------------
    # TESTING
    # -------------------------------------------
    test_p = subparsers.add_parser('test', parents=[common_parser, tt_parser])

    # -------------------------------------------
    # EVAL
    # -------------------------------------------
    eval_p = subparsers.add_parser('eval', parents=[common_parser, ev_parser])

    # -------------------------------------------
    # TESTEVAL
    # -------------------------------------------
    testeval_p = subparsers.add_parser('testeval', parents=[common_parser, tt_parser, ev_parser])
    # -------------------------------------------

    # -------------------------------------------
    # TRAINTESTEVAL
    # -------------------------------------------
    traintesteval_p = subparsers.add_parser('traintesteval', parents=[common_parser, tt_parser, ev_parser])

    global args
    args = main_parser.parse_args()

    # -------------------------------------------
    # Read in the config file, if provided. Otherwise
    # the default config parser class will use the provided
    # defaults.
    # -------------------------------------------
    if args.config:
        if not os.path.exists(args.config):
            ERR_LOG.critical('The config file "{}" could not be found.'.format(args.config))
            sys.exit(2)
        else:
            alt_c = PathRelativeConfigParser()
            alt_c.read(args.config)
            for sec in alt_c.sections():
                for key in alt_c[sec].keys():
                    setattr(args, key, alt_c[sec][key])

    # -------------------------------------------
    # Debug
    # -------------------------------------------

    if DEBUG_ON(args):
        os.makedirs(DEBUG_DIR(args), exist_ok=True)

    # -------------------------------------------
    # Load wordlist files for performance if testing or training
    # -------------------------------------------
    global en_wl, gls_wl, met_wl
    if args.subcommand in ['test', 'train', 'testeval', 'traintesteval']:
        en_wl = EN_WL(conf)
        gls_wl = GL_WL(conf)
        met_wl = MT_WL(conf)

    filelist = flatten(getattr(args, 'in_files', []))
    eval_list = flatten(getattr(args, 'eval_files', []))


    def train(fl):
        cw = ClassifierWrapper()
        measurements, labels = extract_feats(fl, cw, overwrite=args.overwrite, skip_noisy=True)
        train_classifier(cw, measurements, labels, **vars(args))


    def test(fl):
        ERR_LOG.log(NORM_LEVEL, "Beginning classification...")
        classify_docs(fl, **vars(args))
        ERR_LOG.log(NORM_LEVEL, "Classification complete.")


    def eval(fl):
        ERR_LOG.log(NORM_LEVEL, "Beginning evaluation...")
        eval_files(fl, **vars(args))


    def testeval(fl):
        test(fl)
        classified_paths = [get_classified_path(p, getattr(args, 'classified_dir')) for p in fl]
        eval(classified_paths)


    def traintesteval(fl, ep):
        train(fl)
        testeval(ep)


    # Switch between the commands
    if args.subcommand == 'train':
        train(filelist)
    elif args.subcommand == 'test':
        test(filelist)
    elif args.subcommand == 'eval':
        eval(eval_list)
    elif args.subcommand == 'testeval':
        testeval(filelist)
    elif args.subcommand == 'traintesteval':
        traintesteval(filelist, eval_list)
