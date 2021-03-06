import os
from collections import defaultdict
from configparser import ConfigParser, _UNSET, NoSectionError, NoOptionError
from functools import partial

MY_DIR = os.path.dirname(__file__)
def absdir(path):
    return os.path.abspath(os.path.join(MY_DIR, path))

# -------------------------------------------
# Subclass the config parser to be able to obtain
# options from the default config
# -------------------------------------------
def setpaths(conf, path):
    secs = ['paths', 'files']
    for sec in secs:
        if sec in conf.sections():
            for opt in conf[sec]:
                v = conf[sec][opt]
                conf.set(sec, opt, os.path.abspath(os.path.join(os.path.dirname(path), v)))

class PathRelativeConfigParser(ConfigParser):

    def __init__(self, *args, path=None, **kwargs):
        super().__init__(*args, **kwargs)
        setpaths(self, path)

    def read(self, filenames, encoding=None):
        super().read(filenames, encoding=encoding)
        if isinstance(filenames, str):
            setpaths(self, filenames)

    @classmethod
    def load(cls, filename):
        prcp = cls()
        prcp.read(filename)
        return prcp

    def get(self, section, option, *, raw=False, vars=None, fallback=_UNSET):
        if section not in self.sections():
            return fallback
        else:
            return super().get(section, option, raw=raw, vars=vars, fallback=fallback)


# -------------------------------------------
# The following options are concerned with various
# folders for holding temporary and debug files
# are.
# -------------------------------------------

# The directory in which to place the human readable feature files.
def FEAT_DIR(obj):
    return getattr(obj, 'feat_dir')

# Directory to the gold standard data for evaluation.
def GOLD_DIR(obj):
    return getattr(obj, 'gold_dir')

# Directory in which to place output classified files
def OUT_DIR(obj):
    return getattr(obj, 'classified_dir')

# Whether or not to output debugging information
def DEBUG_ON(obj):
    return getattr(obj, 'debug_on')

# The directory in which to store the information about the classifier feature
# weights, and raw labels
def DEBUG_DIR(obj):
    return getattr(obj, 'debug_dir')



# -------------------------------------------
# Path to various text files
# -------------------------------------------
# Large English language wordlist.
EN_WORDLIST = 'en_wordlist'

# List of gloss-line words extracted from ODIN-2.
# 1
GLS_WORDLIST = 'gls_wordlist'

# List of meta line words extracted from ODIN-2.1
MET_WORDLIST = 'met_wordlist'

# List of language names
LNG_NAMES = 'lng_names'

thresh_dict = {}
def get_thresh(config, var):
    global thresh_dict
    if var not in thresh_dict:
        thresh_dict[var] = config.getfloat('thresholds', var)
    return thresh_dict.get(var)

def HIGH_OOV_THRESH(config): return get_thresh(config, 'high_oov')
def MED_OOV_THRESH(config): return get_thresh(config, 'med_oov')

def HIGH_ISCORE_THRESH(config): return get_thresh(config, 'high_iscore')
def MED_ISCORE_THRESH(config): return get_thresh(config, 'med_iscore')
def LOW_ISCORE_THRESH(config): return get_thresh(config, 'low_iscore')


# -------------------------------------------
# Load the Wordlist if it is defined in the config.
# -------------------------------------------
class WordlistFile(set):
    def __init__(self, path):
        super().__init__()
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    self.add(line.split()[0])

USE_BI_LABELS = 'use_bi_labels'

# Some lines appear as combinations of labels, such as "L-G-T" for all
# three on a single line. If this is set to true, these types of
# combined labels are allowed. If set to false, only the first
# of the multiple labels will be used.
USE_MULTI_LABELS = 'use_multi_labels'

# "Flags" are additional information that is intended to be included in
# the information about the line, such as +AC (for Author Citation)
# or +LN (for Language Name). These are stripped out by default, as
# otherwise they would result in an explosion of labels.
STRIP_FLAGS = 'strip_flags'

# =============================================================================
# Feature selection.
#
# In this section, various features are defined and can be enabled or
# disabled by the user. Read the comments, as some definitions are constants
# and should not be edited.
# =============================================================================


# -------------------------------------------
# High-level features.
#
# Set these to True or False, depending
# on whether you want that feature set enabled
# or not.
# -------------------------------------------

# Use the freki-block based features
FREKI_FEATS_ENABLED = True

# Use the text-based features
TEXT_FEATS_ENABLED  = True

# -------------------------------------------
# These three features control whether the
# features are included for the previous line,
# the line before that (prev_prev), or the next
# line.
# -------------------------------------------
true_vals = set(['t','true','1','on','enabled'])

def getbool(args, k):
    val = args.get(k, False)
    return str(val).lower() in true_vals

def USE_PREV_LINE(args):
    return getbool(args, 'use_prev_line')
    # return args.getboolean('featuresets', 'use_prev_line')

def USE_PREV_PREV_LINE(args):
    return getbool(args, 'use_prev_prev_line')
    # return args.getboolean('featuresets', 'use_prev_prev_line')

def USE_NEXT_LINE(args):
    return getbool(args, 'use_next_line')
    # return args.getboolean('featuresets', 'use_next_line')

# -------------------------------------------
# FEATURE CONSTANTS
#
# Associating a variable with the text string used in the config file.
# -------------------------------------------
F_IS_INDENTED = 'is_indented'
F_IS_FIRST_PAGE = 'is_first_page'
F_PREV_LINE_SAME_BLOCK = 'prev_line_same_block'
F_NEXT_LINE_SAME_BLOCK = 'next_line_same_block'
F_HAS_NONSTANDARD_FONT = 'has_nonstandard_font'
F_HAS_SMALLER_FONT = 'has_smaller_font'
F_HAS_LARGER_FONT  = 'has_larger_font'

F_HIGH_ISCORE = 'f_high_iscore'
F_MED_ISCORE  = 'f_med_iscore'
F_LOW_ISCORE  = 'f_low_iscore'

# List of all the above
F_LIST = [F_IS_INDENTED, F_IS_FIRST_PAGE, F_PREV_LINE_SAME_BLOCK, F_NEXT_LINE_SAME_BLOCK, F_HAS_NONSTANDARD_FONT, F_HAS_SMALLER_FONT, F_HAS_LARGER_FONT, F_HIGH_ISCORE, F_MED_ISCORE, F_LOW_ISCORE]

T_PREV_TAG = 'prev_tag'
T_BASIC = 'words'
T_HAS_LANGNAME = 'has_langname'
T_HAS_GRAMS = 'has_grams'
T_HAS_PARENTHETICAL = 'has_parenthetical'
T_HAS_CITATION = 'has_citation'
T_HAS_ASTERISK = 'has_asterisk'
T_HAS_UNDERSCORE = 'has_underscore'
T_HAS_BRACKETING = 'has_bracketing'
T_HAS_QUOTATION = 'has_quotation'
T_HAS_NUMBERING = 'has_numbering'
T_HAS_LEADING_WHITESPACE = 'has_leading_whitespace'
T_HIGH_OOV_RATE = 'high_oov_rate'
T_MED_OOV_RATE = 'med_oov_rate'
T_HIGH_GLS_OOV_RATE = 'high_gls_oov'
T_HIGH_MET_OOV_RATE = 'high_met_oov'
T_MED_GLS_OOV_RATE = 'med_gls_oov'
T_HAS_JPN = 'has_jpn'
T_HAS_GRK = 'has_grk'
T_HAS_KOR = 'has_kor'
T_HAS_CYR = 'has_cyr'
T_HAS_ACC = 'has_acc_lat'
T_HAS_DIA = 'has_dia'
T_HAS_UNI = 'has_uni'
T_HAS_YEAR = 'has_year'

T_LIST = [T_BASIC, T_HAS_LANGNAME, T_HAS_GRAMS, T_HAS_PARENTHETICAL, T_HAS_CITATION, T_HAS_ASTERISK, T_HAS_UNDERSCORE, T_HAS_BRACKETING,
          T_HAS_QUOTATION, T_HAS_NUMBERING, T_HAS_LEADING_WHITESPACE, T_HIGH_OOV_RATE, T_MED_OOV_RATE, T_HIGH_GLS_OOV_RATE, T_MED_GLS_OOV_RATE,
          T_HIGH_GLS_OOV_RATE, T_MED_GLS_OOV_RATE, T_HIGH_MET_OOV_RATE,
          T_HAS_JPN, T_HAS_GRK, T_HAS_KOR, T_HAS_CYR, T_HAS_ACC, T_HAS_DIA, T_HAS_UNI, T_HAS_YEAR]

# =============================================================================
# EDIT THIS SECTION
# =============================================================================

# -------------------------------------------
# Now, to enable/disable a particular feature,
# just comment out the line the feature is
# contained on.
# -------------------------------------------

def enabled_feats(config: ConfigParser, section, featlist):
    enabled = set([])
    for feat in featlist:
        if config.has_option(section, feat):
            b = config.getboolean(section, feat)
            if b:
                enabled.add(feat)
    return enabled

_enabled_freki_feats = None
_enabled_text_feats = None

def ENABLED_FREKI_FEATS(config: ConfigParser):
    global _enabled_freki_feats
    if _enabled_freki_feats is None:
        _enabled_freki_feats = enabled_feats(config, 'freki_features', F_LIST)
    return _enabled_freki_feats


def ENABLED_TEXT_FEATS(config: ConfigParser):
    global _enabled_text_feats
    if _enabled_text_feats is None:
        _enabled_text_feats = enabled_feats(config, 'text_features', T_LIST)
    return _enabled_text_feats


# =============================================================================
# Regular Expressions
#
# These are multiline expressions that were initially used for IGT detection.
#
# These are currently unused, but could be included to fire for lines which
# find themselves contained in such a regex.
# =============================================================================

REGEXES = '''
\s*(\()\d*\).*\n
\s*.*\n
\s*\[`'"].*\n
~
\s*(\()\d*\)\s\w\..*\n
\s*.*\n
\s\[`'"].*\n
~
\s*(\(\d)*\)\s*\(.*\n
\s.*\n
\s*.*\n
\s\[`'"].*\n
~
\s*(\()\d*\).*\n
\s*\w\..*\n
\s*.*\n
\s\[`'"].*\n
~
\s*\w.\s*.*\n
\s*.*\n
\s*\[`'"].*\n
~
\s*\w\)\s*.*\n
\s*.*\n
\s*\[`'"].*\n
~
\s*(\()\w*\).*\n
\s*.*\n
\s*\[`'"].*\n
~
//added 02-03-2005
\s*\d.*.*\n
\s*.*\n
\s*\[`'"].*\n
~
\s*(\()\d*\).*\n
.*\n
\s*.*\n
\s*\[`'"].*\n
~'''