# -------------------------------------------
# Various Directories for where Files are Stored.
# -------------------------------------------
MATCH_DIR = '/Users/rgeorgi/Documents/code/igt-detect/4-match'
FEAT_DIR  = '/Users/rgeorgi/Documents/code/igt-detect/5-feats'

# -------------------------------------------
# Path to wordlist file
# -------------------------------------------
WORDLIST = '/Users/rgeorgi/Documents/code/igt-detect/ngrams/wordsEn.txt'

# -------------------------------------------
# Dependent file libraries.
# -------------------------------------------
XIGT_DIR  = '/NLP_TOOLS/uwcl/xigt/xigt-1.1.0/'
ODIN_UTIL_DIR = '../odinutils'

# -------------------------------------------
# Where is Mallet
# -------------------------------------------
MALLET_DIR = '/Users/rgeorgi/Documents/code/mallet-2.0.7'



# These grams will be searched for case-insensitive.
GRAM_LIST = ['1SG', '1PL', '1SM',
             '2SG', '2P', '2SM',
             '3SG', '3REFL', '3SGP', '3SM', '3P']

# These grams will be searched for case sensitive.
CASED_GRAM_LIST = ['POSS',
                   'ACC','NOM', 'DAT', 'ERG', 'AOR', 'ABS', 'OBL', 'DUAL', 'REFL',
                   'NEG', 'TOP',
                   'FUT', 'PROG', 'PRES', 'PASS']

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
# FEATURE CONSTANTS
#
# Do not remove any of these.
# -------------------------------------------
F_IS_INDENTED = 'is_indented'
F_IS_FIRST_PAGE = 'is_first_page'
F_PREV_LINE_SAME_BLOCK = 'prev_line_same_block'
F_NEXT_LINE_SAME_BLOCK = 'next_line_same_block'
F_HAS_NONSTANDARD_FONT = 'has_nonstandard_font'
F_HAS_SMALLER_FONT = 'has_smaller_font'
F_HAS_LARGER_FONT  = 'has_larger_font'

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
T_HAS_JPN = 'has_jpn'
T_HAS_GRK = 'has_grk'
T_HAS_KOR = 'has_kor'
T_HAS_CYR = 'has_cyr'
T_HAS_DIA = 'has_dia'
T_HAS_UNI = 'has_uni'
T_HAS_YEAR = 'has_year'
T_LOOKS_ENGLISH = 'looks_english'


# -------------------------------------------
# Now, to enable/disable a particular feature,
# just comment out the line the feature is
# contained on.
# -------------------------------------------
FREKI_FEATS = [
    F_IS_INDENTED            # Is the line in question indented more than usual?
    ,F_IS_FIRST_PAGE         # Is the line contained on the first page?
    ,F_PREV_LINE_SAME_BLOCK  # Was the previous line in the same block as this one?
    ,F_NEXT_LINE_SAME_BLOCK  # Is the next line in the same block as this one?
    ,F_HAS_NONSTANDARD_FONT  # Is there a different font than usual on this line?
    ,F_HAS_SMALLER_FONT      # Is there a smaller font than usual on this line?
    ,F_HAS_LARGER_FONT       # Is there a larger font than usual on this line?
]

TEXT_FEATS = [
    T_HAS_LANGNAME # Does this line contain a language name?
    ,T_HAS_PARENTHETICAL # Does this line contain a parenthetical?
    ,T_HAS_CITATION # Does this line contain an author, year citation?
    ,T_HAS_ASTERISK # Does this line contain an asterisk (ungrammatical)
    ,T_HAS_UNDERSCORE # Does this line contain an underscore?
    ,T_HAS_BRACKETING # Does this line have any bracketing?
    ,T_HAS_QUOTATION # Does this line contain an element wrapped in quotes?
    ,T_HAS_NUMBERING # Does this line have a leading numbering?
    ,T_HAS_LEADING_WHITESPACE # Does this line have leading whitespace?
    ,T_HIGH_OOV_RATE # Is the OOV rate per word very high?
    ,T_MED_OOV_RATE  # Is the OOV rate per word moderately high?
    ,T_HAS_JPN # Characters in the Japanese unicode range
    ,T_HAS_GRK # Characters in the Greek unicode range
    ,T_HAS_KOR # Characters in the Korean unicode range
    ,T_HAS_CYR # Characters in the Cyrillic unicode range
    ,T_HAS_DIA # Diacritic characters
    ,T_HAS_UNI # ANY of the above unicode ranges
    ,T_HAS_YEAR # Is there a four-digit year (from 1800-20XX)
    # ,T_LOOKS_ENGLISH # Is the logprob of characters below a threshold of Eng similarity
]

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