#!/usr/bin/env python3
import os, re, sys
from argparse import ArgumentParser
from collections import defaultdict
import importlib

# -------------------------------------------
# Import the config file
# -------------------------------------------


from config import *

# -------------------------------------------
# Now import the dependent other modules.
# -------------------------------------------
sys.path.insert(0, XIGT_DIR)
sys.path.insert(0, ODIN_UTIL_DIR)

from odintxt import odin_blocks
from odinclean import merge_strings, clean_corpus
from xigt.importers.odin import make_igt
from xigt.model import XigtCorpus
from xigt import xigtpath
# -------------------------------------------



# -------------------------------------------
# Levenshtein Distance
# -------------------------------------------
def levenshtein(a,b):
    "Calculates the Levenshtein distance between a and b."
    n, m = len(a), len(b)
    if n > m:
        # Make sure n <= m, to use O(min(n,m)) space
        a,b = b,a
        n,m = m,n

    current = range(n+1)
    for i in range(1,m+1):
        previous, current = current, [i]+[0]*n
        for j in range(1,n+1):
            add, delete = previous[j]+1, current[j-1]+1
            change = previous[j-1]
            if a[j-1] != b[i-1]:
                change = change + 1
            current[j] = min(add, delete, change)

    return current[n]
# -------------------------------------------


def print_pairs(check_block, freki_block, out_f):
    check_max = max([len(l) for l in check_block])
    freki_max = max([len(l) for l in freki_block])

    format_str = '{{:<{}}}{:10}{{:<{}}}'.format(check_max, '', freki_max)
    header_str = '{{:^{}}}{:10}{{:^{}}}'.format(check_max, '', freki_max)
    out_f.write(header_str.format('CHECK', 'FREKI')+'\n')
    out_f.write(header_str.format('-'*check_max, '-'*freki_max)+'\n')

    for i in range(max(len(check_block), len(freki_block))):
        check_line = '' if i >= len(check_block) else check_block[i]
        freki_line = '' if i >= len(freki_block) else freki_block[i]
        dual = format_str.format(check_line, freki_line)
        out_f.write(dual+'\n')
    out_f.write('\n')
    out_f.flush()

# -------------------------------------------

checkfiles = {}
frekifiles = {}
rawfiles   = {}

def gather_check_instances(data):
    instances = []

    cur_instance = ''
    in_data = False
    for line in data:
        if 'doc_id' in line:
            in_data = True
        if in_data:
            if not line.strip() and cur_instance:
                instances.append(cur_instance)
                cur_instance = ''
            else:
                cur_instance += line
    return instances


# -------------------------------------------
# Find the rawfiles
# -------------------------------------------
for root, dir, files in os.walk(OLD_TXT):
    for file in files:
        fullpath = os.path.join(root, file)
        basename = os.path.basename(fullpath)
        if re.search('^[0-9]+\.txt', basename):
            filenum = int(os.path.splitext(basename)[0])
            rawfiles[filenum] = fullpath


# -------------------------------------------
# Find all the checkfiles.
# -------------------------------------------
for root, dir, files in os.walk(CHECK_DIR):
    for file in files:
        fullpath = os.path.join(root, file)
        if fullpath.endswith('.check'):
            filenum = int(os.path.basename(os.path.splitext(fullpath)[0]))
            checkfiles[filenum] = fullpath

# -------------------------------------------
# Now, find all the new FREKI files that match.
# -------------------------------------------
for f in os.listdir(FREKI_DIR):
    filenum = int(os.path.splitext(f)[0])
    frekifiles[filenum] = os.path.join(FREKI_DIR, f)

# =============================================================================
# REMAPPING
# =============================================================================

class FrekiBlock(object):
    def __init__(self, str):
        lines = str.split('\n')
        header_info   = lines[0].split()
        self.doc_id   = header_info[0].split('=')[1]
        self.block_id = header_info[1].split('=')[1]
        self.bbox     = header_info[2].split('=')[1]
        self.line_range = tuple(int(i) for i in header_info[3:])
        self.labels = {}

        self.lines={}
        for line in lines[1:]:
            if not line.strip():
                continue
            lineno, linetxt = re.search('line=([0-9]+):(.*)', line).groups()
            self.lines[int(lineno)] = linetxt

    def textlines(self):
        return [self.lines[i] for i in sorted(self.lines.keys())]

    def full_line_range(self):
        return tuple(i for i in sorted(self.lines.keys()))

    def __str__(self):
        ret_str = ''
        for line in sorted(self.lines.keys()):
            ret_str += 'line={}:{}\n'.format(line, self.lines[line])
        return ret_str

def find_most_similar(d):
    """
    Given a dictionary of the original line numbers, and a list of the freki line
    numbers, and the levenshtein distances; find the best fit for a continuous block.
    """

    orig_linenos  = sorted(d.keys(), key=lambda x: min([int(i) for i in x.split()]))
    freki_linenos = sorted(list(d.items())[0][1].keys())

    smallest_sum = float('inf')
    best_match = None
    for freki_lineno in freki_linenos:

        lev_total = 0

        for i, orig_lineno in enumerate(orig_linenos):
            orig_lev_dict = d[orig_lineno]
            if freki_lineno+i not in orig_lev_dict:
                lev_total += float('inf')
                break
            lev_total += orig_lev_dict[freki_lineno+i]

        if lev_total < smallest_sum:
            smallest_sum = lev_total
            best_match = tuple(freki_lineno+i for i, j in enumerate(orig_linenos))

    return orig_linenos, best_match

# -------------------------------------------
# Now, iterate over the pairs.
# -------------------------------------------
def renum_checks(check_instances, freki_blocks, match_dict = None, match_f=sys.stdout):

    if match_dict is None:
        match_dict = {'matches':0, 'compares':0}

    last_freki_index = 0

    # -------------------------------------------
    # Convert Freki Blocks
    # -------------------------------------------
    fbs = []
    fbmap = {}
    for freki_block in freki_blocks:
        fb = FrekiBlock(freki_block)
        fbs.append(fb)
        for line in fb.lines:
            fbmap[line] = fb

    # -------------------------------------------
    # Convert to ODIN Blocks
    # -------------------------------------------
    odin_lines = []
    for check_instance in check_instances:
        for lineno in check_instance.split('\n'):
            odin_lines.append(lineno)

    # -------------------------------------------
    # Convert the blocks to IGT instances...
    # -------------------------------------------
    xc = XigtCorpus()
    blocks = odin_blocks(odin_lines)
    for odin_block in blocks:
        igt = make_igt(odin_block, {'keep_headers':True, 'replacement_char':'\uFFFD'})
        igt.id = 'i'+re.sub('\s', '-', odin_block['line_range'])
        xc.append(igt)




    # -------------------------------------------
    # Clean the IGT instances...
    # -------------------------------------------
    clean_corpus(xc)

    # -------------------------------------------
    # Now, step through the cleaned lines and try
    # to match them up with the new data
    # -------------------------------------------
    for igt in xc:
        linemap = defaultdict(lambda: {})
        clean_tier = igt[1]


        for clean_item in clean_tier:
            lines = clean_item.attributes['line']
            clean_text = clean_item.value()
            for fb in fbs:
                for lineno in fb.lines:
                    line_text = fb.lines[lineno]

                    # Remove the whitespace and compare the lines
                    line_text  = re.sub('\s','', line_text)
                    clean_text = re.sub('\s', '', clean_text)

                    linemap[lines][lineno] = levenshtein(line_text, clean_text)


        # Find the mapping between line numbers from the original
        # annotation to the new, freki numbers.
        orig_lines, freki_lines = find_most_similar(linemap)


        # match_f.write('igt_id={}\n'.format(igt.id))
        for i, orig_line in enumerate(orig_lines):
            orig_indices = ','.join(orig_line.split())
            frek_index   = freki_lines[i]

            orig_item = xigtpath.find(clean_tier, '//item[@line="{}"]'.format(orig_line))

            match_f.write('{1}:{0}\n'.format(orig_item.attributes['tag'], frek_index))
        # match_f.write('\n')


if __name__ == '__main__':
    p = ArgumentParser()
    p.add_argument('checkfile')

    args = p.parse_args()

    filenum = int(os.path.splitext(os.path.basename(args.checkfile))[0])



    check_data = open(checkfiles[filenum], 'r', encoding='latin-1').readlines()
    freki_data = open(frekifiles[filenum], 'r', encoding='utf-8').readlines()

    with open(os.path.join(MATCH_DIR, str(filenum)+'.matches'), 'w', encoding='utf-8') as f:

        check_instances = gather_check_instances(check_data)
        freki_blocks    = gather_check_instances(freki_data)

        match_dict = {'matches':0, 'compares':0}
        renum_checks(check_instances, freki_blocks, match_dict=match_dict, match_f=f)

