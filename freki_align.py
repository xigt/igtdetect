#!/usr/bin/env python3
import copy
import os, re, sys
from argparse import ArgumentParser
from collections import defaultdict, Iterator
import importlib

# -------------------------------------------
# Import the config file
# -------------------------------------------
from functools import partial

import pickle

import itertools

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


class FrekiFile(Iterator):
    def __init__(self, path):
        self.linedict = {}
        self.blocks = {}
        self.fh = open(path, 'r', encoding='utf-8')

        self._load()

    def _load(self):
        for line in self:
            pass
        self.fh.seek(0)

    def __len__(self):
        return len(list(self.linedict.keys()))

    def __next__(self):
        data = self.fh.__next__()
        block = None
        while data:
            if data.startswith('doc_id'):
                block = data.split()[1].split('=')[1]
            elif data.strip():
                line, text = re.search('^line=([0-9]+):(.*$)', data).groups()
                self.linedict[int(line)] = TextLine(text, int(line))
                self.blocks[int(line)] = block
                return TextLine(text, int(line))

            data = self.fh.__next__()

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

class Configurator(Iterator):
    def __init__(self, odin_spans, freki_len):

        self.maximums = []
        self.minimums = []

        # Iterate backwards through the spans...
        for span in sorted(odin_spans, reverse=True):
            if not self.maximums:
                next_max = freki_len - len(span)
                next_min = 0
            else:
                next_max = self.maximums[0] - len(span)
                next_min = self.minimums[-1] + len(span)

            self.maximums.insert(0, next_max)
            self.minimums.append(next_min)

        self.offsets = [0 for i in self.minimums]
        self.offsets[-1] -=1




    def __next__(self):

        # Try to increment each offset, carrying over
        # if it hits the max.
        i = len(self.offsets)-1

        while i >= 0:
            # print('OFFSETS: {}'.format(self.offsets))
            if self.offsets[i]+self.minimums[i] < self.maximums[i]:
                self.offsets[i] += 1
                break

            # If we've overflowed a column... we want to reset it and
            # the ones that follow it
            elif self.offsets[i]+self.minimums[i] >= self.maximums[i]:

                # Each time we overflow, we want the new
                # offsets to start a little higher to account
                # for the other spans that have moved a little
                respace_value = self.offsets[i-1]+1
                for j in range(i, len(self.offsets)):
                    self.offsets[j] = respace_value

                # Now, set the i to the next position to the left
                # and see if we need to repeat
                i -= 1
            else:
                break

        if i < 0:
            raise StopIteration
        else:
            return [i+j for i, j in zip(self.offsets, self.minimums)]






# -------------------------------------------
# Now, iterate over the pairs.
# -------------------------------------------
def renum_checks(check_instances, ff : FrekiFile, match_dict = None, match_f=sys.stdout):

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

    # We will keep a dictionary that indexes by [igt#][freki-line][odin-line(s)] = lev. dist
    # then, we will seek to find the optimal mapping.
    if False:
        igtlinemap = defaultdict(partial(defaultdict, partial(defaultdict, partial(int, 'inf'))))


        for igt_index, igt in enumerate(xc):
            clean_tier = igt[1]

            for clean_item in clean_tier:
                odin_lines = tuple(int(l) for l in clean_item.attributes['line'].split())
                clean_text = clean_item.value()
                for freki_lineno, freki_line in sorted(ff.linedict.items(), key=lambda x: x[0]):

                    # Remove the whitespace and compare the lines
                    freki_text  = re.sub('\s','', freki_line)
                    clean_text = re.sub('\s', '', clean_text)
                    igtlinemap[igt_index][freki_line.lineno][odin_lines] = levenshtein(freki_text, clean_text)

        with open('test.pickle', 'wb') as f:
            pickle.dump(igtlinemap, f)
        sys.exit()

    else:
        with open('test.pickle', 'rb') as f:
            igtlinemap = pickle.load(f)

    # Once we've computed the lev. distance for each pair of lines, now let's
    # do a search for the best fit, with the constraint that the blocks must
    # occur in the same order as the odin lines do.


    # First, let's calculate [odin_span][freki_spans]
    spanmaps = defaultdict(dict)

    for igt_index in igtlinemap.keys():

        for freki_lineno in sorted(igtlinemap[igt_index].keys()):
            chunk_total = 0

            odin_line_tups = tuple(sorted(igtlinemap[igt_index][freki_lineno]))
            freki_tup = (freki_lineno, freki_lineno + len(odin_line_tups))

            for i, odin_line_tup in enumerate(odin_line_tups):
                chunk_total += igtlinemap[igt_index].get(freki_lineno+i, defaultdict(lambda x: float('inf'))).get(odin_line_tup, float('inf'))


            spanmaps[odin_line_tups][freki_tup] = chunk_total


    # Now, let's search for the optimal non-overlapping configuration.

    # c = Configurator(sorted(spanmaps.keys()), len(ff))



    c = Configurator([(0,1,2),(5,6,7),(9,10,11),(12,13,14)], 16)
    for config in c:
        print(config)


    sys.exit()




        #
        # # Find the mapping between line numbers from the original
        # # annotation to the new, freki numbers.
        # orig_lines, freki_lines = find_most_similar(linemap)
        #
        #
        # # match_f.write('igt_id={}\n'.format(igt.id))
        # for i, orig_line in enumerate(orig_lines):
        #     orig_indices = ','.join(orig_line.split())
        #     frek_index   = freki_lines[i]
        #
        #     orig_item = xigtpath.find(clean_tier, '//item[@line="{}"]'.format(orig_line))
        #
        #     match_f.write('{1}:{0}\n'.format(orig_item.attributes['tag'], frek_index))
        # # match_f.write('\n')


class TextLine(str):
    def __new__(cls, seq=None, lineno=None):
        if seq is None:
            seq = ''
        tl = str.__new__(cls, seq)
        tl.lineno = lineno
        return tl





if __name__ == '__main__':
    p = ArgumentParser()
    p.add_argument('checkfile')

    args = p.parse_args()

    filenum = int(os.path.splitext(os.path.basename(args.checkfile))[0])

    freki_path = os.path.join(FREKI_DIR, '{}.txt'.format(filenum))
    ff = FrekiFile(freki_path)


    check_data = open(checkfiles[filenum], 'r', encoding='latin-1').readlines()

    with open(os.path.join(MATCH_DIR, str(filenum)+'.matches'), 'w', encoding='utf-8') as f:

        check_instances = gather_check_instances(check_data)

        match_dict = {'matches':0, 'compares':0}
        renum_checks(check_instances, ff, match_dict=match_dict, match_f=f)

