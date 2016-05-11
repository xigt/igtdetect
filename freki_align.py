#!/usr/bin/env python3.4
import copy
import os, re, sys
from argparse import ArgumentParser
from collections import defaultdict, Iterator, OrderedDict
import importlib

# -------------------------------------------
# Import the config file
# -------------------------------------------
from functools import partial

import pickle

import itertools

import time
from multiprocessing.pool import Pool

from multiprocessing import Lock

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
import logging
NORM_LEVEL = 1000
logging.addLevelName(NORM_LEVEL, 'norm')
LOG = logging.getLogger()
# -------------------------------------------

class FrekiBlock(object):
    """
    File to hold the "blocks"
    """
    def __init__(self, id=None, lines=None, bbox=None, page=None, doc_id=None):
        self.lines = lines if lines else []
        self.id = id
        self.bbox = bbox
        self.page = page
        self.doc_id = doc_id

    def __getitem__(self, item):
        return self.lines[item]

    def append(self, item):
        self.lines.append(item)

    def label_line(self, lineno, label):
        for i, line in enumerate(self.lines):
            if line.lineno == lineno:
                self.lines[i] = TextLine(line, lineno=line.lineno, fonts=line.fonts, label=label)
                break

    def _max_preamble_width(self):
        return max([len(self._line_preamble(l)) for l in self.lines])

    def _max_tag_width(self):
        try:
            return max([len(l.label) for l in self.lines if l.label is not None])
        except:
            return 1

    def _line_preamble(self, line):
        preamble = 'line={} '.format(line.lineno)
        mtw = self._max_tag_width()
        tag = 'O'
        if line.label:
            tag = line.label
        preamble += ' tag={{:{}}}'.format(mtw).format(tag)
        preamble += ' fonts={}'.format(line.fonts)
        return preamble

    def __repr__(self):
        first_line = self.lines[0].lineno if self.lines else 0
        last_line  = self.lines[-1].lineno if self.lines else 0

        ret_str = 'doc_id={} page={} block_id={} bbox={} {} {}\n'.format(self.doc_id, self.page, self.id, self.bbox, first_line, last_line)
        pre_width = self._max_preamble_width()
        ret_format = '{{:{}}}:{{}}\n'.format(pre_width)
        for line in self.lines:
            ret_str += ret_format.format(self._line_preamble(line), line)
        return ret_str


class FrekiFile(Iterator):
    def __init__(self, path):
        self.linedict = {}
        self.block_ids = {}
        self.block_dict = {}
        self.fh = open(path, 'r', encoding='utf-8')

        self.cur_block = None

        self._load()
        super().__init__()


    def _load(self):
        for line in self:
            pass
        self.fh.seek(0)

    def __len__(self):
        return len(list(self.linedict.keys()))

    def __next__(self):
        data = self.fh.__next__()
        while data:
            if data.startswith('doc_id'):
                blockdata = data.split()
                doc_id, page, block_id, bbox = [x.split('=')[1] for x in blockdata[:4]]
                self.cur_block = FrekiBlock(id=block_id, bbox=bbox, page=page, doc_id=doc_id)
                self.block_dict[block_id] = self.cur_block
            elif data.strip():
                line, fonts, text = re.search('^line=([0-9]+).*?fonts=(.*?):(.*$)', data).groups()
                self.linedict[int(line)] = TextLine(text, int(line))
                self.block_ids[int(line)] = self.cur_block.id
                tl = TextLine(text, int(line), fonts=fonts)
                self.cur_block.append(tl)
                return TextLine(text, int(line), fonts=fonts)

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

def lev_for_line(igt_index, lineno, odin_lines, freki_text, clean_text):
    lev = levenshtein(freki_text, clean_text)
    return (igt_index, lineno, odin_lines, lev)

# -------------------------------------------
# Now, iterate over the pairs.
# -------------------------------------------
def renum_checks(check_instances, ff : FrekiFile, filenum : int, match_dict = None, match_f=sys.stdout):

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
    p = Pool()
    l = Lock()
    picklepath = '{}.pickle'.format(filenum)
    if not os.path.exists(picklepath):

        LOG.log(NORM_LEVEL, "Building levenshtein edit dict for individual lines...")
        igtlinemap = defaultdict(partial(defaultdict, partial(defaultdict, partial(int, 'inf'))))


        for igt_index, igt in enumerate(xc):
            clean_tier = igt[1]

            for clean_item in clean_tier:
                odin_lines = tuple(int(l) for l in clean_item.attributes['line'].split())
                clean_text = clean_item.value()
                for freki_lineno, freki_line in sorted(ff.linedict.items(), key=lambda x: x[0]):

                    def callback(result):
                        l.acquire()
                        igt_index, lineno, odin_lines, lev = result
                        igtlinemap[igt_index][lineno][odin_lines] = lev
                        l.release()




                    # Remove the whitespace and compare the lines
                    freki_text  = re.sub('\s','', freki_line)
                    clean_text = re.sub('\s', '', clean_text)
                    p.apply_async(lev_for_line, args=[igt_index, freki_line.lineno, odin_lines, freki_text, clean_text], callback=callback)
                    # p.apply(lev_for_line, args=[igt_index, freki_line.lineno, odin_lines, freki_text, clean_text])
                    # igtlinemap[igt_index][freki_line.lineno][odin_lines] = levenshtein(freki_text, clean_text)

        p.close()
        p.join()
        LOG.log(NORM_LEVEL, "Dictionary building completed...")

        # Once we've computed the lev. distance for each pair of lines, now let's
        # do a search for the best fit, with the constraint that the blocks must
        # occur in the same order as the odin lines do.

        # First, let's calculate [odin_span][freki_spans]
        spanmaps = defaultdict(dict)

        for igt_index in igtlinemap.keys():

            for freki_lineno in sorted(igtlinemap[igt_index].keys()):
                chunk_total = 0

                odin_line_tups = tuple(sorted(igtlinemap[igt_index][freki_lineno]))
                freki_tup = tuple(freki_lineno+i for i in range(len(odin_line_tups)))

                if [num for num in freki_tup if num > max(igtlinemap[igt_index].keys())]:
                    continue

                for i, odin_line_tup in enumerate(odin_line_tups):
                    chunk_total += igtlinemap[igt_index].get(freki_lineno+i, defaultdict(lambda x: float('inf'))).get(odin_line_tup, float('inf'))


                spanmaps[odin_line_tups][freki_tup] = chunk_total

        # with open(picklepath, 'wb') as f:
        #     pickle.dump(spanmaps, f)

    else:
        with open(picklepath, 'rb') as f:
            spanmaps = pickle.load(f)

    # Now, let's search for the optimal non-overlapping configuration.
    odin_spans = sorted(spanmaps.keys())

    is_inconsistent = True

    best_spans = OrderedDict()


    noisy_spans = OrderedDict()

    cur_score = 0
    last_freki_stop = 0
    for odin_span in odin_spans:
        best_freki_span, score = sorted(spanmaps[odin_span].items(), key=lambda x: x[1]).pop(0)

        # If this span starts occurs before the last one, we
        # have an inconsistency.
        avg_dist = score / len(odin_span)
        # if best_freki_span[0] <= last_freki_stop:
        #     print('inconsistent! {} {} {}'.format(odin_span, best_freki_span, avg_dist))
        # else:
        #     print(avg_dist)



        best_spans[odin_span] = best_freki_span
        cur_score += score

        # If the score is too high, make a note of it
        if avg_dist > 20:
            noisy_spans[odin_span] = avg_dist

    blocks = OrderedDict()
    # Find the mapping between line numbers from the original
    # annotation to the new, freki numbers.
    for odin_span in odin_spans:

        freki_span = best_spans[odin_span]


        # match_f.write('igt_id={}\n'.format(igt.id))
        for i, odin_linenum in enumerate(odin_span):
            orig_indices = ' '.join([str(i) for i in odin_linenum])
            frek_index   = freki_span[i]
            orig_item = xigtpath.find(xc, '//tier[@state="cleaned"]/item[@line="{}"]'.format(orig_indices))
            label = orig_item.attributes['tag']
            # orig_item = xigtpath.find(xc, '//item')



            noise = noisy_spans.get(odin_span, 0.)
            if noise > 20:
                label = '*'+label
            elif noise > 50:
                label = '**'+label
            elif noise > 100:
                label = '***'+label

            freki_block = ff.block_dict[ff.block_ids[frek_index]]
            freki_block.label_line(frek_index, label)
            blocks[freki_block.id] = freki_block


            # match_f.write('{1}:{0}\n'.format(label, frek_index))
    for block in blocks.values():
        match_f.write(str(block)+'\n')
    # match_f.write('\n')


class TextLine(str):
    def __new__(cls, seq=None, lineno=None, fonts=None, label=None):
        if seq is None:
            seq = ''
        tl = str.__new__(cls, seq)
        tl.lineno = lineno
        tl.fonts = fonts
        tl.label = label
        return tl





if __name__ == '__main__':
    p = ArgumentParser()
    p.add_argument('checkfile')

    args = p.parse_args()

    filenum = int(os.path.splitext(os.path.basename(args.checkfile))[0])

    freki_path = os.path.join(FREKI_DIR, '{}.txt'.format(filenum))
    ff = FrekiFile(freki_path)

    check_data = open(args.checkfile, 'r', encoding='latin-1').readlines()

    with open(os.path.join(MATCH_DIR, str(filenum)+'.matches'), 'w', encoding='utf-8') as f:

        match_dict = {'matches':0, 'compares':0}
        renum_checks(gather_check_instances(check_data), ff, filenum, match_dict=match_dict, match_f=f)

