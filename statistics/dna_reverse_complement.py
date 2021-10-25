# -*- coding:utf-8 -*-
import pandas as pd
import sys


def dna_reverse_complement(sequence):
    rule = {'A': 'T', 'G': 'C', 'C': 'G', 'T': 'A'}
    return ''.join([rule[base] for base in sequence])[::-1]


info = pd.read_csv(sys.argv[1], sep='\t', header=None, dtype={0: str})
info.columns = ['barcode', 'seq']
info['seq'] = info.apply(lambda x: dna_reverse_complement(x['seq']), axis=1)
info.to_csv(sys.argv[2], sep='\t', header=None, index=False)
