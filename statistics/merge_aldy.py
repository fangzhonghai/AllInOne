# -*- coding:utf-8 -*-
import pandas as pd
import subprocess
import sys


def aldy(result):
    shell = 'grep ' + '"#Solution 1:" ' + result
    try:
        genotype = subprocess.getoutput(shell).split(': ')[1]
    except IndexError:
        genotype = '-'
    return genotype


info = pd.read_csv(sys.argv[1], sep='\t')
info['CYP2D6'] = info.apply(lambda x: aldy(x['result']), axis=1)
known = pd.read_excel(sys.argv[2])
known.rename(columns={'样本编号': 'sample', 'CYP2D6*5': 'known'}, inplace=True)
merge = pd.merge(info, known[['sample', 'known']], on=['sample'], how='left')
merge.fillna('-', inplace=True)
merge.drop(columns=['result'], inplace=True)
merge.to_excel(sys.argv[3], index=False)
