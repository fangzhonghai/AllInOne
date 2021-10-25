# -*- coding:utf-8 -*-
from io import StringIO
import pandas as pd
import subprocess
import sys


def hlahd(sampleid, sample, result):
    shell = 'grep -v Couldn ' + result + ' | cut -f 1-3'
    genotype = subprocess.getoutput(shell)
    df = pd.read_csv(StringIO(genotype), sep='\t', header=None)
    df.columns = ['Gene', 'Allele1', 'Allele2']
    df.drop_duplicates(inplace=True)
    df['sampleID'] = sampleid
    df['sample'] = sample
    return df


info = pd.read_csv(sys.argv[1], sep='\t')
stat = pd.DataFrame()
for i in range(info.shape[0]):
    tmp = hlahd(info.loc[i, 'sampleID'], info.loc[i, 'sample'], info.loc[i, 'result'])
    stat = stat.append(tmp)
stat.to_excel(sys.argv[2], index=False)
