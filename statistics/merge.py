# -*- coding:utf-8 -*-
import pandas as pd
import sys
format_in = sys.argv[1]
format_out = sys.argv[2]
file_in = sys.argv[3]
file_out = sys.argv[4]
columns = sys.argv[5].split(',')
sample_info = pd.read_csv(sys.argv[6], sep='\t')
if format_in == 'excel':
    df = pd.read_excel(file_in)
else:
    df = pd.read_csv(file_in, sep='\t')
merge = pd.merge(df, sample_info, on=['sampleID'])
if format_out == 'excel':
    merge.to_excel(file_out, index=False, columns=columns)
else:
    merge.to_csv(file_out, index=False, sep='\t', columns=columns)