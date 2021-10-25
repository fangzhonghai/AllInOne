# -*- coding:utf-8 -*-
import pandas as pd
import sys
qc = pd.read_excel(sys.argv[1])
columns = sys.argv[2].split(',')
df = pd.read_excel(sys.argv[3])
merge = pd.merge(df, qc[columns], on=['sampleID'])
merge.to_excel(sys.argv[4], index=False)

