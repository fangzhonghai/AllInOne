# -*- coding:utf-8 -*-
from collections import defaultdict
import pandas as pd
import json
import sys
import os
q20_30 = defaultdict(dict)
fastp = pd.read_csv(sys.argv[1], sep='\t', header=None)
for f in fastp[0].tolist():
    sample = os.path.basename(f).replace('.fastp.json', '')
    with open(f, 'r') as ff:
        f_dict = json.load(ff)
        q20_30[sample]['Q20_before_filtering'] = f_dict['summary']['before_filtering']['q20_rate']
        q20_30[sample]['Q30_before_filtering'] = f_dict['summary']['before_filtering']['q30_rate']
        q20_30[sample]['Q20_after_filtering'] = f_dict['summary']['after_filtering']['q20_rate']
        q20_30[sample]['Q30_after_filtering'] = f_dict['summary']['after_filtering']['q30_rate']
q20_30_df = pd.DataFrame.from_dict(q20_30, orient='index')
q20_30_df['sampleID'] = q20_30_df.index.tolist()
q20_30_df.to_csv(sys.argv[2], sep='\t', index=False)
