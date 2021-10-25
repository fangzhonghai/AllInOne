# -*- coding:utf-8 -*-
import pandas as pd
import sys
import os


def get_qc(cov_report, sampleID):
    cov_report_df = pd.read_csv(cov_report, sep='\t', header=None, skiprows=range(1))
    cov_report_df[0] = cov_report_df[0].str.replace(r'^\s+', '')
    cov_report_df.index = cov_report_df[0].values
    cov_report_df_t = cov_report_df.T
    cov_report_df_t['sampleID'] = sampleID
    return cov_report_df_t.loc[1]


# 读取Q20,Q30
q20_q30 = pd.read_csv(sys.argv[1], sep='\t')
# 合并捕获区域QC
qc = pd.read_csv(sys.argv[2], sep='\t', header=None)
qc_list = list()
for q in qc[0].tolist():
    sample = os.path.basename(q).replace('.stat', '')
    df = get_qc(q, sample)
    qc_list.append(df)
qc_df = pd.concat(qc_list, axis=1)
# 合并
out = pd.merge(qc_df.T, q20_q30, on=['sampleID'])
out.to_excel(sys.argv[3], index=False)
