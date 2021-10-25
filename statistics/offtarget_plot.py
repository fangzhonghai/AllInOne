# -*- coding:utf-8 -*-
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import sys
info = pd.read_csv(sys.argv[1], sep='\t', header=None)
writer = pd.ExcelWriter(sys.argv[2])
stat = pd.DataFrame()
stat_plot = pd.DataFrame(columns=['chromosome', 'rate', 'target'])
for i in range(info.shape[0]):
    sample = info.loc[i, 0]
    bedgraph = pd.read_csv(info.loc[i, 1], sep='\t', header=None)
    offtarget = pd.read_csv(info.loc[i, 2], sep='\t', header=None)
    # 所有比对结果统计
    bedgraph.loc[bedgraph[0].str.contains('_random$'), 0] = 'Unlocalized'
    bedgraph.loc[bedgraph[0].str.contains('_decoy$'), 0] = 'Decoy'
    bedgraph.loc[bedgraph[0].str.contains('^chrUn_') & (~bedgraph[0].str.contains('_decoy$')), 0] = 'Unplaced'
    bedgraph['base'] = (bedgraph[2] - bedgraph[1]) * bedgraph[3]
    bedgraph_stat = bedgraph.groupby([0]).agg({'base': np.sum}).reset_index()
    # offtarget结果统计
    offtarget.loc[offtarget[0].str.contains('_random$'), 0] = 'Unlocalized'
    offtarget.loc[offtarget[0].str.contains('_decoy$'), 0] = 'Decoy'
    offtarget.loc[offtarget[0].str.contains('^chrUn_') & (~offtarget[0].str.contains('_decoy$')), 0] = 'Unplaced'
    offtarget['base'] = (offtarget[2] - offtarget[1]) * offtarget[3]
    offtarget_stat = offtarget.groupby([0]).agg({'base': np.sum}).reset_index()
    # 结果统计汇总
    bedgraph_stat['offtarget'] = offtarget_stat['base']
    bedgraph_stat['ontarget'] = bedgraph_stat['base'] - bedgraph_stat['offtarget']
    bedgraph_stat['sample'] = sample
    bedgraph_stat['all_base'] = np.sum(bedgraph_stat['base'])
    bedgraph_stat['base_rate'] = bedgraph_stat['base'] / bedgraph_stat['all_base']
    bedgraph_stat['offtarget_rate'] = bedgraph_stat['offtarget'] / bedgraph_stat['all_base']
    bedgraph_stat['ontarget_rate'] = bedgraph_stat['ontarget'] / bedgraph_stat['all_base']
    bedgraph_stat.rename(columns={0: 'chromosome'}, inplace=True)
    # seaborn 画图数据
    stat = stat.append(bedgraph_stat)
    off = bedgraph_stat[['chromosome', 'offtarget_rate']].copy()
    off['target'] = 'off'
    off.rename(columns={'offtarget_rate': 'rate'}, inplace=True)
    stat_plot = stat_plot.append(off)
    on = bedgraph_stat[['chromosome', 'ontarget_rate']].copy()
    on['target'] = 'on'
    on.rename(columns={'ontarget_rate': 'rate'}, inplace=True)
    stat_plot = stat_plot.append(on)
stat.to_excel(excel_writer=writer, sheet_name='statistics', index=False)
stat_plot.to_excel(excel_writer=writer, sheet_name='seaborn_plot', index=False)
writer.save()
writer.close()
# 画图
plt.figure(figsize=(20, 15))
chromosome = ['chr' + str(c) for c in range(1, 23)] + ['chrX', 'chrY', 'chrM', 'chrEBV', 'Decoy', 'Unlocalized', 'Unplaced']
ax = sns.boxplot(x="chromosome", y="rate", hue="target", data=stat_plot, fliersize=3, order=chromosome)
ax.legend(title='target', fontsize=20, title_fontsize=20)
plt.setp(ax.get_xticklabels(), fontsize=20)
plt.setp(ax.get_yticklabels(), fontsize=20)
plt.xticks(rotation=60)
plt.ylabel('rate', fontsize=30)
plt.savefig(sys.argv[3])