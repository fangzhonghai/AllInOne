# -*- coding:utf-8 -*-
import matplotlib
matplotlib.use('Agg')
from argparse import ArgumentParser
import matplotlib.pyplot as plt
from functools import reduce
import pandas as pd
import numpy as np
import pysam
import os


ROOT = os.path.abspath(os.path.dirname(__file__))
CONFIG = os.path.join(ROOT, 'etc')


class CYP2D6Caller:
    def __init__(self, info, gene, ratio):
        self.info = pd.read_csv(info, sep='\t')
        self.info.index = self.info['sampleID'].tolist()
        self.info = self.info.to_dict(orient='index')
        self.gene = pd.read_csv(gene, sep='\t', header=None)
        self.ratio = pd.read_csv(ratio, sep='\t', header=None)

    @staticmethod
    def get_target_mean_depth(qc):
        qc_df = pd.read_csv(qc, sep='\t', header=None, skiprows=1)
        qc_df[0] = qc_df[0].str.replace(r'^\s+', '')
        qc_df.index = qc_df[0].tolist()
        return float(qc_df.loc['Target Mean Depth[RM DUP]:', 1])

    @staticmethod
    def cal_region_depth(bam, chromosome, start, stop):
        bam_read = pysam.AlignmentFile(bam, 'rb')
        reads_depth = bam_read.count_coverage(contig=chromosome, start=start-1, stop=stop)
        every_pos_depth = sum(np.array(reads_depth))
        bam_read.close()
        return every_pos_depth

    @staticmethod
    def depth_fix(depths_df):
        depths_df_median = depths_df.median(axis=1)
        depths_df_median_fix = depths_df.div(depths_df_median, axis='rows')
        depths_df_median_fix = depths_df_median_fix.apply(np.round, args=(3,))
        return depths_df_median_fix

    @staticmethod
    def depth_plot(gene, depths_df, prefix):
        plot_start = gene.loc[gene[3] == 'gene'].values[0][1]
        gene_exon = gene[gene[3].str.contains('exon')].copy()
        gene_exon.reset_index(drop=True, inplace=True)
        sample_num = depths_df.shape[1]
        plt.figure(figsize=(18, sample_num * 3))
        y_major_locator = plt.MultipleLocator(0.5)
        depths_median = depths_df.median(axis=1).tolist()
        for i, sample in enumerate(list(depths_df.columns)):
            plt.subplot(sample_num, 1, i + 1)
            plt.title(sample, fontweight='bold')
            plt.axhline(y=1.0, ls=':', c='green')
            plt.plot(range(1, depths_df.shape[0] + 1), depths_df[sample].tolist())
            plt.plot(range(1, depths_df.shape[0] + 1), depths_median, color='deeppink', label='median')
            plt.ylabel('Normalised Depth')
            ax = plt.gca()
            ax.yaxis.set_major_locator(y_major_locator)
            for j in range(gene_exon.shape[0]):
                start = gene_exon.loc[j, 1] - plot_start + 1
                stop = gene_exon.loc[j, 2] - plot_start + 1
                plt.axvline(x=start, ls=':', c='red')
                plt.axvline(x=stop, ls=':', c='red')
                plt.fill_between([start, stop], 0, 2, color='yellow', alpha=0.3)
                plt.text((stop - start) / 5 + start, 0.5, 'exon ' + str(j + 1), verticalalignment='center')
            plt.legend()
        plt.tight_layout()
        plt.subplots_adjust(left=0.1, right=0.9)
        plt.savefig(prefix)
        plt.close()

    @staticmethod
    def ratio_plot(gene, depths_df, prefix):
        plot_start = gene.loc[gene[3] == 'gene'].values[0][1]
        gene_exon = gene[gene[3].str.contains('exon')].copy()
        gene_exon.reset_index(drop=True, inplace=True)
        sample_num = depths_df.shape[1]
        plt.figure(figsize=(18, sample_num * 3))
        y_major_locator = plt.MultipleLocator(0.5)
        for i, sample in enumerate(list(depths_df.columns)):
            plt.subplot(sample_num, 1, i + 1)
            plt.title(sample, fontweight='bold')
            plt.axhline(y=1.0, ls=':', c='green')
            plt.plot(range(1, depths_df.shape[0] + 1), depths_df[sample].tolist())
            plt.ylabel('Ratio')
            ax = plt.gca()
            ax.yaxis.set_major_locator(y_major_locator)
            for j in range(gene_exon.shape[0]):
                start = gene_exon.loc[j, 1] - plot_start + 1
                stop = gene_exon.loc[j, 2] - plot_start + 1
                plt.axvline(x=start, ls=':', c='red')
                plt.axvline(x=stop, ls=':', c='red')
                plt.fill_between([start, stop], 0, 2, color='yellow', alpha=0.3)
                plt.text((stop - start) / 5 + start, 0.5, 'exon ' + str(j + 1), verticalalignment='center')
        plt.tight_layout()
        plt.subplots_adjust(left=0.1, right=0.9)
        plt.savefig(prefix)
        plt.close()

    @staticmethod
    def perfect_gene_intron(gene):
        chromosome, start, stop, feature = list(), list(), list(), list()
        for i in range(1, 9):
            chromosome.append('chr22')
            stop.append(gene.loc[gene[3] == 'exon' + str(i)].values[0][1] - 1)
            start.append(gene.loc[gene[3] == 'exon' + str(i + 1)].values[0][2] + 1)
            feature.append('intron' + str(i))
        df = pd.DataFrame({0: chromosome, 1: start, 2: stop, 3: feature}, columns=[0, 1, 2, 3])
        return gene.append(df)

    @staticmethod
    def copy_ratio(depths_df, gene, region_dic):
        regions_depths_df_list = list()
        plot_start = gene.loc[gene[3] == 'gene'].values[0][1]
        depths_df.index = depths_df.index + plot_start
        for i in region_dic:
            for j in region_dic[i]:
                region = i + j
                start = gene.loc[gene[3] == region].values[0][1]
                stop = gene.loc[gene[3] == region].values[0][2]
                df = depths_df.loc[start:stop]
                regions_depths_df_list.append(df)
        regions_depths_df = reduce(lambda x, y: x.append(y), regions_depths_df_list)
        col_name = '-'.join([k + '_'.join(region_dic[k]) for k in region_dic])
        regions_depths_median = regions_depths_df.median()
        ratio_df = regions_depths_median.to_frame(name=col_name + '.median')
        return ratio_df

    @staticmethod
    def copy_threshold(ratio_df, col, col_copy, ratio_info_df):
        for i in range(ratio_info_df.shape[0]):
            left = ratio_info_df.loc[i, 0]
            right = ratio_info_df.loc[i, 1]
            ratio_df.loc[(ratio_df[col] < right) & (ratio_df[col] >= left), col_copy] = i
        ratio_max = ratio_info_df[1].tolist()[-1]
        ratio_df.loc[ratio_df[col] >= ratio_max, col_copy] = 6
        # 根据注释流程输入要求将拷贝数大于2的改成2
        # ratio_df.loc[ratio_df[col_copy] >= 3, col_copy] = 2
        return ratio_df
    
    @staticmethod
    def copy_caller(ratio_df, ratio):
        cols = ratio_df.columns.values.tolist()
        for col in cols:
            ratio_df[col + '.copy'] = '.'
            ratio_df = CYP2D6Caller.copy_threshold(ratio_df, col, col + '.copy', ratio)
        return ratio_df

    @classmethod
    def cyp2d6_copy_number(cls, parsed_args):
        cc = CYP2D6Caller(parsed_args.info, parsed_args.gene, parsed_args.ratio)
        chromosome = cc.gene.loc[cc.gene[3] == 'gene'].values[0][0]
        start = cc.gene.loc[cc.gene[3] == 'gene'].values[0][1]
        stop = cc.gene.loc[cc.gene[3] == 'gene'].values[0][2]
        depths_normalise_dic = dict()
        for sample in cc.info:
            mean_depth = cc.get_target_mean_depth(cc.info[sample]['qc'])
            depths = cc.cal_region_depth(cc.info[sample]['bam'], chromosome, start, stop)
            depths_normalise = np.round(depths/mean_depth, 3)
            depths_normalise_dic[sample] = depths_normalise
        depths_normalise_df = pd.DataFrame(depths_normalise_dic)
        depths_normalise_df_fix = cc.depth_fix(depths_normalise_df)
        cc.depth_plot(cc.gene, depths_normalise_df, parsed_args.prefix + '.CYP2D6.normalise_depths.plot.pdf')
        cc.ratio_plot(cc.gene, depths_normalise_df_fix, parsed_args.prefix + '.CYP2D6.ratio.plot.pdf')
        ratio_df = cc.copy_ratio(depths_normalise_df_fix, cc.perfect_gene_intron(cc.gene), {'intron': ['2', '5', '6'], 
                                                                                                     'exon': ['1', '2', '3', '5', '6']})
        ratio_df = cc.copy_caller(ratio_df, cc.ratio)
        ratio_df['genotype'] = ratio_df['intron2_5_6-exon1_2_3_5_6.median.copy'].astype('str') + 'N'
        ratio_df['sampleID'] = ratio_df.index
        ratio_df.to_excel(parsed_args.prefix + '.CYP2D6Caller.xlsx', index=False)


def main():
    parser = ArgumentParser()
    parser.add_argument('-info', help='sample info', required=True)
    parser.add_argument('-gene', help='CYP2D6 gene info', default=os.path.join(CONFIG, 'CYP2D6.info.txt'))
    parser.add_argument('-ratio', help='CYP2D6 copy ratio', default=os.path.join(CONFIG, 'CYP2D6.ratio.txt'))
    parser.add_argument('-prefix', required=True, help='output file prefix')
    parsed_args = parser.parse_args()
    CYP2D6Caller.cyp2d6_copy_number(parsed_args)


if __name__ == '__main__':
    main()
