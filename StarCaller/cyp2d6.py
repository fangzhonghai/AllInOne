# -*- coding:utf-8 -*-
import matplotlib
matplotlib.use('Agg')
from argparse import ArgumentParser
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from pysam import AlignmentFile
from io import BytesIO
from PIL import Image
import pandas as pd
import numpy as np
import os


ROOT = os.path.abspath(os.path.dirname(__file__))
CONFIG = os.path.join(ROOT, 'etc')


class CYP2D6Caller:
    def __init__(self, cyp2d6):
        self.cyp2d6 = pd.read_csv(cyp2d6, sep='\t', header=None)
        self.cyp2d6.columns = ['chromosome', 'start', 'stop', 'feature']
        self.cyp2d6.index = self.cyp2d6['feature'].tolist()
        self.cyp2d6_dic = self.cyp2d6.to_dict(orient='index')
        for i in range(1, 9):
            self.cyp2d6_dic[f'intron{i}'] = {}
            self.cyp2d6_dic[f'intron{i}']['chromosome'] = 'chr22'
            self.cyp2d6_dic[f'intron{i}']['start'] = self.cyp2d6_dic[f'exon{i+1}']['stop'] + 1
            self.cyp2d6_dic[f'intron{i}']['stop'] = self.cyp2d6_dic[f'exon{i}']['start'] - 1
            self.cyp2d6_dic[f'intron{i}']['feature'] = f'intron{i}'

    @staticmethod
    def mean_depth(qc):
        # 获取样本平均深度
        qc_df = pd.read_csv(qc, sep='\t', header=None, skiprows=1)
        qc_df[0] = qc_df[0].str.replace(r'^\s+', '')
        qc_df.index = qc_df[0].tolist()
        return float(qc_df.loc['Target Mean Depth[RM DUP]:', 1])

    @staticmethod
    def depth_fix(depth_df):
        # 批次样本中值矫正
        median = depth_df.median(axis=1)
        median_fix = depth_df.div(median, axis='rows')
        return median_fix.apply(np.round, args=(3,))

    def region_depth(self, depth_df):
        depth_df.index = depth_df.index + self.cyp2d6_dic['gene']['start']
        # 过滤出批次样本特定区域的深度
        region = ['exon1', 'intron2', 'exon3', 'exon5', 'intron5', 'exon6', 'intron6']
        region_df = pd.DataFrame()
        for r in region:
            df = depth_df.loc[self.cyp2d6_dic[r]['start']:self.cyp2d6_dic[r]['stop']]
            region_df = region_df.append(df)
        return region_df

    def plot(self, depth_df, depth_df_fix, region_df, out):
        imgs = list()
        km_center_dic = dict()
        median = depth_df.median(axis=1).values
        y_locator = plt.MultipleLocator(0.5)
        median_dic = region_df.median().to_dict()
        for sample in depth_df.columns:
            fig, ax = plt.subplots(2, 1, figsize=(18, 8))
            ax[0].set_title(sample)
            ax[0].axhline(y=1.0, ls=':', c='green')
            ax[0].plot(range(depth_df.shape[0]), depth_df[sample].values, color='blue', label='normal')
            ax[0].plot(range(depth_df.shape[0]), median, color='pink', label='median')
            ax[0].plot(range(depth_df.shape[0]), depth_df_fix[sample].values, color='purple', label='fix')
            ax[0].legend()
            ax[0].set_ylabel('Depth')
            ax[0].yaxis.set_major_locator(y_locator)
            for i in range(1, 10):
                start = self.cyp2d6_dic[f'exon{i}']['start'] - self.cyp2d6_dic['gene']['start']
                stop = self.cyp2d6_dic[f'exon{i}']['stop'] - self.cyp2d6_dic['gene']['start']
                ax[0].axvline(x=start, ls=':', c='red')
                ax[0].axvline(x=stop, ls=':', c='red')
                ax[0].fill_between([start, stop], 0, 2, color='yellow', alpha=0.3)
                ax[0].text((stop-start)/5+start, 0.5, f'exon{i}', verticalalignment='center')
            # 取中间Q2-Q3之间的数值进行聚类
            q25, q75 = np.quantile(region_df[sample].values, [0.25, 0.75])
            km = KMeans(n_clusters=1, random_state=10).fit(region_df[sample][(region_df[sample] <= q75) & (region_df[sample] >= q25)].values.reshape(-1, 1))
            km_center = np.round(km.cluster_centers_[0][0], 3)
            km_center_dic[sample] = km_center
            # 频率分布直方图
            ax[1].hist(region_df[sample].values, bins=50)
            ax[1].set_ylabel('Density')
            ax[1].set_xlabel('normal depth')
            ax[1].axvline(x=median_dic[sample], ls=':', c='red', label='median')
            ax[1].axvline(x=km_center, ls=':', c='green', label='kmeans')
            ax[1].legend()
            # 多个图分页
            buf = BytesIO()
            plt.savefig(buf, format='jpg', dpi=300)
            buf.seek(0)
            img = Image.open(buf)
            imgs.append(img)
            plt.close()
        imgs[0].save(f'{out}.plot.pdf', "PDF", resolution=300.0, save_all=True, append_images=imgs[1:])
        # 统计结果
        out_df = region_df.median().to_frame(name='median').apply(np.round, args=(3,)).join(pd.DataFrame.from_dict(km_center_dic, orient='index'))
        out_df.rename(columns={0: 'kmeans'}, inplace=True)
        out_df.to_csv(f'{out}.ratio.tsv', sep='\t')

    @classmethod
    def calling(cls, parsed_args):
        # 样本信息
        info = pd.read_csv(parsed_args.info, sep='\t')
        info.index = info['sampleID'].tolist()
        info_dic = info.to_dict(orient='index')
        # 创建类并进行深度统计
        cc = cls(parsed_args.cyp2d6)
        depth_dic = dict()
        for sample in info_dic:
            bam = AlignmentFile(info_dic[sample]['bam'], 'rb')
            depth_dic[sample] = np.sum(bam.count_coverage(contig=cc.cyp2d6_dic['gene']['chromosome'], start=cc.cyp2d6_dic['gene']['start']-1
                                                          , stop=cc.cyp2d6_dic['gene']['stop']), axis=0) / cc.mean_depth(info_dic[sample]['qc'])
            depth_dic[sample] = np.round(depth_dic[sample], 3)
            bam.close()
        depth_df = pd.DataFrame(depth_dic)
        depth_df_fix = cc.depth_fix(depth_df)
        # region_df = cc.region_depth(depth_df_fix)
        region_df = cc.region_depth(depth_df)
        cc.plot(depth_df, depth_df_fix, region_df, f'{parsed_args.prefix}')


def main():
    parser = ArgumentParser()
    parser.add_argument('-info', help='sample info file', required=True)
    parser.add_argument('-cyp2d6', help='cyp2d6 info file', default=os.path.join(CONFIG, 'CYP2D6.info.txt'))
    parser.add_argument('-prefix', help='output file prefix', required=True)
    parsed_args = parser.parse_args()
    CYP2D6Caller.calling(parsed_args)


if __name__ == '__main__':
    main()
