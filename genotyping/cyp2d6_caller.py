# -*- coding:utf-8 -*-
import matplotlib
matplotlib.use('Agg')
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from collections import defaultdict
from argparse import ArgumentParser
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from pysam import AlignmentFile
from io import BytesIO
from PIL import Image
import pandas as pd
import numpy as np
import logging
import joblib
import sys
import os


logger = logging.getLogger()
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler = logging.StreamHandler()
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)


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
    def get_mean_depth(qc):
        # 获取样本平均深度
        # qc_df = pd.read_csv(qc, sep='\t', header=None, skiprows=1)
        # qc_df[0] = qc_df[0].str.replace(r'^\s+', '')
        # qc_df.index = qc_df[0].tolist()
        # return float(qc_df.loc['Target Mean Depth[RM DUP]:', 1])
        qc_df = pd.read_csv(qc, sep='\t', header=None, skiprows=3)
        qc_df[0] = qc_df[0].str.replace(r'^\s+', '')
        qc_df.index = qc_df[0].values
        return float(qc_df.loc['[Target] Average depth(rmdup)', 1])

    @staticmethod
    def fix_depth(depth_df):
        # 批次样本中值矫正
        median = depth_df.median(axis=1)
        median_fix = depth_df.div(median, axis='index')
        return median_fix.apply(np.round, args=(3,))

    @staticmethod
    def cal_baseline(depth_df):
        # 计算基线(median)
        baseline = depth_df.median(axis=1)
        return baseline

    @staticmethod
    def de_baseline(depth_df, baseline):
        # 去基线
        depth_df = depth_df.sub(baseline[0], axis='index')
        depth_df = depth_df.mask(depth_df < 0, 0)
        return depth_df

    @staticmethod
    def cal_region_center(region_dic):
        center_dic = defaultdict(dict)
        for r in region_dic:
            for sample in region_dic[r].columns:
                q25, q75 = np.quantile(region_dic[r][sample].values, [0.25, 0.75])
                km = KMeans(n_clusters=1, random_state=10).fit(region_dic[r][sample][(region_dic[r][sample] <= q75) & (region_dic[r][sample] >= q25)].values.reshape(-1, 1))
                km_center = np.round(km.cluster_centers_[0][0], 3)
                center_dic[sample][r] = km_center
        center_df = pd.DataFrame.from_dict(center_dic, orient='index')
        return center_df

    @staticmethod
    def train_model(model, train):
        # 训练集文件列名需包含exon1,intron2,exon3,exon5,intron5,exon6,intron6,known;已知型别为0N,1N,2N
        try:
            train_set = pd.read_csv(train, sep='\t')
        except:
            train_set = pd.read_excel(train)
        train_set = train_set.sample(frac=1)
        if train_set.shape[0] < 400:
            logger.info('Training set maybe too little...')
        # 默认的参数训练模型，一般不需改动
        rfc = RandomForestClassifier(n_estimators=100, random_state=100)
        x = train_set[['exon1', 'intron2', 'exon3', 'exon5', 'intron5', 'exon6', 'intron6']]
        y = train_set['known']
        rfc.fit(x, y)
        logger.info(f'Accuracy in train set is {rfc.score(x, y)}.')
        logger.info('Feature importances: exon1:{},intron2:{},exon3:{},exon5:{},intron5:{},exon6:{},intron6:{}'.format(*np.round(rfc.feature_importances_, 3)))
        # 10fold交叉验证
        cv = 10
        cv_scores = cross_val_score(rfc, x, y, cv=cv)
        if cv_scores.min() >= 0.99:
            logger.info('Trained model looks good! {} fold cross validate min accuracy is {}.'.format(cv, cv_scores.min()))
        else:
            logger.info('{} fold min accuracy is {}. Maybe need modify the RandomForestClassifier parameter and retrain'.format(cv, cv_scores.min()))
        joblib.dump(filename=model, value=rfc)

    @staticmethod
    def classify(model_file, center_df):
        model = joblib.load(filename=model_file)
        center_df['genotype'] = center_df.apply(lambda x: model.predict(np.array([[x['exon1'], x['intron2'], x['exon3'], x['exon5'], x['intron5'], x['exon6'], x['intron6']]]))[0], axis=1)
        return center_df

    def cal_region_depth(self, info):
        # 样本信息
        info = pd.read_csv(info, sep='\t')
        info.index = info['sampleID'].tolist()
        info_dic = info.to_dict(orient='index')
        # 计算CYP2D6基因每个位置标准化深度
        depth_dic = dict()
        for sample in info_dic:
            bam = AlignmentFile(info_dic[sample]['bam'], 'rb')
            depth_dic[sample] = np.sum(
                bam.count_coverage(contig=self.cyp2d6_dic['gene']['chromosome'], start=self.cyp2d6_dic['gene']['start']-1, stop=self.cyp2d6_dic['gene']['stop']), axis=0) / self.get_mean_depth(info_dic[sample]['qc'])
            depth_dic[sample] = np.round(depth_dic[sample], 3)
            bam.close()
        depth_df = pd.DataFrame(depth_dic)
        return depth_df

    def collect_region_depth(self, depth_df):
        # 过滤出批次样本特定区域的深度
        depth_df.index = depth_df.index + self.cyp2d6_dic['gene']['start']
        region = ['exon1', 'intron2', 'exon3', 'exon5', 'intron5', 'exon6', 'intron6']
        region_df = pd.DataFrame()
        region_dic = dict()
        for r in region:
            df = depth_df.loc[self.cyp2d6_dic[r]['start']:self.cyp2d6_dic[r]['stop']]
            region_dic[r] = df
            region_df = region_df.append(df)
        return region_df, region_dic

    def plot(self, depth_df, depth_df_fix, region_df, prefix):
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
        imgs[0].save(f'{prefix}.cyp2d6.pdf', "PDF", resolution=300.0, save_all=True, append_images=imgs[1:])
        # 统计结果
        out_df = region_df.median().to_frame(name='median').apply(np.round, args=(3,)).join(pd.DataFrame.from_dict(km_center_dic, orient='index'))
        out_df.rename(columns={0: 'kmeans'}, inplace=True)
        out_df.to_csv(f'{prefix}.center.tsv', sep='\t')

    @classmethod
    def calling(cls, parsed_args):
        cc = cls(parsed_args.cyp2d6)
        if parsed_args.pipe == 'train_model':
            logger.info('Start training model...')
            cc.train_model(parsed_args.model, parsed_args.train)
            logger.info(f'Save the model {parsed_args.model} and exit.')
            sys.exit(0)
        if parsed_args.pipe == 'classify_direct':
            try:
                center_df = pd.read_csv(parsed_args.center, sep='\t')
            except:
                center_df = pd.read_excel(parsed_args.center)
            center_df = cc.classify(parsed_args.model, center_df)
            center_df.to_csv(f'{parsed_args.prefix}.classify.tsv', sep='\t')
            sys.exit(0)
        depth_df = cc.cal_region_depth(parsed_args.info)
        if parsed_args.pipe == 'cal_baseline':
            baseline = cc.cal_baseline(depth_df)
            baseline.to_csv(f'{parsed_args.prefix}.baseline.tsv', sep='\t', header=False, index=False)
        elif parsed_args.pipe == 'cal_ratio' or parsed_args.pipe == 'classify':
            baseline = pd.read_csv(parsed_args.baseline, sep='\t', header=None)
            depth_df = cc.de_baseline(depth_df, baseline)
            depth_df_fix = cc.fix_depth(depth_df)
            region_df, region_dic = cc.collect_region_depth(depth_df)
            cc.plot(depth_df, depth_df_fix, region_df, f'{parsed_args.prefix}')
            center_df = cc.cal_region_center(region_dic)
            if parsed_args.pipe == 'classify':
                center_df = cc.classify(parsed_args.model, center_df)
                center_df.to_csv(f'{parsed_args.prefix}.classify.tsv', sep='\t')
            else:
                center_df.to_csv(f'{parsed_args.prefix}.center.region.tsv', sep='\t')
        else:
            raise Exception('-pipe must be cal_ratio/cal_baseline/classify/classify_direct/train_model!')


def main():
    parser = ArgumentParser()
    parser.add_argument('-cyp2d6', help='cyp2d6 info file', default=os.path.join(CONFIG, 'CYP2D6.info.txt'))
    parser.add_argument('-baseline', help='0N baseline', default=os.path.join(CONFIG, 'baseline.0N.txt'))
    parser.add_argument('-center', help='region center file, must include columns: exon1,intron2,exon3,exon5,intron5,exon6,intron6')
    parser.add_argument('-pipe', help='pipeline cal_ratio/cal_baseline/classify/classify_direct/train_model')
    parser.add_argument('-model', help='model file to save or read')
    parser.add_argument('-prefix', help='output file prefix')
    parser.add_argument('-train', help='training set file')
    parser.add_argument('-info', help='sample info file')
    parsed_args = parser.parse_args()
    CYP2D6Caller.calling(parsed_args)


if __name__ == '__main__':
    main()
