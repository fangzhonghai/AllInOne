# -*- coding:utf-8 -*-
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import AdaBoostClassifier
from collections import defaultdict
from argparse import ArgumentParser
from pysam import AlignmentFile
import pandas as pd
import numpy as np
import logging
import joblib


logger = logging.getLogger()
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler = logging.StreamHandler()
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)


BASE_INDEX = {'A': 0, 'C': 1, 'G': 2, 'T': 3}


class ACE:
    def __init__(self, genome, bam):
        self.ace_info = generate_ace_info(genome)
        self.bam = AlignmentFile(bam, 'rb')

    def snp_ratio(self):
        snp = ['rs4341', 'rs4342']
        ref_depth = 0
        pos_depth = 0
        for rs in snp:
            depth_array = self.bam.count_coverage(self.ace_info[rs]['chromosome'], self.ace_info[rs]['pos'] - 1, self.ace_info[rs]['pos'])
            ref_depth += depth_array[BASE_INDEX[self.ace_info[rs]['ref']]][0]
            pos_depth += np.sum(depth_array)
        ratio = ref_depth / pos_depth
        return 1 - ratio

    def depth_ratio(self):
        rs = 'rs1799752'
        forward_region_depth = np.sum(self.bam.count_coverage(self.ace_info[rs]['chromosome'], self.ace_info[rs]['pos'] - 15, self.ace_info[rs]['pos'] - 1))
        back_region_depth = np.sum(self.bam.count_coverage(self.ace_info[rs]['chromosome'], self.ace_info[rs]['pos'] - 1, self.ace_info[rs]['pos'] + 13))
        ratio = forward_region_depth / back_region_depth
        return 1 - ratio

    def split_reads_ratio(self):
        rs = 'rs1799752'
        reads_count = 0
        split_reads_count = 0
        for read in self.bam.fetch(self.ace_info[rs]['chromosome'], self.ace_info[rs]['pos'] - 1, self.ace_info[rs]['pos'] + 13):
            if read.is_duplicate or read.is_qcfail or read.is_unmapped:
                continue
            reads_count += 1
            if 'S' in read.cigarstring or 'H' in read.cigarstring:
                split_reads_count += 1
        ratio = split_reads_count / reads_count
        return ratio

    @classmethod
    def calculate_ratio(cls, parsed_args):
        info = pd.read_csv(parsed_args.info, sep='\t')
        info.index = info['sampleID'].tolist()
        info_dic = info.to_dict('index')
        ratio_dic = defaultdict(dict)
        for sample in info_dic:
            ace = cls(parsed_args.genome, info_dic[sample]['bam'])
            ratio_dic[sample] = {'snp_ratio': ace.snp_ratio(), 'depth_ratio': ace.depth_ratio(), 'split_reads_ratio': ace.split_reads_ratio()}
            ace.bam.close()
        ratio_df = pd.DataFrame.from_dict(ratio_dic, orient='index')
        return ratio_df

    @staticmethod
    def ace_classify(model_file, ratio_df):
        model = joblib.load(filename=model_file)
        ratio_df['genotype'] = ratio_df.apply(lambda x: model.predict(np.array([[x['snp_ratio'], x['depth_ratio'], x['split_reads_ratio']]]))[0], axis=1)
        return ratio_df

    @staticmethod
    def train_model(model_file, train_set_file):
        # 训练集文件列名snp_ratio,depth_ratio,split_reads_ratio,known;已知型别为I/I,I/D,D/D
        train_set = pd.read_csv(train_set_file, sep='\t')
        train_set = train_set.sample(frac=1)
        if train_set.shape[0] < 400:
            logger.info('Training set maybe too little...')
        # 默认的参数训练模型，一般不需改动
        abc = AdaBoostClassifier(n_estimators=50, learning_rate=1.0, algorithm='SAMME.R', random_state=100)
        x = train_set[['snp_ratio', 'depth_ratio', 'split_reads_ratio']]
        y = train_set['known']
        abc.fit(x, y)
        # 10fold交叉验证
        cv = 10
        cv_scores = cross_val_score(abc, x, y, cv=cv)
        if cv_scores.min() >= 0.98:
            logger.info('Trained model looks good! {} fold cross validate min accuracy is {}.'.format(cv, cv_scores.min()))
        else:
            logger.info('{} fold min accuracy is {}. Maybe need modify the AdaBoostClassifier parameter and retrain'.format(cv, cv_scores.min()))
        joblib.dump(filename=model_file, value=abc)


def generate_ace_info(genome) -> dict:
    if genome == 'hg19' or genome == 'GRCh37':
        ace_info = {
            'rs4341': {'chromosome': 'chr17', 'pos': 61565990, 'ref': 'G', 'alt': 'C'},
            'rs4342': {'chromosome': 'chr17', 'pos': 61565998, 'ref': 'A', 'alt': 'C'},
            'rs4343': {'chromosome': 'chr17', 'pos': 61566031, 'ref': 'G', 'alt': 'A'},
            'rs1799752': {'chromosome': 'chr17', 'pos': 61565891}
        }
        return ace_info
    elif genome == 'hg38' or genome == 'GRCh38':
        ace_info = {
            'rs4341': {'chromosome': 'chr17', 'pos': 63488629, 'ref': 'G', 'alt': 'C'},
            'rs4342': {'chromosome': 'chr17', 'pos': 63488637, 'ref': 'A', 'alt': 'C'},
            'rs4343': {'chromosome': 'chr17', 'pos': 63488670, 'ref': 'G', 'alt': 'A'},
            'rs1799752': {'chromosome': 'chr17', 'pos': 63488530}
        }
        return ace_info
    else:
        raise Exception('Genome must be hg19/hg38/GRCh37/GRCh38!')


def main():
    parser = ArgumentParser()
    parser.add_argument('-pipe', help='what to do? calculate_ratio,train_model,ace_classify,ace_classify_direct', required=True)
    parser.add_argument('-ratio', help='ratio file, must include columns: snp_ratio,depth_ratio,split_reads_ratio')
    parser.add_argument('-genome', help='Human reference genome, hg19/hg38/GRCh37/GRCh38', default='hg38')
    parser.add_argument('-model', help='model file to save or read')
    parser.add_argument('-train', help='training set file')
    parser.add_argument('-info', help='Sample information')
    parser.add_argument('-out', help='output tsv file')
    parsed_args = parser.parse_args()
    if parsed_args.pipe == 'calculate_ratio':
        ratio_df = ACE.calculate_ratio(parsed_args)
        ratio_df.to_csv(parsed_args.out, sep='\t')
    elif parsed_args.pipe == 'ace_classify':
        ratio_df = ACE.calculate_ratio(parsed_args)
        ratio_df = ACE.ace_classify(parsed_args.model, ratio_df)
        ratio_df.to_csv(parsed_args.out, sep='\t')
    elif parsed_args.pipe == 'ace_classify_direct':
        ratio_df = pd.read_csv(parsed_args.ratio, sep='\t')
        ratio_df = ACE.ace_classify(parsed_args.model, ratio_df)
        ratio_df.to_csv(parsed_args.out, sep='\t', index=False)
    elif parsed_args.pipe == 'train_model':
        ACE.train_model(parsed_args.model, parsed_args.train)
    else:
        raise Exception('-pipe must be calculate_ratio/train_model/ace_classify/ace_classify_direct!')


if __name__ == '__main__':
    main()
