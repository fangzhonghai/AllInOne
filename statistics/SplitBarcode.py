# -*- coding:utf-8 -*-
from argparse import ArgumentParser
import pandas as pd
import numpy as np
import gzip


class SplitBarcode:
    def __init__(self, info, fq1, fq2):
        self.info = pd.read_csv(info, sep='\t', header=None, dtype={0: str})
        self.info.columns = ['barcode', 'seq']
        self.info['seq'] = self.info.apply(lambda x: dna_reverse_complement(x['seq']), axis=1)
        self.info.index = self.info['barcode'].tolist()
        self.info_dic = self.info.to_dict(orient='index')
        self.fq1 = gzip.open(fq1, 'r')
        self.fq2 = gzip.open(fq2, 'r')

    @staticmethod
    def match(reads_seq, barcode_seq, max_mismatch):
        match_num = np.sum(np.array(list(reads_seq.strip()[100:])) == np.array(list(barcode_seq.encode())))
        if match_num >= 10 - max_mismatch:
            return True
        else:
            return False

    @classmethod
    def write(cls, parsed_args):
        sb = cls(parsed_args.info, parsed_args.fq1, parsed_args.fq2)
        for barcode in sb.info_dic:
            sb.info_dic[barcode]['fq1'] = gzip.open(parsed_args.prefix + barcode + '_1.fq.gz', 'w')
            sb.info_dic[barcode]['fq2'] = gzip.open(parsed_args.prefix + barcode + '_2.fq.gz', 'w')
        while True:
            try:
                reads1_name, reads1_seq, reads1_sign, reads1_quality = (next(sb.fq1) for _ in range(4))
                reads2_name, reads2_seq, reads2_sign, reads2_quality = (next(sb.fq2) for _ in range(4))
            except:
                break
            for barcode in sb.info_dic:
                if sb.match(reads2_seq, sb.info_dic[barcode]['seq'], parsed_args.max_mismatch):
                    sb.info_dic[barcode]['fq1'].write(reads1_name)
                    sb.info_dic[barcode]['fq1'].write(reads1_seq)
                    sb.info_dic[barcode]['fq1'].write(reads1_sign)
                    sb.info_dic[barcode]['fq1'].write(reads1_quality)
                    sb.info_dic[barcode]['fq2'].write(reads2_name)
                    sb.info_dic[barcode]['fq2'].write(reads2_seq[0:100] + b'\n')
                    sb.info_dic[barcode]['fq2'].write(reads2_sign)
                    sb.info_dic[barcode]['fq2'].write(reads2_quality[0:100] + b'\n')
                    break
        for barcode in sb.info_dic:
            sb.info_dic[barcode]['fq1'].close()
            sb.info_dic[barcode]['fq2'].close()
        sb.fq1.close()
        sb.fq2.close()


def dna_reverse_complement(sequence):
    rule = {'A': 'T', 'G': 'C', 'C': 'G', 'T': 'A'}
    return ''.join([rule[base] for base in sequence])[::-1]


def main():
    parser = ArgumentParser()
    parser.add_argument('-info', help='barcode info', required=True)
    parser.add_argument('-prefix', help='output file prefix', required=True)
    parser.add_argument('-fq1', help='fq1 file', required=True)
    parser.add_argument('-fq2', help='fq2 file', required=True)
    parser.add_argument('-max_mismatch', help='max mismatch base number', default=1, type=int)
    parsed_args = parser.parse_args()
    SplitBarcode.write(parsed_args)


if __name__ == '__main__':
    main()
