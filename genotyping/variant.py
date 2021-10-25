# -*- coding:utf-8 -*-
from collections import defaultdict, namedtuple
from argparse import ArgumentParser
from pysam import AlignmentFile
import pandas as pd
import numpy as np
import tabix
import re
import os


VariantGenotype = namedtuple('VariantGenotype', ['CHROM', 'POS', 'ID', 'REF', 'ALT', 'ref_calling', 'alt_calling', 'QD', 'GT', 'DP',
                                                 'Ratio', 'genotype', 'genotype_corrected'])
VariantRecord = namedtuple('VariantRecord', ['ref_calling', 'alt_calling', 'QD', 'GT', 'DP', 'Ratio'])
ROOT = os.path.abspath(os.path.dirname(__file__))
CONFIG = os.path.join(ROOT, 'etc')


class Variant:
    def __init__(self, bam, vcf):
        self.bam = AlignmentFile(bam, 'rb')
        self.vcf = tabix.open(vcf)
        self.GenotypeDict = defaultdict(dict)

    @staticmethod
    def position_depth(bam, chromosome, position):
        depth = sum(sum(np.array(bam.count_coverage(contig=chromosome, start=position - 1, stop=position))))
        return int(depth)

    def parsing(self, chromosome, position):
        try:
            vcf_record = next(self.vcf.query(chromosome, position - 1, position))
        except (StopIteration, tabix.TabixError):
            gt = '0/0'
            dp = Variant.position_depth(self.bam, chromosome, position)
            return VariantRecord('.', '.', '.', gt, dp, '.')
        ref, alt = vcf_record[3], vcf_record[4]
        try:
            qd = re.findall('QD=(.*?);', vcf_record[7])[0]
        except IndexError:
            qd = '.'
        info = vcf_record[9].split(':')
        gt = info[0].replace('|', '/')
        ad, dp = info[1], int(info[2])
        # ratio = int(ad.split(',')[-1]) / dp
        ratio = int(ad.split(',')[-1]) / np.sum([int(i) for i in ad.split(',')])
        return VariantRecord(ref, alt, qd, gt, dp, ratio)

    @staticmethod
    def genotyping(drug_record, parsed_record):
        genotype_corrected = 'Fail'
        if parsed_record.GT == '0/0':
            genotype = drug_record.REF + '/' + drug_record.REF
            if parsed_record.DP >= drug_record.depth_threshold:
                genotype_corrected = genotype
            return VariantGenotype(drug_record.CHROM, drug_record.POS, drug_record.ID, drug_record.REF, drug_record.ALT, drug_record.REF,
                                   drug_record.REF, parsed_record.QD, parsed_record.GT, parsed_record.DP, parsed_record.Ratio,
                                   genotype, genotype_corrected)
        elif parsed_record.GT in ['0/1', '1/0']:
            genotype = parsed_record.ref_calling + '/' + parsed_record.alt_calling
            if parsed_record.DP >= drug_record.depth_threshold:
                if parsed_record.Ratio > drug_record.het_hom_ratio_threshold:
                    genotype_corrected = parsed_record.alt_calling + '/' + parsed_record.alt_calling
                elif parsed_record.Ratio < drug_record.ref_het_ratio_threshold:
                    genotype_corrected = parsed_record.ref_calling + '/' + parsed_record.ref_calling
                else:
                    genotype_corrected = genotype
        elif parsed_record.GT == '1/1':
            genotype = parsed_record.alt_calling + '/' + parsed_record.alt_calling
            if parsed_record.DP >= drug_record.depth_threshold:
                if parsed_record.Ratio <= drug_record.het_hom_ratio_threshold:
                    genotype_corrected = parsed_record.ref_calling + '/' + parsed_record.alt_calling
                else:
                    genotype_corrected = genotype
        else:
            alt_1, alt_2 = (alt for alt in parsed_record.alt_calling.split(','))
            genotype = alt_1 + '/' + alt_2
            ratio_1, ratio_2 = 1 - parsed_record.Ratio, parsed_record.Ratio
            if parsed_record.DP >= drug_record.depth_threshold:
                if ratio_1 > drug_record.het_hom_ratio_threshold:
                    genotype_corrected = alt_1 + '/' + alt_1
                elif ratio_2 > drug_record.het_hom_ratio_threshold:
                    genotype_corrected = alt_2 + '/' + alt_2
                else:
                    genotype_corrected = genotype
        if drug_record.corrected == 'no' and genotype_corrected != 'Fail':
            genotype_corrected = genotype
        return VariantGenotype(drug_record.CHROM, drug_record.POS, drug_record.ID, drug_record.REF, drug_record.ALT,
                               parsed_record.ref_calling, parsed_record.alt_calling, parsed_record.QD, parsed_record.GT, parsed_record.DP,
                               parsed_record.Ratio, genotype, genotype_corrected)

    @staticmethod
    def genotyping_rs368234815(drug_record, parsed_record, parsed_record_adjacent):
        vg = Variant.genotyping(drug_record, parsed_record)
        vg_adj = Variant.genotyping(drug_record, parsed_record_adjacent)
        qd = vg.QD + ',' + vg_adj.QD
        gt = vg.GT + ',' + vg_adj.GT
        dp = str(vg.DP) + ',' + str(vg_adj.DP)
        ratio = str(vg.Ratio) + ',' + str(vg_adj.Ratio)
        genotype_corrected = 'Fail'
        if vg.GT == '0/0':
            if vg_adj.GT == '0/0':
                ref_calling = vg.ref_calling
                alt_calling = vg.alt_calling
                genotype = ref_calling + '/' + alt_calling
                if vg.genotype_corrected != 'Fail' and vg_adj.genotype_corrected != 'Fail':
                    genotype_corrected = vg.genotype_corrected
            else:
                ref = vg.ref_calling[0:2]
                alt = vg.alt_calling[0:2]
                ref_adj, alt_adj = (adj for adj in vg_adj.genotype.split('/'))
                ref_calling = ref + ref_adj
                alt_calling = alt + alt_adj
                genotype = ref_calling + '/' + alt_calling
                if vg.genotype_corrected != 'Fail' and vg_adj.genotype_corrected != 'Fail':
                    ref_adj, alt_adj = (adj for adj in vg_adj.genotype_corrected.split('/'))
                    genotype_corrected = ref + ref_adj + '/' + alt + alt_adj
        else:
            if vg_adj.GT == '0/0':
                ref_calling = vg.ref_calling + vg_adj.ref_calling[-1]
                alt_calling = vg.alt_calling + vg_adj.alt_calling[-1]
                genotype = ref_calling + '/' + alt_calling
                if vg.genotype_corrected != 'Fail' and vg_adj.genotype_corrected != 'Fail':
                    ref, alt = (adj for adj in vg.genotype_corrected.split('/'))
                    genotype_corrected = ref + vg_adj.ref_calling[-1] + '/' + alt + vg_adj.alt_calling[-1]
            else:
                ref, alt = (adj for adj in vg.genotype.split('/'))
                ref_adj, alt_adj = (adj for adj in vg_adj.genotype.split('/'))
                ref_calling = ref + ref_adj
                alt_calling = alt + alt_adj
                genotype = ref_calling + '/' + alt_calling
                if vg.genotype_corrected != 'Fail' and vg_adj.genotype_corrected != 'Fail':
                    ref, alt = (adj for adj in vg.genotype_corrected.split('/'))
                    ref_adj, alt_adj = (adj for adj in vg_adj.genotype_corrected.split('/'))
                    genotype_corrected = ref + ref_adj + '/' + alt + alt_adj
        return VariantGenotype(drug_record.CHROM, drug_record.POS, drug_record.ID, drug_record.REF, drug_record.ALT, ref_calling,
                               alt_calling, qd, gt, dp, ratio, genotype, genotype_corrected)

    @classmethod
    def calling(cls, parsed_args):
        variant = cls(parsed_args.bam, parsed_args.vcf)
        drug = pd.read_excel(os.path.join(CONFIG, parsed_args.drug + '.xlsx'))
        drug.rename(columns={'#CHROM': 'CHROM'}, inplace=True)
        for drug_record in drug.itertuples():
            if drug_record.ID == 'rs368234815':
                parsed_record = variant.parsing(drug_record.CHROM, drug_record.POS)
                parsed_record_adjacent = variant.parsing(drug_record.CHROM, drug_record.POS + 2)
                variant.GenotypeDict[drug_record.ID] = \
                    variant.genotyping_rs368234815(drug_record, parsed_record, parsed_record_adjacent)._asdict()
            else:
                parsed_record = variant.parsing(drug_record.CHROM, drug_record.POS)
                variant.GenotypeDict[drug_record.ID] = variant.genotyping(drug_record, parsed_record)._asdict()
        variant.bam.close()
        df = pd.DataFrame.from_dict(variant.GenotypeDict, orient='index')
        df.to_csv(parsed_args.out, sep='\t', index=False)


def main():
    parser = ArgumentParser()
    parser.add_argument('-bam', help='bam file', required=True)
    parser.add_argument('-vcf', help='vcf file', required=True)
    parser.add_argument('-out', help='output file', required=True)
    parser.add_argument('-drug', help='drug variant information, variant or ace', default='variant')
    parsed_args = parser.parse_args()
    Variant.calling(parsed_args)


if __name__ == '__main__':
    main()
