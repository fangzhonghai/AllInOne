# -*- coding:utf-8 -*-
from argparse import ArgumentParser
import pandas as pd


class AceConsistency:
    def __init__(self, info, known):
        self.info = pd.read_csv(info, sep='\t')
        self.known = pd.read_excel(known)
        self.known.rename(columns={'样本编号': 'sample', 'rs1799752(ACE)': 'known'}, inplace=True)

    @staticmethod
    def consistency(ref, alt, genotype):
        genotype = genotype.replace('/', '')
        if genotype == ref + ref:
            ace = 'D/D'
        elif genotype == ref + alt:
            ace = 'I/D'
        else:
            ace = 'I/I'
        return ace

    @classmethod
    def compare(cls, parsed_args):
        ac = cls(parsed_args.info, parsed_args.known)
        stat = pd.DataFrame()
        for i in range(ac.info.shape[0]):
            result = pd.read_csv(ac.info.loc[i, 'result'], sep='\t')
            result['sampleID'] = ac.info.loc[i, 'sampleID']
            result['sample'] = ac.info.loc[i, 'sample']
            if parsed_args.corrected:
                result['ace'] = result.apply(lambda x: ac.consistency(x['REF'], x['ALT'], x['genotype_corrected']), axis=1)
            else:
                result['ace'] = result.apply(lambda x: ac.consistency(x['REF'], x['ALT'], x['genotype']), axis=1)
            stat = stat.append(result, sort=False)
        merge = pd.merge(stat, ac.known[['sample', 'known']], on=['sample'], how='left')
        merge.fillna('-', inplace=True)
        merge.to_excel(parsed_args.out, index=False)


def main():
    parser = ArgumentParser()
    parser.add_argument('-corrected', help='use corrected genotype', action='store_true')
    parser.add_argument('-info', help='sample info')
    parser.add_argument('-known', help='known genotype')
    parser.add_argument('-out', help='output excel file')
    parsed_args = parser.parse_args()
    AceConsistency.compare(parsed_args)


if __name__ == '__main__':
    main()
