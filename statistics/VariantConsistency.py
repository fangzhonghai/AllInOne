# -*- coding:utf-8 -*-
from argparse import ArgumentParser
import pandas as pd


class VariantConsistency:
    def __init__(self, info, known):
        self.info = pd.read_csv(info, sep='\t')
        self.known = pd.read_excel(known)
        self.known.rename(columns={'样本编号': 'sample'}, inplace=True)
        self.known.index = self.known['sample'].tolist()
        self.known_dic = self.known.to_dict(orient='index')

    def consistency(self, result, corrected) -> dict:
        for rs in result:
            if result[rs]['sample'] in self.known_dic:
                if corrected:
                    genotype = [result[rs]['genotype_corrected'].replace('/', ''), ''.join(reversed(result[rs]['genotype_corrected'].split('/')))]
                else:
                    genotype = [result[rs]['genotype'].replace('/', ''), ''.join(reversed(result[rs]['genotype'].split('/')))]
                result[rs]['known'] = self.known_dic[result[rs]['sample']][rs]
                if result[rs]['known'] in genotype:
                    result[rs]['consistence'] = 'yes'
                elif result[rs]['known'] == '-':
                    result[rs]['consistence'] = '-'
                else:
                    result[rs]['consistence'] = 'no'
                ref, hom = result[rs]['REF'] + result[rs]['REF'], [base + base for base in result[rs]['ALT'].split(',')]
                if result[rs]['known'] == ref:
                    result[rs]['known GT'] = 'ref'
                elif result[rs]['known'] in hom:
                    result[rs]['known GT'] = 'hom'
                elif result[rs]['known'] == '-':
                    result[rs]['known GT'] = '-'
                else:
                    result[rs]['known GT'] = 'het'
            else:
                result[rs]['known'], result[rs]['known GT'], result[rs]['consistence'] = '-', '-', '-'
        return result

    @classmethod
    def compare(cls, parsed_args):
        vc = cls(parsed_args.info, parsed_args.known)
        stat = pd.DataFrame()
        for i in range(vc.info.shape[0]):
            result = pd.read_csv(vc.info.loc[i, 'result'], sep='\t')
            result['sampleID'] = vc.info.loc[i, 'sampleID']
            result['sample'] = vc.info.loc[i, 'sample']
            result.index = result['ID'].tolist()
            result = result.to_dict(orient='index')
            stat = stat.append(pd.DataFrame.from_dict(vc.consistency(result, parsed_args.corrected), orient='index'), sort=False)
        stat.to_excel(parsed_args.out, index=False)


def main():
    parser = ArgumentParser()
    parser.add_argument('-corrected', help='use corrected genotype', action='store_true')
    parser.add_argument('-info', help='sample info')
    parser.add_argument('-known', help='known genotype')
    parser.add_argument('-out', help='output excel file')
    parsed_args = parser.parse_args()
    VariantConsistency.compare(parsed_args)


if __name__ == '__main__':
    main()
