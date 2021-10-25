# -*- coding:utf-8 -*-
import os
import re
import sys
import time
import logging
import subprocess
import pandas as pd
from argparse import ArgumentParser


logger = logging.getLogger()
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler = logging.StreamHandler()
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

ROOT = os.path.abspath(os.path.dirname(__file__))
CONFIG = os.path.join(ROOT, 'src')
PIPELINE = os.path.join(ROOT, 'script')


class CreateJob:
    def __init__(self, flow, work_dir, info_df):
        self.flow = flow
        self.work_dir = work_dir
        self.info_df = info_df

    def make(self):
        self.flow.index = self.flow['Name'].tolist()
        job_conf_dic = self.flow.to_dict('index')
        self.info_df.index = self.info_df['sampleID'].tolist()
        info_df_dic = self.info_df.to_dict('index')
        for step in job_conf_dic.keys():
            pipe = os.path.join(PIPELINE, step + '.sh')
            if job_conf_dic[step]['Type'] == 'single':
                for sample in info_df_dic.keys():
                    script_dir = os.path.join(self.work_dir, sample, 'shell')
                    if not os.path.exists(script_dir):
                        os.makedirs(script_dir)
                    if step != 'Filter':
                        with open(os.path.join(script_dir, step + '.sh'), 'w') as f:
                            content = '''#!/bin/bash
sh {} {} {} {}
'''.format(pipe, self.work_dir, ROOT, sample)
                            f.write(content)
                    else:
                        fq1 = info_df_dic[sample]['fq1']
                        fq2 = info_df_dic[sample]['fq2']
                        with open(os.path.join(script_dir, step + '.sh'), 'w') as f:
                            content = '''#!/bin/bash
sh {} {} {} {} {} {}
'''.format(pipe, self.work_dir, ROOT, sample, fq1, fq2)
                            f.write(content)
            else:
                script_dir = os.path.join(self.work_dir, 'shell')
                if not os.path.exists(script_dir):
                    os.makedirs(script_dir)
                with open(os.path.join(script_dir, step + '.sh'), 'w') as f:
                    content = '''#!/bin/bash
sh {} {} {}
'''.format(pipe, self.work_dir, ROOT)
                    f.write(content)

    @classmethod
    def create_job(cls, parsed_args):
        flow = pd.read_csv(parsed_args.step, sep='\t')
        work_dir = parsed_args.work_dir
        samples_info_df = pd.read_csv(parsed_args.info, sep='\t')
        logger.info('Start Create Scripts!')
        cj = cls(flow, work_dir, samples_info_df)
        cj.make()
        logger.info('Create Scripts Finished!')


class SubJobFrame:
    def __init__(self, flow, work_dir, samples):
        self.flow = flow
        self.work_dir = work_dir
        self.samples = samples

    def generate_job_graph(self) -> dict:
        job_graph = dict()
        job_batch_dir = os.path.join(self.work_dir, 'shell')
        self.flow.index = self.flow['Name'].tolist()
        job_conf_dic = self.flow.to_dict('index')
        for step in job_conf_dic.keys():
            if job_conf_dic[step]['Type'] == 'single':
                for sample in self.samples:
                    job_single_dir = os.path.join(self.work_dir, sample, 'shell')
                    job = os.path.join(job_single_dir, step + '.sh')
                    job_graph[job] = {}
                    job_graph[job]['Name'] = step
                    job_graph[job]['Type'] = 'single'
                    job_graph[job]['Resources'] = job_conf_dic[step]['Resources']
                    job_graph[job]['JobID'] = ''
                    job_graph[job]['Children'] = []
                    if os.path.exists(job + '.complete'):
                        job_graph[job]['Status'] = 'complete'
                    else:
                        job_graph[job]['Status'] = 'incomplete'
                    parents = job_conf_dic[step]['Parents'].split(',')
                    job_graph[job]['Parents'] = []
                    for parent in parents:
                        if parent != 'None':
                            if job_conf_dic[parent]['Type'] == 'single':
                                job_graph[job]['Parents'].append(os.path.join(job_single_dir, parent + '.sh'))
                            else:
                                job_graph[job]['Parents'].append(os.path.join(job_batch_dir, parent + '.sh'))
            else:
                job = os.path.join(job_batch_dir, step + '.sh')
                job_graph[job] = {}
                job_graph[job]['Name'] = step
                job_graph[job]['Type'] = 'batch'
                job_graph[job]['Resources'] = job_conf_dic[step]['Resources']
                job_graph[job]['JobID'] = ''
                job_graph[job]['Children'] = []
                if os.path.exists(job + '.complete'):
                    job_graph[job]['Status'] = 'complete'
                else:
                    job_graph[job]['Status'] = 'incomplete'
                parents = job_conf_dic[step]['Parents'].split(',')
                job_graph[job]['Parents'] = []
                for parent in parents:
                    if parent != 'None':
                        if job_conf_dic[parent]['Type'] == 'single':
                            for sample in self.samples:
                                job_single_dir = os.path.join(self.work_dir, sample, 'shell')
                                job_graph[job]['Parents'].append(os.path.join(job_single_dir, parent + '.sh'))
                        else:
                            job_graph[job]['Parents'].append(os.path.join(job_batch_dir, parent + '.sh'))
        for job in job_graph.keys():
            for parent in job_graph[job]['Parents']:
                job_graph[parent]['Children'].append(job)
        return job_graph

    @staticmethod
    def job_num_in_sge():
        command = "qstat | grep `whoami` |wc -l"
        status, output = subprocess.getstatusoutput(command)
        return status, output

    @staticmethod
    def job_id_in_sge(command):
        status, output = subprocess.getstatusoutput(command)
        try:
            job_id = re.findall(r"Your job (\d+) ", output)[0]
        except IndexError:
            job_id = ''
        return status, job_id

    @staticmethod
    def job_status_in_sge(job_id):
        command = "qstat | grep " + "\"" + job_id + " " + "\""
        status, output = subprocess.getstatusoutput(command)
        return status, output

    @staticmethod
    def parents_status(job_graph, job):
        if len(job_graph[job]['Parents']) == 0:
            return 'complete'
        status_list = [job_graph[parent]['Status'] for parent in job_graph[job]['Parents']]
        if 'incomplete' not in status_list:
            return 'complete'
        else:
            return 'incomplete'

    @staticmethod
    def kill_job(job_graph, jobs):
        for job in jobs:
            if os.path.exists(job + '.complete'):
                continue
            if job_graph[job]['JobID'] != '':
                _ = subprocess.getoutput('qdel ' + job_graph[job]['JobID'])
        logger.info('Running and pending jobs were killed!')

    @classmethod
    def submit(cls, job_graph, job):
        status, job_num = cls.job_num_in_sge()
        if status != 0:
            logger.info('qstat error!')
            sys.exit(1)
        while int(job_num) >= 4000:
            time.sleep(600)
            status, job_num = cls.job_num_in_sge()
            if status != 0:
                logger.info('qstat error!')
                sys.exit(1)
        job_path = os.path.dirname(job)
        command = "qsub -wd " + job_path + " " + job_graph[job]['Resources'] + " " + job
        status, job_id = cls.job_id_in_sge(command)
        return status, job_id

    @classmethod
    def work_flow(cls, parsed_args):
        # 读取流程、工作路径、样本编号
        flow = pd.read_csv(parsed_args.step, sep='\t')
        work_dir = parsed_args.work_dir
        samples_info_df = pd.read_csv(parsed_args.info, sep='\t')
        samples = samples_info_df['sampleID'].tolist()
        # 生成流程图
        sjf = cls(flow, work_dir, samples)
        job_graph = sjf.generate_job_graph()
        logger.info('All Jobs Graph Created!')
        # 按流程图依赖顺序投递并监控任务状态
        jobs = list(filter(lambda x: len(job_graph[x]['Parents']) == 0, job_graph.keys()))
        while len(jobs) > 0:
            jobs_add = []
            jobs_remove = []
            for job in jobs:
                if os.path.exists(job + '.complete'):
                    job_graph[job]['Status'] = 'complete'
                if job_graph[job]['Status'] == 'incomplete':
                    if job_graph[job]['JobID'] == '':
                        if sjf.parents_status(job_graph, job) == 'complete':
                            status, job_id = sjf.submit(job_graph, job)
                            if status != 0:
                                logger.error(job + ' Submit Failed!')
                                sys.exit(1)
                            job_graph[job]['JobID'] = job_id
                            logger.info(job + ' Submit Success! JobID is ' + job_id)
                    else:
                        status, output = sjf.job_status_in_sge(job_graph[job]['JobID'])
                        if status != 0 and output == '' and not os.path.exists(job + '.complete'):
                            time.sleep(60)
                            if not os.path.exists(job + '.complete'):
                                logger.error(job + ' Run Failed!')
                                sjf.kill_job(job_graph, jobs)
                                sys.exit(1)
                else:
                    jobs_remove.append(job)
                    jobs_add += job_graph[job]['Children']
                    logger.info(job + ' Finished!')
            if len(jobs_add) == 0:
                time.sleep(60)
            jobs = list(set(list(set(jobs) - set(jobs_remove)) + jobs_add))
        logger.info('All Jobs Finished!')


def main():
    parser = ArgumentParser()
    parser.add_argument('-step', help='pipeline all steps', default=os.path.join(CONFIG, 'allsteps.tsv'))
    parser.add_argument('-work_dir', help='work dir', required=True)
    parser.add_argument('-info', help='sample info', required=True)
    parser.add_argument('-create_only', help='only create scripts', action='store_true')
    parser.add_argument('-submit_only', help='only submit scripts', action='store_true')
    parsed_args = parser.parse_args()
    os.system('cp ' + parsed_args.info + ' ' + os.path.join(parsed_args.work_dir, 'input.list'))
    if not parsed_args.submit_only:
        CreateJob.create_job(parsed_args)
    if parsed_args.create_only:
        logger.info('Only Create Scripts and Exit!')
        sys.exit(0)
    SubJobFrame.work_flow(parsed_args)


if __name__ == '__main__':
    main()
