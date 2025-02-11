import argparse, os, copy, itertools, pickle, sys, json
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from scipy.stats import wilcoxon, ttest_rel
from statistics import mean
from typing import Literal
import numpy
import pandas as pd
import csv

sys.path.append('/root/workspace/diff_util/lib/')
#from encoder import savepath_postfix
from tqdm import tqdm

sys.path.append('/root/workspace/lib/')
from experiment_utils import *

DIFF_DATA_DIR = '/root/workspace/data/Defects4J/diff'
RESULT_DATA_DIR = '/root/workspace/data/Defects4J/result'

voting_functions = {
    # key: (alpha, tau)
    (1, 'max'): (lambda r: r.score/r.max_rank),
    (0, 'max'): (lambda r: 1/r.max_rank),
    (1, 'dense'): (lambda r: r.score/r.dense_rank),
    (0, 'dense'): (lambda r: 1/r.dense_rank),
}

# Get metric dictionary for bug2commit
# metric : mean rank, mean number of iterations
def org_fonte_metric():
    GT = load_BIC_GT("/root/workspace/data/Defects4J/BIC_dataset")

    fonte_setting = ('False', 'None', '(\'add\', 0.0)', 'None', 'None', 'True', 'None', 'None')
    fonte_metric_dict = {'MRR' : 0, 'num_iters' : 0, 'acc@1' : 0, 'acc@2' : 0, 'acc@3' : 0, 'acc@5' : 0, 'acc@10' : 0}

    # Iterate through projects
    for project in os.listdir(DIFF_DATA_DIR):
        [pid, vid] = project[:-1].split("-")
        BIC = GT.set_index(["pid", "vid"]).loc[(pid, vid), "commit"]
        project_dir = os.path.join(DIFF_DATA_DIR, project)

        with open(os.path.join(project_dir, 'num_iters.pkl'), 'rb') as file:
            num_iter_dict = pickle.load(file)
    
        # Settings : ['HSFL', 'score_mode', 'ensemble', 'use_br', 'use_diff', 'stage2', 'use_stopword', 'adddel']
        fonte_scores_df = pd.read_hdf(os.path.join(project_dir, 'fonte_scores.hdf'))

        row = fonte_scores_df.loc[fonte_setting]

        commit_df = row['commit'].dropna()
        score_df = row['vote'].dropna()
        rank_df = score_df.rank(method='max', ascending=False)

        # Index of the BIC
        BIC_ind = commit_df.loc[commit_df == BIC].index[0]
        BIC_rank = rank_df.loc[BIC_ind]

        n_list = [1, 2, 3, 5, 10]

        fonte_metric_dict['num_iters'] += num_iter_dict[fonte_setting] / 130
        fonte_metric_dict['MRR'] += 1 / (BIC_rank * 130)
        for n in n_list:
            if BIC_rank <= n:
                fonte_metric_dict[f'acc@{n}'] += 1
    
    return fonte_metric_dict

def handle_metric(metric_dict, rank):
    metric_dict['MRR'] 

# Get metric dictionary for bug2commit
# method : fonte, bug2commit, ensemble
# mode : all, project
# metric : mean rank, mean number of iterations
def get_metric_dict(method: Literal['fonte', 'bug2commit', 'ensemble'], mode: Literal['all', 'project']):
    savepath = f"/root/workspace/analyze/data/{method}/metric_{mode}.pkl"

    # If file already exists, read it
    if os.path.isfile(savepath):
        with open(savepath, 'rb') as file:
            return pickle.load(file)
    
    GT = load_BIC_GT("/root/workspace/data/Defects4J/BIC_dataset")
    res_dict = dict()

    # Iterate through projects
    for _, row in GT.iterrows():
        pid, vid, BIC = row.pid, row.vid, row.commit

        # Closure-131b has wrong BIC data
        if pid == 'Closure' and vid == '131':
            continue
        
        proj_dir = os.path.join(RESULT_DATA_DIR, f'{pid}-{vid}b')

        # Get rank
        with open(os.path.join(proj_dir, 'vote', f'{method}.pkl'), 'rb') as file:
            vote_dict = pickle.load(file)
        
        for stage2, value in vote_dict.items():
            if method == 'fonte': # Fonte doesn't have extra setting
                rank = value['rank'].get(BIC)
                setting_key = frozenset({'stage2' : stage2}.items())

                if mode == 'project':
                    if setting_key not in res_dict:
                        res_dict[setting_key] = dict()

                    res_dict[setting_key][f'{pid}-{vid}b'] = {'rank' : rank}

                else:
                    if setting_key not in res_dict:
                        res_dict[setting_key] = {'MRR': 0, 'acc@1': 0, 'acc@2': 0, 'acc@3': 0, 'acc@5': 0, 'acc@10': 0, 'num_iter': 0}

                    res_dict[setting_key]['MRR'] += 1 / (rank * 129)
                    res_dict[setting_key]['acc@1'] += 1 if rank <= 1 else 0
                    res_dict[setting_key]['acc@2'] += 1 if rank <= 2 else 0
                    res_dict[setting_key]['acc@3'] += 1 if rank <= 3 else 0
                    res_dict[setting_key]['acc@5'] += 1 if rank <= 5 else 0
                    res_dict[setting_key]['acc@10'] += 1 if rank <= 10 else 0

            else: # Bug2Commit and Ensemble have extra settings
                for setting, vote_df in value.items():
                    rank = vote_df['rank'].get(BIC)
                    setting_key = frozenset((dict(setting) | {'stage2' : stage2}).items())

                    if mode == 'project':
                        if setting_key not in res_dict:
                            res_dict[setting_key] = dict()

                        res_dict[setting_key][f'{pid}-{vid}b'] = {'rank' : rank}

                    else:
                        if setting_key not in res_dict:
                            res_dict[setting_key] = {'MRR': 0, 'acc@1': 0, 'acc@2': 0, 'acc@3': 0, 'acc@5': 0, 'acc@10': 0, 'num_iter': 0}

                        res_dict[setting_key]['MRR'] += 1 / (rank * 129)
                        res_dict[setting_key]['acc@1'] += 1 if rank <= 1 else 0
                        res_dict[setting_key]['acc@2'] += 1 if rank <= 2 else 0
                        res_dict[setting_key]['acc@3'] += 1 if rank <= 3 else 0
                        res_dict[setting_key]['acc@5'] += 1 if rank <= 5 else 0
                        res_dict[setting_key]['acc@10'] += 1 if rank <= 10 else 0
        
        # Get iteration (Bug2Commit doesn't have iteration data for possible 0 score BIC)
        if method != 'bug2commit':
            with open(os.path.join(proj_dir, 'iteration', f'{method}.pkl'), 'rb') as file:
                iter_dict = pickle.load(file)
        
            for stage2, value in iter_dict.items():
                if method == 'fonte': # Fonte doesn't have extra setting
                    setting_key = frozenset({'stage2' : stage2}.items())

                    if mode == 'project':
                        res_dict[setting_key][f'{pid}-{vid}b']['num_iter'] = value
                    else:
                        res_dict[setting_key]['num_iter'] += value
                
                else: #Ensemble has extra settings
                    for setting, num_iter in value.items():
                        setting_key = frozenset((dict(setting) | {'stage2' : stage2}).items())

                        if mode == 'project':
                            res_dict[setting_key][f'{pid}-{vid}b']['num_iter'] = num_iter
                        else:
                            res_dict[setting_key]['num_iter'] += num_iter
    
    # Save & return the dictionary
    #os.makedirs(savepath, exist_ok=True)
    with open(savepath, 'wb') as file:
        pickle.dump(res_dict, file)

    return res_dict

# Create csv file for ART ANOVA
def metrics_to_csv(method: Literal['fonte', 'bug2commit', 'ensemble']):
    savepath = f"/root/workspace/analyze/data/{method}/metrics.csv"

    """if os.path.isfile(savepath):
        print(f'{savepath} already exists!')
        return"""
    
    GT = load_BIC_GT("/root/workspace/data/Defects4J/BIC_dataset")
    metric_dict = get_metric_dict(method, mode='project')
    
    setting_key_list = list(dict(next(iter(metric_dict))).keys())
    field = ['project'] + setting_key_list + ['DependentName', 'DependentValue']

    with open(savepath, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(field)

        for setting, proj_dict in metric_dict.items():
            setting_dict = dict(setting)
            setting_row = [setting_dict[key] for key in setting_key_list]

            for project, sub_dict in proj_dict.items():
                writer.writerow([project] + setting_row + ['rank', sub_dict['rank']])
                
                if method != 'bug2commit': # Bug2Commit doesn't have iteration data
                    writer.writerow([project] + setting_row + ['num_iter', sub_dict['num_iter']])

if __name__ == "__main__":
    # Generate score data
    #print('Generating score data')
    """for folder in tqdm(os.listdir(DIFF_DATA_DIR)):
        # Get BIC data
        print(f'Fonte_score_eval : Working on {folder}')
        [pid, vid] = folder[:-1].split("-")
        results_dict = score_eval_all(pid, vid, args.tool, args.formula, args.lamb, voting_functions[(args.alpha, args.tau)])
        results_df = pd.concat(results_dict, \
            names=['HSFL', 'score_mode', 'ensemble', 'use_br', 'use_diff', 'stage2', 'use_stopword', 'adddel']).unstack()
        
        results_df.to_hdf(os.path.join(DIFF_DATA_DIR, f'{folder}/fonte_scores.hdf'), key='data', mode='w')
    
    # Generate iteration data
    num_iters_dict = dict()
    #print('Generating iteration data')
    for folder in tqdm(os.listdir(DIFF_DATA_DIR)):
        print(f'Weighted bisection : Working on {folder}')
        [pid, vid] = folder[:-1].split("-")
        result_dict = bisection_all(pid, vid)

        with open(os.path.join(DIFF_DATA_DIR, folder, 'num_iters.pkl'), 'wb') as file:
            pickle.dump(result_dict, file)"""
    
    # Generating csv file
    #metrics_to_csv(False)
    #metrics_to_csv(True)
    #print(get_metric_dict('ensemble', False))
    metrics_to_csv('ensemble')