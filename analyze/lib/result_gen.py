import argparse, os, copy, itertools, pickle, sys, json
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from scipy.stats import wilcoxon, ttest_rel
from statistics import mean
from typing import Literal
import numpy
import pandas as pd
import csv

sys.path.append('/root/workspace/lib/')
from experiment_utils import *

RESULT_DATA_DIR = '/root/workspace/data/Defects4J/result'

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
    
    # Load manual data only
    all_GT = load_BIC_GT("/root/workspace/data/Defects4J/BIC_dataset")
    GT = all_GT[all_GT['provenance'].str.contains("Manual", na=False)]

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

                    res_dict[setting_key]['MRR'] += 1 / (rank * len(GT))
                    res_dict[setting_key]['acc@1'] += 1 if rank <= 1 else 0
                    res_dict[setting_key]['acc@2'] += 1 if rank <= 2 else 0
                    res_dict[setting_key]['acc@3'] += 1 if rank <= 3 else 0
                    res_dict[setting_key]['acc@5'] += 1 if rank <= 5 else 0
                    res_dict[setting_key]['acc@10'] += 1 if rank <= 10 else 0

            else: # Bug2Commit and Ensemble have extra settings
                for setting, vote_df in value.items():
                    rank = vote_df['all']['rank'].get(BIC) if method == 'bug2commit' else vote_df['rank'].get(BIC)
                    #vote = vote_df['vote'].get(BIC)
                    
                    setting_key = frozenset((dict(setting) | {'stage2' : stage2}).items())

                    #if vote == 0:
                    #    print('0 score BIC', pid, vid, setting_key)

                    if mode == 'project':
                        res_dict.setdefault(setting_key, dict())
                        res_dict[setting_key][f'{pid}-{vid}b'] = {'rank' : rank}

                    else:
                        if setting_key not in res_dict:
                            res_dict[setting_key] = {'MRR': 0, 'acc@1': 0, 'acc@2': 0, 'acc@3': 0, 'acc@5': 0, 'acc@10': 0}

                        res_dict[setting_key]['MRR'] += 1 / (rank * len(GT))
                        res_dict[setting_key]['acc@1'] += 1 if rank <= 1 else 0
                        res_dict[setting_key]['acc@2'] += 1 if rank <= 2 else 0
                        res_dict[setting_key]['acc@3'] += 1 if rank <= 3 else 0
                        res_dict[setting_key]['acc@5'] += 1 if rank <= 5 else 0
                        res_dict[setting_key]['acc@10'] += 1 if rank <= 10 else 0
        
        # Get iteration (Bug2Commit doesn't have iteration data for possible 0 score BIC)
        with open(os.path.join(proj_dir, 'iteration', f'{method}.pkl'), 'rb') as file:
            iter_dict = pickle.load(file)
    
        for stage2, value in iter_dict.items():
            if method == 'fonte': # Fonte doesn't have extra setting
                setting_key = frozenset({'stage2' : stage2}.items())

                if mode == 'project':
                    res_dict[setting_key][f'{pid}-{vid}b']['num_iter'] = value
                else:
                    res_dict[setting_key]['num_iter'] += value / len(GT)
            
            else: #Ensemble has extra settings
                for setting, num_iter in value.items():
                    setting_key = frozenset((dict(setting) | {'stage2' : stage2}).items())
                    res_dict.setdefault(setting_key, dict())

                    if mode == 'project':
                        res_dict[setting_key].setdefault(f'{pid}-{vid}b', dict())
                        res_dict[setting_key][f'{pid}-{vid}b']['num_iter'] = num_iter
                    else:
                        res_dict[setting_key]['num_iter'] = res_dict[setting_key].get('num_iter', 0) + num_iter / len(GT)
    
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
    metrics_to_csv('ensemble')