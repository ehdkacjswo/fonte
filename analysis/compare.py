import argparse, os, copy, itertools, pickle, sys
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from scipy.stats import wilcoxon, friedmanchisquare
from statistics import mean
import numpy as np
from tqdm import tqdm
import pandas as pd
import seaborn as sns
#from statsmodels.stats.multitest import multipletests

sys.path.append('/root/workspace/lib/')
from experiment_utils import *

from result_gen import get_metric_dict, org_fonte_metric

DIFF_DATA_DIR = '/root/workspace/data/Defects4J/diff'

def print_metric(metric_dict):
    print(f"MRR : {metric_dict['MRR']}, acc@1 : {metric_dict['acc@1']}, acc@2 : {metric_dict['acc@2']}, acc@3 : {metric_dict['acc@3']}, acc@5 : {metric_dict['acc@5']}, acc@10 : {metric_dict['acc@10']}, # Iters : {metric_dict['num_iters']}")

# fix : use_stopword and 
# Settings : ['score_mode', 'use_br', 'use_diff', 'stage2', 'use_stopword', 'adddel']
def get_best_set_bug2commit(use_br=True):

    # ['score_mode', 'use_br', 'use_diff', 'stage2', 'use_stopword', 'adddel']
    tot_metric_dict = get_metric_dict(bug2commit=True)
    best_list = [(tup, tot_metric_dict[tup]['MRR'], -tot_metric_dict[tup]['num_iters']) \
        for tup in tot_metric_dict.keys() if (tup[1] == str(use_br) and tup[3] == 'True' and tup[4] == 'True')]

    best_list.sort(key=lambda x : x[1:], reverse=True)

    pareto_list = [best_list[0]]
    for (tup, mrr, num_iters) in best_list[1:]:
        if pareto_list[-1][1] == mrr: # Same rank
            if pareto_list[-1][2] == num_iters: # Same metrics
                pareto_list.append((tup, mrr, num_iters))
            
        elif pareto_list[-1][2] < num_iters: # New setting has worse rank but better number of iterations
            pareto_list.append((tup, mrr, num_iters))

    print(['score_mode', 'use_br', 'use_diff', 'stage2', 'use_stopword', 'adddel'])

    for (tup, mrr, num_iters) in pareto_list:
        print(tup)
        print_metric(tot_metric_dict[tup])

# fix : use_stopword and 
def get_best_set(bug2commit=False, \
    fix={'HSFL':'False', 'use_br':'False', 'stage2':'True', 'use_stopword':'True'}, exclude=[]):

    # 대상 var이 한개인 경우는 작동하지 않는다
    # 현재 exclude 고려 X
    def setting_str_to_tup(setting):
        if setting[0] == '(' and setting[-1] == ')':
            setting = setting[1:-1]
        
        option_list = setting.split(',')

        if '(\'add\'' in option_list:
            ind = option_list.index('(\'add\'')
            option_list[ind] += f',{option_list[ind + 1]}'
            option_list = option_list[:ind + 1] + option_list[ind + 2:]

        setting_list = [None] * len(param_list)

        for ind, param in enumerate(param_list):
            if param in fix_list:
                setting_list[ind] = fix[param]
            elif param in remain_list:
                setting_list[ind] = option_list[remain_list.index(param)]

        return tuple(setting_list)

    if bug2commit:
        param_list = ['score_mode', 'use_br', 'use_diff', 'stage2', 'use_stopword', 'adddel']
    else:
        param_list = ['HSFL', 'score_mode', 'ensemble', 'use_br', 'use_diff', 'stage2', 'use_stopword', 'adddel']
    
    # 일단 given key는 전부 안에 있다고 봐야
    # exclude에서 fix, non-column 제외하는것도 일단 미루기
    param_ind_dict = {param : ind for ind, param in enumerate(param_list)}
    fix_list = [param for param in fix.keys() if param in param_list]
    exclude_list = [param for param in exclude if (param in param_list and param not in fix_list)]
    remain_list = [param for param in param_list if (param not in fix_list and param not in exclude_list)]

    fix_list.sort(key=lambda x: param_ind_dict[x])
    exclude_list.sort(key=lambda x: param_ind_dict[x])

    post_hoc_setting = ','.join([param + ':' + fix[param] for param in fix_list] + exclude_list)
    
    # ['score_mode', 'use_br', 'use_diff', 'stage2', 'use_stopword', 'adddel']
    tot_metric_dict = get_metric_dict(bug2commit)
    best_set_dict = dict()

    data = pd.read_csv(f"/root/workspace/analyze/data/{'bug2commit' if bug2commit else 'all'}/post_hoc/{post_hoc_setting}.csv")

    for metric in ['rank', 'num_iters']:
        best_set = set()

        # Load post-hoc results
        metric_data = data[data["metric"] == metric]

        # Extract all unique settings
        setting_set = set(metric_data["setting_1"]).union(metric_data["setting_2"])

        # Iterate through all settings
        for setting in setting_set:
            
            # Setting must not have large value for any other settings
            related_data = metric_data[
                ((metric_data["setting_1"] == setting) | (metric_data["setting_2"] == setting)) & \
                (metric_data["p.value"] < 0.05)
            ]

            related_set = set(related_data["setting_1"]).union(related_data["setting_2"])
            if len(related_set) == 0:
                #print(setting)
                continue

            setting_tup = setting_str_to_tup(setting)
            setting_metric = tot_metric_dict[setting_tup]['MRR' if metric == 'rank' else metric]

            # Check if the setting is better than 
            is_best = True
            for related_setting in related_set:
                related_tup = setting_str_to_tup(related_setting)
                related_metric = tot_metric_dict[related_tup]['MRR' if metric == 'rank' else metric]

                if (metric == 'rank' and related_metric > setting_metric) or (metric != 'rank' and related_metric < setting_metric):
                    is_best = False
                    #print(setting)
                    break
            
            if is_best:
                #print(setting_tup)
                best_set.add(setting_tup)

        best_set_dict[metric] = best_set

        if len(best_set) == 0:
            print(f'No setting with significantly better {metric}')

    # Every settings are statistically identical
    if len(best_set_dict['rank']) == 0 and len(best_set_dict['num_iters']) == 0:
        # Get all settings
        return
    
    # No setting is 
    elif len(best_set_dict['rank']) == 0:
        best_set = best_set_dict['num_iters']
    
    elif len(best_set_dict['num_iters']) == 0:
        best_set = best_set_dict['rank']
    
    else:
        best_set = best_set_dict['rank'] & best_set_dict['num_iters']

    best_list = [(tup, tot_metric_dict[tup]['MRR'], -tot_metric_dict[tup]['num_iters']) for tup in best_set]
    best_list.sort(key=lambda x : x[1:], reverse=True)

    pareto_list = [best_list[0]]
    for (tup, mrr, num_iters) in best_list[1:]:
        if pareto_list[-1][1] == mrr: # Same MRR
            if pareto_list[-1][2] == num_iters: # Same metrics (Elsewise worse metric)
                pareto_list.append((tup, mrr, num_iters))
            
        elif pareto_list[-1][2] < num_iters: # New setting has worse MRR but better number of iterations
            pareto_list.append((tup, mrr, num_iters))

    print(param_list)

    for (tup, mrr, num_iters) in pareto_list:
        print(tup)
        print_metric(tot_metric_dict[tup])

def compare_setting(setting1, setting2, bug2commit=True):

    def get_metric(fonte_df, iter_dict, setting, BIC):
        fonte_row = fonte_df.loc[setting]

        commit_df = fonte_row['commit'].dropna()
        score_df = fonte_row['vote'].dropna()
        rank_df = score_df.rank(method='max', ascending=False)

        # Index of the BIC
        BIC_ind = commit_df.loc[commit_df == BIC].index[0]

        return int(rank_df.loc[BIC_ind]), iter_dict[setting]

    if bug2commit:
        setting1 = ('None', setting1[0], 'extra', setting1[1], setting1[2], setting1[3], setting1[4], setting1[5])
        setting2 = ('None', setting2[0], 'extra', setting2[1], setting2[2], setting2[3], setting2[4], setting2[5])
    
    GT = load_BIC_GT("/root/workspace/data/Defects4J/BIC_dataset")
    metric_dict1 = [[], []]
    metric_dict2 = [[], []]

    for project in os.listdir(DIFF_DATA_DIR):
        [pid, vid] = project[:-1].split("-")
        BIC = GT.set_index(["pid", "vid"]).loc[(pid, vid), "commit"]
        project_dir = os.path.join(DIFF_DATA_DIR, project)

        metric_dict = dict()

        with open(os.path.join(project_dir, 'num_iters.pkl'), 'rb') as file:
            num_iter_dict = pickle.load(file)
    
        fonte_scores_df = pd.read_hdf(os.path.join(project_dir, 'fonte_scores.hdf'))
        
        rank1, iter1 = get_metric(fonte_scores_df, num_iter_dict, setting1, BIC)
        rank2, iter2 = get_metric(fonte_scores_df, num_iter_dict, setting2, BIC)

        metric_dict1[0] += [rank1]
        metric_dict1[1] += [iter1]
        metric_dict2[0] += [rank2]
        metric_dict2[1] += [iter2]
    
    print('Wilcoxon')
    print(wilcoxon(metric_dict1[0], metric_dict2[0], alternative='less'))
    print(wilcoxon(metric_dict1[1], metric_dict2[1], alternative='less'))

# ['HSFL', 'score_mode', 'ensemble', 'use_br', 'use_diff', 'stage2', 'use_stopword', 'adddel']
if __name__ == "__main__":
    # ['score_mode', 'use_br', 'use_diff', 'stage2', 'use_stopword', 'adddel']
    bug2commit_metric_dict = get_metric_dict(bug2commit=True)
    org_bug2commit_metric = bug2commit_metric_dict[('score', 'False', 'False', 'True', 'True', 'add')]

    print('Original Bug2Commit without bug report')
    print_metric(bug2commit_metric_dict[('score', 'False', 'False', 'True', 'True', 'add')])

    print('New best bug2commit without bug report')
    get_best_set_bug2commit(use_br=False)
    compare_setting(('score', 'False', 'False', 'True', 'True', 'all-sep'), ('score', 'False', 'False', 'True', 'True', 'add'), True)
    compare_setting(('score', 'False', 'False', 'True', 'True', 'add'), ('score', 'False', 'False', 'True', 'True', 'all-sep'), True)
    compare_setting(('score', 'False', 'True', 'True', 'True', 'del'), ('score', 'False', 'False', 'True', 'True', 'add'), True)
    compare_setting(('score', 'False', 'False', 'True', 'True', 'add'), ('score', 'False', 'True', 'True', 'True', 'del'), True)

    print('===============================================================================')

    print('Original Bug2Commit with bug report')
    print_metric(bug2commit_metric_dict[('score', 'True', 'False', 'True', 'True', 'add')])

    print('New best bug2commit with bug report')
    get_best_set_bug2commit(use_br=True)
    compare_setting(('both', 'True', 'True', 'True', 'True', 'all-sep'), ('score', 'True', 'False', 'True', 'True', 'add'), True)

    print('===============================================================================')

    fonte_metric_dict = get_metric_dict(bug2commit=False)

    print('Original Fonte without bug report')
    print_metric(org_fonte_metric())
    #print_metric(fonte_metric_dict[('False', 'rank', "('add', 1.0)", 'False', 'False', 'True')])
    print('New Fonte without bug report')
    #get_best_set(bug2commit=False, fix={'HSFL':'False', 'use_br':'False', 'stage2':'True', 'use_stopword':'True'}, exclude=[])
    get_best_set(bug2commit=False, fix={'HSFL':'False', 'use_br':'False', 'use_stopword':'True'}, exclude=[])
    compare_setting(('False', 'rank', "('add', 1.0)", 'False', 'True', 'True', 'True', 'all-sep'), ('False', 'None', '(\'add\', 0.0)', 'None', 'None', 'True', 'None', 'None'), False)
    compare_setting(('False', 'rank', "('add', 1.0)", 'False', 'True', 'True', 'True', 'del'), ('False', 'None', '(\'add\', 0.0)', 'None', 'None', 'True', 'None', 'None'), False)
    print('New Fonte with bug report')
    get_best_set(bug2commit=False, fix={'HSFL':'False', 'use_br':'True', 'stage2':'True', 'use_stopword':'True'}, exclude=[])
    compare_setting(('False', 'rank', "('add', 0.8)", 'True', 'True', 'True', 'True', 'add'), ('False', 'None', '(\'add\', 0.0)', 'None', 'None', 'True', 'None', 'None'), False)
    compare_setting(('False', 'rank', "('add', 0.7)", 'True', 'True', 'True', 'True', 'all-uni'), ('False', 'None', '(\'add\', 0.0)', 'None', 'None', 'True', 'None', 'None'), False)
    compare_setting(('False', 'rank', "('add', 0.9)", 'True', 'True', 'True', 'True', 'all-uni'), ('False', 'None', '(\'add\', 0.0)', 'None', 'None', 'True', 'None', 'None'), False)
    compare_setting(('False', 'rank', "('add', 0.8)", 'True', 'True', 'True', 'True', 'all-sep'), ('False', 'None', '(\'add\', 0.0)', 'None', 'None', 'True', 'None', 'None'), False)
    compare_setting(('False', 'rank', "('add', 1.0)", 'True', 'True', 'True', 'True', 'all-sep'), ('False', 'None', '(\'add\', 0.0)', 'None', 'None', 'True', 'None', 'None'), False)
    
    
    
    #best_metric = tot_metric_dict[('score', 'False', 'True', 'True', 'True', 'del')]
    #print(best_metric['MRR'], best_metric['acc@1'], best_metric['acc@2'], best_metric['acc@3'], best_metric['acc@5'], best_metric['acc@10'], best_metric['num_iters'])

    """get_best_set_bug2commit(use_br=True)
    get_best_set(bug2commit=False, fix={'HSFL':'False', 'use_br':'True', 'stage2':'True', 'use_stopword':'True'}, exclude=[])
    
    #print('Original Fonte')
    #print('MRR : 0.5277061540997692, acc@1 : 47, acc@2 : 66, acc@3 : 85, acc@5 : 98, acc@10 : 110, # Iters : 3.5076923076923032')
    
    #get_best_set(bug2commit=False, fix={'HSFL':'False', 'use_br':'True', 'stage2':'True', 'use_stopword':'True'}, exclude=[])
    #get_best_set_bug2commit(use_br=True)
    #compare_setting(setting1=('score', 'False', 'True', 'True', 'True', 'del'), setting2=('score', 'False', 'False', 'True', 'True', 'add'))"""