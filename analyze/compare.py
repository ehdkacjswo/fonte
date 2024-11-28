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

DIFF_DATA_DIR = '/root/workspace/data/Defects4J/diff'

# Get metric dictionary for bug2commit
# metric : mean rank, mean number of iterations
def get_metric_dict(bug2commit=True):
    savepath = f"/root/workspace/analyze/data/{'bug2commit' if bug2commit else 'all'}/metric_dict.pkl"

    # If file already exists, read it
    if os.path.isfile(savepath):
        with open(savepath, 'rb') as file:
            return pickle.load(file)
    
    GT = load_BIC_GT("/root/workspace/data/Defects4J/BIC_dataset")
    tot_metric_dict = dict()

    # Iterate through projects
    for project in tqdm(os.listdir(DIFF_DATA_DIR)):
        [pid, vid] = project[:-1].split("-")
        BIC = GT.set_index(["pid", "vid"]).loc[(pid, vid), "commit"]
        project_dir = os.path.join(DIFF_DATA_DIR, project)

        with open(os.path.join(project_dir, 'num_iters.pkl'), 'rb') as file:
            num_iter_dict = pickle.load(file)
    
        # Settings : ['HSFL', 'score_mode', 'ensemble', 'use_br', 'use_diff', 'stage2', 'use_stopword', 'adddel']
        fonte_scores_df = pd.read_hdf(os.path.join(project_dir, 'fonte_scores.hdf'))

        # Iterate through extra scores of every settings
        for setting, row in fonte_scores_df.iterrows():
            if bug2commit: # Bug2Commit only case
                if setting[0] != 'None' or setting[2] != '(\'add\', 0.0)':
                    continue
                metric_dict_key = tuple(option for ind, option in enumerate(tuple(setting)) if ind not in [0, 2])
            
            else: # Bug2Commit with Fonte
                if setting[2] == '(\'add\', 1.0)' or setting[2] == '(\'add\', 0.0)':
                    continue
                metric_dict_key = tuple(setting)

            commit_df = row['commit'].dropna()
            score_df = row['vote'].dropna()
            rank_df = score_df.rank(method='max', ascending=False)

            # Index of the BIC
            BIC_ind = commit_df.loc[commit_df == BIC].index[0]
            BIC_rank = rank_df.loc[BIC_ind]

            setting_tup = tuple(setting)
            n_list = [1, 2, 3, 5, 10]

            if metric_dict_key not in tot_metric_dict:
                tot_metric_dict[metric_dict_key] = dict()
                tot_metric_dict[metric_dict_key]['rank'] = 0
                tot_metric_dict[metric_dict_key]['num_iters'] = 0
                tot_metric_dict[metric_dict_key]['MRR'] = 0

                for n in n_list:
                    tot_metric_dict[metric_dict_key][f'acc@{n}'] = 0

            tot_metric_dict[metric_dict_key]['rank'] += BIC_rank / 130
            tot_metric_dict[metric_dict_key]['num_iters'] += num_iter_dict[setting_tup][1] / 130

            tot_metric_dict[metric_dict_key]['MRR'] += 1 / (BIC_rank * 130)
            for n in n_list:
                if BIC_rank <= n:
                    tot_metric_dict[metric_dict_key][f'acc@{n}'] += 1
    
    with open(savepath, 'wb') as file:
        pickle.dump(tot_metric_dict, file)
    
    return tot_metric_dict

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
        print(tup, mrr, -num_iters)

# fix : use_stopword and 
def get_best_set(bug2commit=False, \
    fix={'HSFL':'False', 'use_br':'True', 'stage2':'True', 'use_stopword':'True'}, exclude=[]):

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
    
    best_set = best_set_dict['rank'] & best_set_dict['num_iters']
    best_list = [(tup, tot_metric_dict[tup]['MRR'], -tot_metric_dict[tup]['num_iters']) for tup in best_set]
    best_list.sort(key=lambda x : x[1:], reverse=True)

    pareto_list = [best_list[0]]
    for (tup, mrr, num_iters) in best_list[1:]:
        if pareto_list[-1][1] == mrr: # Same rank
            if pareto_list[-1][2] == num_iters: # Same metrics
                pareto_list.append((tup, mrr, num_iters))
            
        elif pareto_list[-1][2] < num_iters: # New setting has worse rank but better number of iterations
            pareto_list.append((tup, mrr, num_iters))

    print(param_list)

    for (tup, mrr, num_iters) in pareto_list:
        print(tup, mrr, -num_iters)
    
    """best_list = [(tup, tot_metric_dict[tup]['MRR'], tot_metric_dict[tup]['acc@1'], tot_metric_dict[tup]['acc@2'],\
        tot_metric_dict[tup]['acc@3'], tot_metric_dict[tup]['acc@5'], tot_metric_dict[tup]['acc@10'], \
        -tot_metric_dict[tup]['num_iters']) for tup in best_set]
    best_list.sort(key=lambda x : x[1:], reverse=True)
    #print(best_list)

    pareto_list = []
    for val1 in best_list:
        add = True
        
        for val2 in best_list:
            if val1[0] == val2[0]:
                continue
            
            # No : Worse than val2 on some points, same on others

            same = True
            better = False
            worse = False

            for i in range(1, 8):
                if val1[i] > val2[i]:
                    better = True
                    same = False
                elif val2[i] > val1[i]:
                    worse = True
                    same = False
            
            if worse and not better:
                add = False
                break
        
        if add:
            pareto_list.append(val1)

    print(param_list)

    for val in pareto_list:
        print(val)"""

if __name__ == "__main__":
    #compare_extra_score()
    #compare_num_iters()
    #check_param_iter({'score_mode': 'both', 'stage2': 'True'}, strict=False)

    """with open('/root/workspace/num_iters.pkl', 'rb') as file:
        num_iter_dict = pickle.load(file)
    
    num_iter_list = list(num_iter_dict.items())
    num_iter_list = [(a, [b[0] in ])]
    num_iter_list.sort(key=lambda x : mean([a[0] ]))"""

    #compare_fonte_score()
    #best_setting_iter()
    #num_iters_to_csv()
    #compare_bug2commit_simple()
    #compare_bug2commit_simple()
    #get_bug2commit_best_set(False)
    #compare_bug2commit_simple()

    """a = get_metric_dict(bug2commit=True)

    for setting, val in a.items():
        if setting[1] == 'False' and setting[3] == 'True' and setting[4] == 'True':
            print(setting, val)"""
    get_best_set_bug2commit(use_br=True)
    get_best_set()