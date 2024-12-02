import argparse, os, copy, itertools, pickle, sys
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from scipy.stats import wilcoxon, ttest_rel
from statistics import mean
import numpy
import pandas as pd
import csv

sys.path.append('/root/workspace/diff_util/lib/')
from encoder import savepath_postfix
from tqdm import tqdm

sys.path.append('/root/workspace/lib/')
from experiment_utils import *

DIFF_DATA_DIR = '/root/workspace/data/Defects4J/diff'

voting_functions = {
    # key: (alpha, tau)
    (1, 'max'): (lambda r: r.score/r.max_rank),
    (0, 'max'): (lambda r: 1/r.max_rank),
    (1, 'dense'): (lambda r: r.score/r.dense_rank),
    (0, 'dense'): (lambda r: 1/r.dense_rank),
}

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

def metrics_to_csv(bug2commit=True):
    savepath = f"/root/workspace/analyze/data/{'bug2commit' if bug2commit else 'all'}/metrics.csv"

    """if os.path.isfile(savepath):
        print(f'{savepath} already exists!')
        return"""
    
    GT = load_BIC_GT("/root/workspace/data/Defects4J/BIC_dataset")
    project_metric_dict = dict()

    for project in tqdm(os.listdir(DIFF_DATA_DIR)):
        [pid, vid] = project[:-1].split("-")
        BIC = GT.set_index(["pid", "vid"]).loc[(pid, vid), "commit"]
        project_dir = os.path.join(DIFF_DATA_DIR, project)

        metric_dict = dict()

        with open(os.path.join(project_dir, 'num_iters.pkl'), 'rb') as file:
            num_iter_dict = pickle.load(file)
    
        fonte_scores_df = pd.read_hdf(os.path.join(project_dir, 'fonte_scores.hdf'))

        # Iterate through extra scores of every settings
        # Settings : ['HSFL', 'score_mode', 'ensemble', 'use_br', 'use_diff', 'stage2', 'use_stopword', 'adddel']
        for setting, row in fonte_scores_df.iterrows():

            # Consider Bug2Commit score only cases
            if bug2commit and setting[2] != '(\'add\', 0.0)':
                continue
            
            if not bug2commit and (setting[2] == '(\'add\', 1.0)' or setting[2] == '(\'add\', 0.0)'):
                continue

            commit_df = row['commit'].dropna()
            score_df = row['vote'].dropna()
            rank_df = score_df.rank(method='max', ascending=False)

            # Index of the BIC
            BIC_ind = commit_df.loc[commit_df == BIC].index[0]
            BIC_rank = rank_df.loc[BIC_ind]
            
            if bug2commit:
                setting_tup = tuple(option for ind, option in enumerate(tuple(setting)) if ind not in [0, 2])
            else:
                setting_tup = tuple(setting)

            if setting_tup not in metric_dict:
                metric_dict[setting_tup] = dict()
            else:
                print('ERRORRRRR!!!!!!!')
            
            metric_dict[setting_tup]['rank'] = int(BIC_rank)
            metric_dict[setting_tup]['num_iters'] = num_iter_dict[tuple(setting)][1]
        
        project_metric_dict[project] = metric_dict
    
    with open(savepath, 'w', newline='') as file:
        writer = csv.writer(file)
        if bug2commit:
            field = ['project', 'score_mode', 'use_br', 'use_diff', 'stage2', 'use_stopword', 'adddel', 'DependentName', 'DependentValue']
            writer.writerow(field)

            for project, metric_dict in project_metric_dict.items():
                for (score_mode, use_br, use_diff, stage2, use_stopword, adddel), val in metric_dict.items():
                    writer.writerow([project, score_mode, use_br, use_diff, stage2, use_stopword, adddel, 'rank', val['rank']])
                    writer.writerow([project, score_mode, use_br, use_diff, stage2, use_stopword, adddel, 'num_iters', val['num_iters']])
            
        else:
            field = ['project', 'HSFL', 'score_mode', 'ensemble', 'use_br', 'use_diff', 'stage2', 'use_stopword', 'adddel', 'DependentName', 'DependentValue']
            writer.writerow(field)

            for project, metric_dict in project_metric_dict.items():
                for (HSFL, score_mode, ensemble, use_br, use_diff, stage2, use_stopword, adddel), val in metric_dict.items():
                    writer.writerow([project, HSFL, score_mode, ensemble, use_br, use_diff, stage2, use_stopword, adddel, 'rank', val['rank']])
                    writer.writerow([project, HSFL, score_mode, ensemble, use_br, use_diff, stage2, use_stopword, adddel, 'num_iters', val['num_iters']])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute commit scores")
    parser.add_argument('--tool', type=str, default="git",
        help="history retrieval tool, git or shovel (default: git)")
    parser.add_argument('--formula', type=str, default="Ochiai",
        help="SBFL formula (default: Ochiai)")
    parser.add_argument('--alpha', type=int, default=0,
        help="alpha (default: 0)")
    parser.add_argument('--tau', type=str, default="max",
        help="tau (default: max)")
    parser.add_argument('--lamb', type=float, default=0.1,
        help="lambda (default: 0.1)")
    parser.add_argument('--skip-stage-2', action="store_true",
        help="skiping stage 2 (default: False)")
    parser.add_argument('--no-openrewrite', action="store_true",
        help="not using openrewrite in Stage 2(default: False)")
    parser.add_argument('--output', '-o',
        help="path to output file (example: output.csv)")
    args = parser.parse_args()

    assert args.alpha in [0, 1]
    assert args.tau in ["max", "dense"]
    assert 0 <= args.lamb < 1

    # Generate iteration data
    """for folder in tqdm(os.listdir(DIFF_DATA_DIR)):
        # Get BIC data
        print(f'Fonte_score_eval : Working on {folder}')
        [pid, vid] = folder[:-1].split("-")
        results_dict = score_eval_all(pid, vid, args.tool, args.formula, args.lamb, voting_functions[(args.alpha, args.tau)])
        results_df = pd.concat(results_dict, \
            names=['HSFL', 'score_mode', 'ensemble', 'use_br', 'use_diff', 'stage2', 'use_stopword', 'adddel']).unstack()
        
        results_df.to_hdf(os.path.join(DIFF_DATA_DIR, f'{folder}/fonte_scores.hdf'), key='data', mode='w')"""
    
    # Generate iteration data
    """num_iters_dict = dict()
    for folder in tqdm(os.listdir(DIFF_DATA_DIR)):
        print(f'Weighted bisection : Working on {folder}')
        [pid, vid] = folder[:-1].split("-")
        result_dict = bisection_all(pid, vid)

        with open(os.path.join(DIFF_DATA_DIR, folder, 'num_iters.pkl'), 'wb') as file:
            pickle.dump(result_dict, file)"""
    
    metrics_to_csv(False)
    metrics_to_csv(True)