import argparse, os, copy, itertools, pickle, sys
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from scipy.stats import wilcoxon, friedmanchisquare
from statistics import mean
import numpy as np
from tqdm import tqdm
import pandas as pd
import seaborn as sns

sys.path.append('/root/workspace/lib/')
from experiment_utils import *

DIFF_DATA_DIR = '/root/workspace/data/Defects4J/diff'


# Compare the extra scores based on metrics
# Metrics : acc(1,2,3,5,10), MRR, Mean number of iterations (score rank)
def compare_bug2commit_simple(use_br=False):
    GT = load_BIC_GT("/root/workspace/data/Defects4J/BIC_dataset")
    tot_metric_dict = dict()
    
    # Iterate through every projects
    for folder in tqdm(os.listdir(DIFF_DATA_DIR)):
        [pid, vid] = folder[:-1].split("-")
        BIC = GT.set_index(["pid", "vid"]).loc[(pid, vid), "commit"]
        
        # Bug2Commit score and number of iterations
        scores_df = pd.read_hdf(os.path.join(DIFF_DATA_DIR, f'{folder}/scores.hdf'))
        with open(os.path.join(DIFF_DATA_DIR, f'{folder}/num_iters.pkl'), 'rb') as file:
            num_iters_dict = pickle.load(file)

        # Iterate through extra scores of every settings (use_br, use_diff, stage2, use_stopword, adddel)
        for setting, row in scores_df.iterrows():
            if not use_br and setting[0] == 'True':
                continue

            commit_df = row['commit_hash'].dropna()
            rank_df = row['rank'].dropna()

            # Index of the BIC
            BIC_ind = commit_df.loc[commit_df == BIC].index[0]
            setting_tup = tuple(setting)

            if setting_tup not in tot_metric_dict:
                tot_metric_dict[setting_tup] = dict()
            
            setting_metric_dict = tot_metric_dict[setting_tup]

            # Acc data
            BIC_rank = rank_df.loc[BIC_ind]
            for n in [1, 2, 3, 5, 10]:
                acc_str = f'acc@{n}'
                setting_metric_dict[acc_str] = setting_metric_dict.get(acc_str, 0) + (1 if BIC_rank <= n else 0)

            # MRR data
            # 130 is number of projects
            setting_metric_dict['MRR'] = setting_metric_dict.get('MRR', 0) + 1 / (BIC_rank * 130)

            # Mean of number of iterations
            # Using Bug2Commit alone could possibly cause BIC score to be 0, so use only score mode 'rank'
            setting_metric_dict['num_iters'] = setting_metric_dict.get('num_iters', 0) + \
                num_iters_dict[('None', 'rank', '(\'add\', 0.0)') + setting_tup][0] / 130

    # Possible options for each parameters
    option_dict = {
        'use_br' : ['True', 'False'] if use_br else ['False'],
        'use_diff' : ['True', 'False'],
        'stage2' : ['skip', 'True', 'False'],
        'use_stopword' : ['True', 'False'],
        'adddel' : ['add', 'del', 'all-uni', 'all-sep']
        #'adddel' : ['all-sep']
    }

    setting_list = list(itertools.product(option_dict['use_br'], option_dict['use_diff'], \
        option_dict['stage2'], option_dict['use_stopword'], option_dict['adddel']))
    
    # Compare the options of parameters
    for ind, param in enumerate(['use_br', 'use_diff', 'stage2', 'use_stopword', 'adddel']):
        if not use_br and param == 'use_br':
            continue
        print(f'Working on {param}')
        
        option_metric_dict = dict()
        param_option_list = option_dict[param]
        
        for setting in setting_list: # For every possible setting
            for option in param_option_list: # For possible options for selected parameter
                if setting[ind] == option:
                    if option not in option_metric_dict:
                        option_metric_dict[option] = dict()

                    for metric, val in tot_metric_dict[setting].items(): # Gather the metrics
                        option_metric_dict[option][metric] = \
                            option_metric_dict[option].get(metric, []) + [val]
                    break
        
        # Compare two different
        for ind1 in range(len(param_option_list)):
            for ind2 in range(ind1 + 1, len(param_option_list)):
                option1 = param_option_list[ind1]
                option2 = param_option_list[ind2]

                print(f'Comparing {option1} and {option2}')

                for metric in option_metric_dict[option1].keys():
                    val1 = mean(option_metric_dict[option1][metric])
                    val2 = mean(option_metric_dict[option2][metric])

                    try:
                        pvalue = wilcoxon(option_metric_dict[option1][metric], option_metric_dict[option2][metric]).pvalue
                    except:
                        pvalue = 'Identical'

                    print(f'{metric} for {option1} : {val1}')
                    print(f'{metric} for {option2} : {val2}')
                    print(f'P-value : {pvalue}')

        # Dictionary for drawing box plot
        metric_plot_dict = dict()

        for option in param_option_list:
            metric_dict = option_metric_dict[option]

            for metric, val in metric_dict.items():
                if metric not in metric_plot_dict:
                    metric_plot_dict[metric] = []
                
                metric_plot_dict[metric].append(val)

        for metric, val in metric_plot_dict.items():
            pvalue = friedmanchisquare(*val).pvalue if len(param_option_list) > 2 else wilcoxon(*val).pvalue
            plt.figure()
            plt.boxplot(val, tick_labels=param_option_list)
            plt.title(f"{param}-{metric}-{pvalue}")
            plt.xlabel("X-axis")
            plt.ylabel("Y-axis")
            plt.grid(True)
            plt.show()
            plt.close()  # Close the figure to free memory

def compare_bug2commit_complex(use_br=False):
    GT = load_BIC_GT("/root/workspace/data/Defects4J/BIC_dataset")
    tot_metric_dict = dict()
    
    # Iterate through every projects
    for folder in tqdm(os.listdir(DIFF_DATA_DIR)):
        [pid, vid] = folder[:-1].split("-")
        BIC = GT.set_index(["pid", "vid"]).loc[(pid, vid), "commit"]
        
        # Bug2Commit score and number of iterations
        scores_df = pd.read_hdf(os.path.join(DIFF_DATA_DIR, f'{folder}/scores.hdf'))
        with open(os.path.join(DIFF_DATA_DIR, f'{folder}/num_iters.pkl'), 'rb') as file:
            num_iters_dict = pickle.load(file)

        # Iterate through extra scores of every settings (use_br, use_diff, stage2, use_stopword, adddel)
        for setting, row in scores_df.iterrows():
            if not use_br and setting[0] == 'True':
                continue

            commit_df = row['commit_hash'].dropna()
            rank_df = row['rank'].dropna()

            # Index of the BIC
            BIC_ind = commit_df.loc[commit_df == BIC].index[0]
            setting_tup = tuple(setting)

            if setting_tup not in tot_metric_dict:
                tot_metric_dict[setting_tup] = dict()
            
            setting_metric_dict = tot_metric_dict[setting_tup]

            # Acc data
            BIC_rank = rank_df.loc[BIC_ind]
            for n in [1, 2, 3, 5, 10]:
                acc_str = f'acc@{n}'
                setting_metric_dict[acc_str] = setting_metric_dict.get(acc_str, 0) + (1 if BIC_rank <= n else 0)

            # MRR data
            # 130 is number of projects
            setting_metric_dict['MRR'] = setting_metric_dict.get('MRR', 0) + 1 / (BIC_rank * 130)

            # Mean of number of iterations
            # Using Bug2Commit alone could possibly cause BIC score to be 0, so use only score mode 'rank'
            setting_metric_dict['num_iters'] = setting_metric_dict.get('num_iters', 0) + \
                num_iters_dict[('None', 'rank', '(\'add\', 0.0)') + setting_tup][0] / 130

    # Possible options for each parameters
    option_dict = {
        'use_br' : ['True', 'False'] if use_br else ['False'],
        'use_diff' : ['True', 'False'],
        'stage2' : ['skip', 'True', 'False'],
        'use_stopword' : ['True', 'False'],
        'adddel' : ['add', 'del', 'all-uni', 'all-sep']
        #'adddel' : ['all-sep']
    }

    setting_list = list(itertools.product(option_dict['use_br'], option_dict['use_diff'], \
        option_dict['stage2'], option_dict['use_stopword'], option_dict['adddel']))
    
    # Compare the options of parameters
    for ind, param in enumerate(['use_br', 'use_diff', 'stage2', 'use_stopword', 'adddel']):
        if not use_br and param == 'use_br':
            continue
        print(f'Working on {param}')
        
        option_metric_dict = dict()
        param_option_list = option_dict[param]
        
        for setting in setting_list: # For every possible setting
            for option in param_option_list: # For possible options for selected parameter
                if setting[ind] == option:
                    if option not in option_metric_dict:
                        option_metric_dict[option] = dict()

                    for metric, val in tot_metric_dict[setting].items(): # Gather the metrics
                        option_metric_dict[option][metric] = \
                            option_metric_dict[option].get(metric, []) + [val]
                    break
        
        # Compare two different
        for ind1 in range(len(param_option_list)):
            for ind2 in range(ind1 + 1, len(param_option_list)):
                option1 = param_option_list[ind1]
                option2 = param_option_list[ind2]

                print(f'Comparing {option1} and {option2}')

                for metric in option_metric_dict[option1].keys():
                    val1 = mean(option_metric_dict[option1][metric])
                    val2 = mean(option_metric_dict[option2][metric])

                    try:
                        pvalue = wilcoxon(option_metric_dict[option1][metric], option_metric_dict[option2][metric]).pvalue
                    except:
                        pvalue = 'Identical'

                    print(f'{metric} for {option1} : {val1}')
                    print(f'{metric} for {option2} : {val2}')
                    print(f'P-value : {pvalue}')

        # Dictionary for drawing box plot
        metric_plot_dict = dict()

        for option in param_option_list:
            metric_dict = option_metric_dict[option]

            for metric, val in metric_dict.items():
                if metric not in metric_plot_dict:
                    metric_plot_dict[metric] = []
                
                metric_plot_dict[metric].append(val)

        for metric, val in metric_plot_dict.items():
            pvalue = friedmanchisquare(*val).pvalue if len(param_option_list) > 2 else wilcoxon(*val).pvalue
            plt.figure()
            plt.boxplot(val, tick_labels=param_option_list)
            plt.title(f"{param}-{metric}-{pvalue}")
            plt.xlabel("X-axis")
            plt.ylabel("Y-axis")
            plt.grid(True)
            plt.show()
            plt.close()  # Close the figure to free memory

def compare_all_simple():
    if os.path.isfile('/root/workspace/analyze/data/fonte_metrics.pkl'):
        with open('/root/workspace/analyze/data/fonte_metrics.pkl', 'rb') as file:
            metric_dict = pickle.load(file)

    else:
        GT = load_BIC_GT("/root/workspace/data/Defects4J/BIC_dataset")
        metric_dict = dict()

        for folder in tqdm(os.listdir(DIFF_DATA_DIR)):
            [pid, vid] = folder[:-1].split("-")
            BIC = GT.set_index(["pid", "vid"]).loc[(pid, vid), "commit"]
            
            fonte_scores_df = pd.read_hdf(os.path.join(DIFF_DATA_DIR, f'{folder}/fonte_scores.hdf'))

            # Iterate through extra scores of every settings
            for setting, row in fonte_scores_df.iterrows():
                commit_df = row['commit'].dropna()
                score_df = row['vote'].dropna()
                rank_df = score_df.rank(method='max', ascending=False)

                # Index of the BIC
                BIC_ind = commit_df.loc[commit_df == BIC].index[0]
                setting_tup = tuple(setting)

                if setting_tup not in metric_dict:
                    metric_dict[setting_tup] = dict()
                
                setting_metric_dict = metric_dict[setting_tup]

                # Acc data
                BIC_rank = rank_df.loc[BIC_ind]
                for n in [1, 2, 3, 5, 10]:
                    acc_str = f'acc@{n}'
                    setting_metric_dict[acc_str] = setting_metric_dict.get(acc_str, 0) + (1 if BIC_rank <= n else 0)

                # MRR data
                # 130 is number of projects
                BIC_rev_rank = 1 / BIC_rank
                setting_metric_dict['MRR'] = setting_metric_dict.get('MRR', 0) + BIC_rev_rank / 130

                # Score data
                BIC_score = score_df.loc[BIC_ind]
                score_mean = score_df.mean()
                
                setting_metric_dict['score_diff'] = \
                    setting_metric_dict.get('score_diff', []) + [BIC_score - score_mean]
                setting_metric_dict['score_ratio'] = \
                    setting_metric_dict.get('score_ratio', []) + [0 if score_mean == 0 else BIC_score / score_mean]
                
                # Rank data
                rev_rank_df = 1 / rank_df
                rev_rank_mean = rev_rank_df.mean()

                setting_metric_dict['rev_rank_diff'] = \
                    setting_metric_dict.get('rev_rank_diff', []) + [BIC_rev_rank - rev_rank_mean]
                setting_metric_dict['rev_rank_ratio'] = \
                    setting_metric_dict.get('rev_rank_ratio', []) + [BIC_rev_rank / rev_rank_mean]

                # Both data
                both_df = score_df / rank_df
                BIC_both = both_df.loc[BIC_ind]
                both_mean = both_df.mean()

                setting_metric_dict['both_diff'] = \
                    setting_metric_dict.get('both_diff', []) + [BIC_both - both_mean]
                setting_metric_dict['both_ratio'] = \
                    setting_metric_dict.get('both_ratio', []) + [0 if both_mean == 0 else BIC_both / both_mean]
        
        with open('/root/workspace/analyze/data/fonte_metrics.pkl', 'wb') as file:
            pickle.dump(metric_dict, file)

    # Possible options for each parameters
    """option_dict = {
        #'use_br' : ['True', 'False'],
        'use_br' : ['False'],
        'use_diff' : ['True', 'False'],
        'stage2' : ['skip', 'True', 'False'],
        'use_stopword' : ['True', 'False'],
        'adddel' : ['add', 'del', 'all-uni', 'all-sep']
        #'adddel' : ['all-sep']
    }

    setting_list = list(itertools.product(option_dict['use_br'], option_dict['use_diff'], \
        option_dict['stage2'], option_dict['use_stopword'], option_dict['adddel']))
    
    # Compare the options of parameters
    for ind, param in enumerate(param_list = ['HSFL', 'score_mode', 'ensemble', 'use_br', 'use_diff', 'stage2', 'use_stopword', 'adddel']):
        print(f'Working on {param}')
        
        option_metric_dict = dict()
        param_option_list = option_dict[param]
        
        for setting in setting_list: # For every possible setting
            for option in param_option_list: # For possible options for selected parameter
                if setting[ind] == option:
                    if setting not in option_metric_dict:
                        option_metric_dict[option] = dict()

                    for metric, val in metric_dict[setting].items(): # Gather the metrics
                        option_metric_dict[option][metric] = \
                            option_metric_dict[option].get(metric, []) + (val if type(val) is list else [val])
                    break
        
        for ind1 in range(len(param_option_list)):
            for ind2 in range(ind1 + 1, len(param_option_list)):
                option1 = param_option_list[ind1]
                option2 = param_option_list[ind2]

                print(f'Comparing {option1} and {option2}')

                for metric in option_metric_dict[option1].keys():
                    val1 = mean(option_metric_dict[option1][metric])
                    val2 = mean(option_metric_dict[option2][metric])

                    try:
                        pvalue = kruskal(option_metric_dict[option1][metric], option_metric_dict[option2][metric]).pvalue
                    except:
                        pvalue = 'Identical'

                    print(f'{metric} for {option1} : {val1}')
                    print(f'{metric} for {option2} : {val2}')
                    print(f'P-value : {pvalue}')"""
        
# For fixed target parameter(s), check whether 
def check_param_iter(target_dict: dict, strict = True):
    with open('/root/workspace/num_iters.pkl', 'rb') as file:
        num_iter_dict = pickle.load(file)

    # Possible options for each parameters
    option_dict = {
        'HSFL' : ['True', 'False', 'None'],
        'score_mode' : ['score', 'rank', 'both', 'None'],
        'ensemble' : ['mul', '(\'add\', 0.1)', '(\'add\', 0.2)', '(\'add\', 0.3)', \
        '(\'add\', 0.4)',  '(\'add\', 0.5)', '(\'add\', 0.6)', '(\'add\', 0.7)', '(\'add\', 0.8)', \
        '(\'add\', 0.9)',],
        'use_br' : ['False', 'None'],
        'use_diff' : ['True', 'False', 'None'],
        'stage2' : ['skip', 'True', 'False'],
        'use_stopword' : ['True', 'False', 'None'],
        'adddel' : ['add', 'all-uni', 'all-sep', 'None']
    }

    param_ind_dict = {'HSFL' : 0, 'score_mode' : 1, 'ensemble' : 2, 'use_br' : 3, \
        'use_diff' : 4, 'stage2' : 5, 'use_stopword' : 6, 'adddel' : 7}
    
    param_list = ['HSFL', 'score_mode', 'ensemble', 'use_br', 'use_diff', 'stage2', 'use_stopword', 'adddel']

    # 아닌거 목록 만들고, 맞는거 목록 만들어서 쭉 돌리기?
    # List of possible options for target parameters
    target_param_list = list(target_dict.keys())
    target_option_list = [option_dict[param] for param in target_param_list]
    target_option_list = list(itertools.product(*target_option_list))

    # List of possible options for non-target parameters
    non_target_param_list = [param for param in param_list if param not in target_dict]
    non_target_option_list = [option_dict[param] for param in non_target_param_list]
    non_target_option_list = list(itertools.product(*non_target_option_list))

    # Iterate through every non-target options
    for non_target_option in non_target_option_list:

        # Setting with target option
        target_setting = tuple([target_dict[param] if param in target_dict else \
            non_target_option[non_target_param_list.index(param)] for param in param_list])
        
        # Invalid setting
        if target_setting not in num_iter_dict:
            continue
        
        target_num_iter_list = [num_iter[0] for num_iter in num_iter_dict[target_setting]]

        # Ignore 
        if 'None' in target_num_iter_list:
            continue
        
        # Iterate through every target options
        for target_option in target_option_list:
            if 'None' in target_option or target_option == tuple(target_dict.values()):
                continue
            
            # Setting with non-target option
            non_target_setting = tuple([target_option[target_param_list.index(param)] \
                if param in target_dict else target_setting[ind] for ind, param in enumerate(param_list)])
            
            non_target_num_iter_list = [num_iter[0] for num_iter in num_iter_dict[non_target_setting]]

            # Ignore 
            if 'None' in non_target_num_iter_list:
                continue
            
            if strict:
                for target_num_iter, non_target_num_iter in zip(target_num_iter_list, non_target_num_iter_list):
                    if target_num_iter[0] > non_target_num_iter:
                        print(non_target_setting)
                        break
            
            else:
                if mean(target_num_iter_list) > mean(non_target_num_iter_list):
                    print(f'{target_setting} : {mean(target_num_iter_list)}')
                    print(f'{non_target_setting} : {mean(non_target_num_iter_list)}')

    """
    for setting, num_iters1 in num_iters_dict.items():
        # Check if current setting is target or not
        is_target = True

        for (param, option) in target_list:
            if setting[param_ind_dict[param]] != option:
                is_target = False
                break
        
        if not is_target:
            continue

        target_option_list = [option_dict[param] for (param, _) in target_list]
        target_option_list = list(itertools.product(*target_option_list))

        for target_option in target_option_list:
            is_target = True

            for ind, (_, option) in enumerate(target_list):
                if target_option[ind] != option:
                    is_target = False
                    break"""
            
# Generate csv file to feed ARTool
# Combination of score_mode('score', 'both') & ensemble('mul', 'add'0.0)
# have to be ignored to prevent possible error (BIC has 0 score)
# But ARTool needs full combination for every parameters
# So generate two csv files that ignores each of them

def num_iters_to_csv():
    with open('/root/workspace/num_iters.pkl', 'rb') as file:
        num_iter_dict = pickle.load(file)
    
    # List of options that have to be ignored
    ignore_score_mode_list = ['score', 'both']
    ignore_ensemble_list = ['mul', '(\'add\', 0.0)']

    no_score_mode_rows = []
    no_ensemble_rows = []
    
    # HSFL is none when ensemble = 'add'0.0
    # score_mode, use_br, use_diff, use_stopword, adddel is none when ensemble = 'add'1.0
    for (HSFL, score_mode, ensemble, use_br, use_diff, stage2, use_stopword, adddel), num_iter_list in num_iter_dict.items():
        if score_mode not in ignore_score_mode_list:
            continue




def metric_to_csv():
    for project in tqdm(os.listdir(DIFF_DATA_DIR)):
        with open(os.path.join(DIFF_DATA_DIR, project, 'num_iters.pkl'), 'rb') as file:
            num_iter_dict = pickle.load(file)
    
        fonte_scores_df = pd.read_hdf(os.path.join(DIFF_DATA_DIR, f'{project}/fonte_scores.hdf'))


    
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
    compare_all_simple()
