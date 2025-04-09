import argparse, os, copy, itertools, pickle, sys, json, subprocess, html
import pandas as pd
from scipy.stats import wilcoxon
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import math
from interval import interval, inf
from tqdm import tqdm

sys.path.append('/root/workspace/lib/')
from experiment_utils import load_BIC_GT

sys.path.append('/root/workspace/analysis/lib')
from result_gen import get_metric_dict

sys.path.append('/root/workspace/data_collector/lib/')
from encoder import *
from utils import CustomInterval, get_src_from_commit

#sys.path.append('/root/workspace/analysis/lib/')
#from util import *
#from stage2 import get_style_change_data

RESULT_DATA_DIR = '/root/workspace/data/Defects4J/result'
CORE_DATA_DIR = '/root/workspace/data/Defects4J/core'
BASE_DATA_DIR = '/root/workspace/data/Defects4J/baseline'
DIFF_DATA_DIR = '/root/workspace/data/Defects4J/diff'

# Contains methods that compare settings (By project / all)
# Possible settings : stage2, tracker, diff_tool, diff_type, use_id, classify_id, use_br
# cmp_use_id
# cmp_classify_id

# Print how using IDs affect the voting for feature pairs + 
# Vote setting doesn't have to contain 'use_id'
# How using id changes performance + Identifier distributions
def cmp_use_id_proj(pid='Jsoup', vid='25', stage2='precise', \
    feature_setting={'tracker': 'git', 'diff_tool' : 'base', 'diff_type' : 'greedy_id', 'adddel' : 'add'}, \
    vote_setting={'use_br' : True}):
    
    GT = load_BIC_GT("/root/workspace/data/Defects4J/BIC_dataset")
    BIC = GT.set_index(["pid", "vid"]).loc[(pid, vid), "commit"]

    # Load data (Feature, bug_feature)
    with open(os.path.join(DIFF_DATA_DIR, f'{pid}-{vid}b', 'feature.pkl'), 'rb') as file:
        feature_dict = pickle.load(file)
    
    with open(os.path.join(DIFF_DATA_DIR, f'{pid}-{vid}b', 'bug_feature.pkl'), 'rb') as file:
        bug_feature_dict = pickle.load(file)
    feature_dict, encoder_dict, bug_feature_dict = load_feature_data(pid, vid)
    encoder_setting = feature_setting.copy()
    del encoder_setting['adddel']
    feature_data, encoder = feature_dict[stage2][frozenset(feature_setting.items())], encoder_dict[stage2][frozenset(encoder_setting.items())]

    # Get vocabulary from encoder
    id_vocab = encoder.id_vocab
    id_vocab = {ind : word for word, ind in id_vocab.items()}

    # Load voting results
    with open(os.path.join(RESULT_DATA_DIR, f'{pid}-{vid}b', 'vote', 'bug2commit.pkl'), 'rb') as file:
        vote_dict = pickle.load(file)
    
    new_vote_dict = vote_dict[stage2][frozenset((feature_setting | vote_setting | {'use_id' : True}).items())]
    org_vote_dict = vote_dict[stage2][frozenset((feature_setting | vote_setting | {'use_id' : False}).items())]

    # Get the set of commit types
    commit_type_set = next(iter(feature_data.values())).keys()
    num_commits = len(feature_data)
    
    print(f'BIC rank : {int(org_vote_dict["all"].loc[BIC, "rank"])}/{num_commits} -> {int(new_vote_dict["all"].loc[BIC, "rank"])}/{num_commits}')

    for bug_type, bug_feature in bug_feature_dict.items():
        #if bug_type.startswith('br'):
        #    continue

        bug_feature, _ = encoder.encode(bug_feature, update_vocab=False, mode='code' if bug_type == 'test_code' else 'text')
        bug_feature = [(word, freq) for word, freq in bug_feature.items()]
        bug_feature.sort(key=lambda x:x[1], reverse=True)
        
        for commit_type in commit_type_set:
            print(f'\nBug) {bug_type}, Commit) {commit_type}')
            new_vote_df = new_vote_dict[frozenset({'commit' : commit_type, 'bug' : bug_type}.items())]
            org_vote_df = org_vote_dict[frozenset({'commit' : commit_type, 'bug' : bug_type}.items())]

            print(f'BIC rank : {int(org_vote_df.loc[BIC, "rank"])}/{num_commits} -> {int(new_vote_df.loc[BIC, "rank"])}/{num_commits}')
            
            BIC_freq_dict, commit_freq_dict = dict(), dict()

            for commit, commit_feature in feature_data.items():
                if commit == BIC:
                    BIC_vec = commit_feature.get(commit_type, dict()).get('id', Counter())
                    BIC_len = sum(BIC_vec.values())

                    if BIC_len == 0:
                        print('BIC doesn\'t contain any ID')
                        continue
                    else:
                        for word, freq in BIC_vec.items():
                            BIC_freq_dict[word] = freq / BIC_len

                else:
                    commit_vec = commit_feature.get(commit_type, dict()).get('id', Counter())
                    commit_len = sum(commit_vec.values())

                    if commit_len == 0:
                        continue
                    else:
                        for word, freq in commit_vec.items():
                            commit_freq_dict[word] = commit_freq_dict.get(word, 0) + freq / (commit_len * (num_commits - 1))
            
            for (word, freq) in bug_feature:
                BIC_freq, commit_freq = BIC_freq_dict.get(word, 0), commit_freq_dict.get(word, 0)
                
                if BIC_freq > 0 or commit_freq > 0:
                    print(f'{id_vocab[word]} : {freq} BIC : {BIC_freq * 100:.2f}%, else : {commit_freq * 100:.2f}%')

# Print how using IDs affect the voting for feature pairs + 
# Vote setting doesn't have to contain 'use_id'
# How using id changes performance
#vote_setting={'tracker': 'git', 'diff_tool' : 'base', 'diff_type' : 'gumtree_id', 'classify_id' : True, 'adddel' : 'add', 'use_br' : True}
def cmp_use_id_all(stage2='precise', \
    vote_setting={'tracker': 'git', 'diff_tool' : 'base', 'diff_type' : 'gumtree_id', 'classify_id' : True, 'adddel' : 'add', 'use_br' : True}):

    all_GT = load_BIC_GT("/root/workspace/data/Defects4J/BIC_dataset")
    GT = all_GT[all_GT['provenance'].str.contains("Manual", na=False)]
    res_dict = dict()
    
    for _, row in tqdm(GT.iterrows()):
        pid, vid, BIC = row.pid, row.vid, row.commit

        # Load voting results
        with open(os.path.join(RESULT_DATA_DIR, f'{pid}-{vid}b', 'vote', 'bug2commit.pkl'), 'rb') as file:
            vote_dict = pickle.load(file)
        
        vote_dict = \
            {'org': vote_dict[stage2][frozenset((vote_setting | {'use_id' : False}).items())], \
            'new' : vote_dict[stage2][frozenset((vote_setting | {'use_id' : True}).items())]}

        for type_pair in vote_dict['org'].keys() & vote_dict['new'].keys():
            res_dict.setdefault(type_pair, {'num_proj' : 0, \
                'org' : {'rank' : list(), 'BIC_vote' : list(), 'mean_vote' : list(), 'MRR' : 0, 'acc@1' : 0, 'acc@2' : 0, 'acc@3' : 0, 'acc@5' : 0, 'acc@10' : 0}, \
                'new' : {'rank' : list(), 'BIC_vote' : list(), 'mean_vote' : list(), 'MRR' : 0, 'acc@1' : 0, 'acc@2' : 0, 'acc@3' : 0, 'acc@5' : 0, 'acc@10' : 0}})
            res_dict[type_pair]['num_proj'] += 1

            for vote_type, metric_dict in res_dict[type_pair].items():
                if vote_type == 'num_proj':
                    continue

                rank = int(vote_dict[vote_type][type_pair].loc[BIC, 'rank'])
                BIC_vote = float(vote_dict[vote_type][type_pair].loc[BIC, 'vote'])
                mean_vote = float(vote_dict[vote_type][type_pair]['vote'].mean())

                metric_dict['rank'].append(rank)
                metric_dict['BIC_vote'].append(BIC_vote)
                metric_dict['mean_vote'].append(mean_vote)

                metric_dict['MRR'] += 1 / rank
                metric_dict['acc@1'] += 1 if rank <= 1 else 0
                metric_dict['acc@2'] += 1 if rank <= 2 else 0
                metric_dict['acc@3'] += 1 if rank <= 3 else 0
                metric_dict['acc@5'] += 1 if rank <= 5 else 0
                metric_dict['acc@10'] += 1 if rank <= 10 else 0
    
    for type_pair, metric_dict in res_dict.items():
        if type_pair == 'all':
            print('\nALL')
        else:
            print(f"\nBug) {dict(type_pair)['bug']}, Commit) {dict(type_pair)['commit']}")
        
        # Wilcoxon signed rank test 
        try:
            _, better_p = wilcoxon(metric_dict['org']['rank'], metric_dict['new']['rank'], alternative='greater')
            _, worse_p = wilcoxon(metric_dict['org']['rank'], metric_dict['new']['rank'], alternative='less')
            print(f'Rank P-value) Better : {better_p:.3f}, Worse : {worse_p:.3f}')
        except:
            print('Identical ranks')
        
        try:
            _, better_p = wilcoxon(metric_dict['org']['BIC_vote'], metric_dict['new']['BIC_vote'], alternative='less')
            _, worse_p = wilcoxon(metric_dict['org']['BIC_vote'], metric_dict['new']['BIC_vote'], alternative='greater')
            print(f'Vote P-value) Better : {better_p:.3f}, Worse : {worse_p:.3f}')
        except:
            print('Identical vote')

        org_BIC_ratio_list = [BIC_ratio / total_ratio if total_ratio > 0 else 1 \
            for (BIC_ratio, total_ratio) in zip(metric_dict['org']['BIC_vote'], metric_dict['org']['mean_vote'])]
        new_BIC_ratio_list = [BIC_ratio / total_ratio if total_ratio > 0 else 1 \
            for (BIC_ratio, total_ratio) in zip(metric_dict['new']['BIC_vote'], metric_dict['new']['mean_vote'])]
        
        try:
            _, better_p = wilcoxon(org_BIC_ratio_list, new_BIC_ratio_list, alternative='less')
            _, worse_p = wilcoxon(org_BIC_ratio_list, new_BIC_ratio_list, alternative='greater')
            print(f'Vote ratio P-value) Better : {better_p:.3f}, Worse : {worse_p:.3f}')
        except:
            print('Identical vote ratio')
        
        # Print metrics
        mean_org_vote = sum(metric_dict['org']['BIC_vote']) / len(metric_dict['org']['BIC_vote'])
        mean_new_vote = sum(metric_dict['new']['BIC_vote']) / len(metric_dict['new']['BIC_vote'])
        print(f"Vote : {mean_org_vote:.3f} -> {mean_new_vote:.3f} ({mean_new_vote - mean_org_vote:.3f})")

        mean_org_ratio = sum(org_BIC_ratio_list) / len(org_BIC_ratio_list)
        mean_new_ratio = sum(new_BIC_ratio_list) / len(new_BIC_ratio_list)
        print(f"Vote ratio : {mean_org_ratio:.3f} -> {mean_new_ratio:.3f} ({mean_new_ratio - mean_org_ratio:.3f})")
        
        for metric_type in metric_dict['org'].keys():
            if metric_type in ['rank', 'BIC_vote', 'mean_vote']:
                continue
            
            org_metric = metric_dict['org'][metric_type] / (metric_dict['num_proj'] if metric_type == 'MRR' else 1)
            new_metric = metric_dict['new'][metric_type] / (metric_dict['num_proj'] if metric_type == 'MRR' else 1)
            
            print(f"{metric_type} : {org_metric:.3f} -> {new_metric:.3f} ({new_metric - org_metric:.3f})")

# Check the number of each identifiers
# Doesn't work for all_sep (all_sep voting is not working correctly too)
def cmp_classify_id_all(stage2='precise', \
    feature_setting={'tracker': 'git', 'diff_tool' : 'gumtree', 'diff_type' : 'gumtree_id', 'adddel' : 'add'}, \
    vote_setting={'use_id' : True, 'use_br' : False}):
    
    all_GT = load_BIC_GT("/root/workspace/data/Defects4J/BIC_dataset")
    GT = all_GT[all_GT['provenance'].str.contains("Manual", na=False)]
    res_dict = dict()
    total_id_dist, BIC_id_dist = {'class' : 0, 'method' : 0, 'variable' : 0}, {'class' : 0, 'method' : 0, 'variable' : 0} # Different on all_sep
    
    for _, row in tqdm(GT.iterrows()):
        pid, vid, BIC = row.pid, row.vid, row.commit

        # Load voting results
        with open(os.path.join(DIFF_DATA_DIR, f'{pid}-{vid}b', 'feature.pkl'), 'rb') as file:
            feature_dict = pickle.load(file)

        with open(os.path.join(RESULT_DATA_DIR, f'{pid}-{vid}b', 'vote', 'bug2commit.pkl'), 'rb') as file:
            vote_dict = pickle.load(file)
        
        # Count the number of identifiers
        feature_dict = feature_dict[stage2][frozenset(feature_setting.items())]
        total_id_cnt, BIC_id_cnt = {'class' : 0, 'method' : 0, 'variable' : 0}, {'class' : 0, 'method' : 0, 'variable' : 0}

        for commit, commit_feature_dict in feature_dict.items():
            for commit_type, feature_vec in commit_feature_dict.items():
                if commit_type in total_id_cnt:
                    id_cnt = sum(feature_vec['id'].values())
                    total_id_cnt[commit_type] += id_cnt
                    
                    if commit == BIC:
                        BIC_id_cnt[commit_type] += id_cnt
        
        # Maybe identifier missing
        for id_type in total_id_cnt.keys():
            total_id_dist[id_type] += total_id_cnt[id_type] / sum(total_id_cnt.values())
            #BIC_id_dist[id_type] += BIC_id_cnt[id_type] / sum(BIC_id_cnt.values())
        
        vote_dict = \
            {True: vote_dict[stage2][frozenset((feature_setting | vote_setting | {'classify_id' : True}).items())], \
            False : vote_dict[stage2][frozenset((feature_setting | vote_setting | {'classify_id' : False}).items())]}

        for classify_id, sub_vote_dict in vote_dict.items():
            for type_pair in sub_vote_dict.keys():
                if type_pair == 'all':
                    continue
                commit_type, bug_type = dict(type_pair)['commit'], dict(type_pair)['bug']
                if (classify_id and commit_type not in set(['class', 'method', 'variable'])) or \
                    (not classify_id and commit_type != 'id_all'):
                    continue

                res_dict.setdefault(bug_type, dict())
                res_dict[bug_type].setdefault(commit_type, \
                    {'num_proj' : 0, 'BIC_vote' : list(), 'mean_vote' : list(), 'rank' : list(), \
                    'MRR' : 0, 'acc@1' : 0, 'acc@2' : 0, 'acc@3' : 0, 'acc@5' : 0, 'acc@10' : 0})
                
                # Get metric
                metric_dict = res_dict[bug_type][commit_type]
                metric_dict['num_proj'] += 1

                rank = int(sub_vote_dict[type_pair].loc[BIC, 'rank'])
                vote = float(sub_vote_dict[type_pair].loc[BIC, 'vote'])
                vote_mean = float(sub_vote_dict[type_pair]['vote'].mean())

                metric_dict['rank'].append(rank)
                metric_dict['BIC_vote'].append(vote)
                metric_dict['mean_vote'].append(vote_mean)

                metric_dict['MRR'] += 1 / rank
                metric_dict['acc@1'] += 1 if rank <= 1 else 0
                metric_dict['acc@2'] += 1 if rank <= 2 else 0
                metric_dict['acc@3'] += 1 if rank <= 3 else 0
                metric_dict['acc@5'] += 1 if rank <= 5 else 0
                metric_dict['acc@10'] += 1 if rank <= 10 else 0
    
    # Print num of identifiers
    print(f"All) Class:{total_id_dist['class'] / len(GT)}, Method:{total_id_dist['method'] / len(GT)}, Variable:{total_id_dist['variable'] / len(GT)}")
    #print(f"BIC) Class:{BIC_id_cnt['class'] / len(GT)}, Method:{BIC_id_cnt['method'] / len(GT)}, Variable:{BIC_id_cnt['variable'] / len(GT)}")
    
    for bug_type, commit_type_dict in res_dict.items():

        for commit_type, metric_dict in commit_type_dict.items():
            print(f"\nBug) {bug_type}, Commit) {commit_type}")
            
            w, p = wilcoxon(metric_dict['org']['rank'], metric_dict['new']['rank'], alternative='greater')
            print(f'P-value : {p}')
            
            for metric_type, metric in metric_dict.items():
                if metric_type in ['rank', 'num_proj', 'BIC_vote', 'mean_vote']:
                    continue
                
                print(f"{metric_type} : {metric / (metric_dict['num_proj'] if metric_type == 'MRR' else 1)})")


# Compare how the settings affect the rank of each commit & bug feature types
# Two settings must have same encoder (vocab), so 'tracker', 'diff_tool', 'diff_type' has to be identical
# Two settings must have same type pairs, so 'use_br', 'classify_id' has to be identical
def cmp_type_pair(stage2='precise',
    base_setting={'tracker' : 'git', 'diff_tool' : 'base', 'diff_type' : 'greedy_id', 'use_br' : False}, \
    org_setting = {'adddel' : 'add', 'use_id' : False}, \
    new_setting = {'adddel' : 'add', 'use_id' : True}):

    # Load manual data only
    all_GT = load_BIC_GT("/root/workspace/data/Defects4J/BIC_dataset")
    GT = all_GT[all_GT['provenance'].str.contains("Manual", na=False)]

    # Initialize settings
    org_setting = frozenset((base_setting | org_setting).items())
    new_setting = frozenset((base_setting | new_setting).items())

    res_dict, rank_list_dict = dict(), dict()

    # Iterate through projects
    for _, row in GT.iterrows():
        pid, vid, BIC = row.pid, row.vid, row.commit
        with open(os.path.join(RESULT_DATA_DIR, f'{pid}-{vid}b', 'vote', 'bug2commit.pkl'), 'rb') as file:
            vote_dict = pickle.load(file)
        
        org_vote_df, new_vote_df = vote_dict[stage2][org_setting], vote_dict[stage2][new_setting]

        for type_pair in org_vote_df.keys():

            # In fact, some type pair could be missing based on setting
            # But haven't decided how to handle such cases
            org_rank = org_vote_df[type_pair].loc[BIC, 'rank']
            new_rank = new_vote_df[type_pair].loc[BIC, 'rank']

            res_dict.setdefault(type_pair, {'better' : 0, 'same' : 0, 'worse' : 0, 'better_sum' : 0, 'worse_sum' : 0})
            rank_list_dict.setdefault(type_pair, list())

            # Save data
            rank_list_dict[type_pair].append(org_rank - new_rank)

            if org_rank > new_rank:
                res_dict[type_pair]['better'] += 1
                res_dict[type_pair]['better_sum'] += org_rank - new_rank
            elif org_rank < new_rank:
                res_dict[type_pair]['worse'] += 1
                res_dict[type_pair]['worse_sum'] += new_rank - org_rank
            else:
                res_dict[type_pair]['same'] += 1

    for type_pair, type_pair_res in res_dict.items():
        if type_pair == 'all':
            print('All')
        else:
            print(f"Commit) {dict(type_pair)['commit']}, Bug) {dict(type_pair)['bug']}")

        print(f"Better : {type_pair_res['better']}({type_pair_res['better_sum']}), Worse : {type_pair_res['worse']}({type_pair_res['worse_sum']}), Same : {type_pair_res['same']}")

        if type_pair_res['better'] > 0 or type_pair_res['worse'] > 0:
            w, p = wilcoxon(rank_list_dict[type_pair], alternative='greater')
            print("WSR (Imrpoved)", w, p)
            w, p = wilcoxon(rank_list_dict[type_pair], alternative='less')
            print("WSR (Worsen)", w, p)

# Compare the settings
def compare_settings(org_method, new_method, org_setting, new_setting):
    org_metric_dict = get_metric_dict(method=org_method, mode='project')
    #org_metric = metric_converter(org_metric_dict[org_setting])

    new_metric_dict = get_metric_dict(method=new_method, mode='project')
    #new_metric = metric_converter(new_metric_dict[new_setting])

    # Bug2Commit의 경우, num_iter에만 beta가 있다

    for metric_key in ['rank', 'num_iter']:
        print(f'Metric) {metric_key}')

        org_metric_setting, new_metric_setting = org_setting.copy(), new_setting.copy()

        # Bug2Commit uses extra setting (beta) for num_iter
        if metric_key == 'rank':
            if org_method == 'bug2commit':
                org_metric_setting.pop('beta', None)
            
            if new_method == 'bug2commit':
                new_metric_setting.pop('beta', None)
        
        org_metric_setting = frozenset(org_metric_setting.items())
        new_metric_setting = frozenset(new_metric_setting.items())
        
        #print(org_metric_dict.keys())
        org_metric = org_metric_dict[org_metric_setting]
        new_metric = new_metric_dict[new_metric_setting]

        # Project level comparison
        num_better, num_same, num_worse = 0, 0, 0
        better, worse = 0, 0
        better_dict, worse_dict = dict(), dict()

        for project in org_metric.keys():
            org_value = org_metric[project][metric_key]
            new_value = new_metric[project][metric_key]

            if org_value == new_value:
                num_same += 1

            elif org_value > new_value:
                num_better += 1
                better += org_value - new_value
                better_dict[project] = org_value - new_value
            
            else:
                num_worse += 1
                worse += new_value - org_value
                worse_dict[project] = new_value - org_value
        
        print(f'Total: {num_better + num_worse + num_same}, Better: {num_better} ({better}), Worse: {num_worse} ({worse}), Same: {num_same}')

        print('Better projects)')
        for project, value in sorted(better_dict.items(), key=lambda item: item[1], reverse=True):
            print(f'{project}) {org_metric[project][metric_key]} {new_metric[project][metric_key]}')
        
        print('Worse projects)')
        for project, value in sorted(worse_dict.items(), key=lambda item: item[1], reverse=True):
            print(f'{project}) {org_metric[project][metric_key]} {new_metric[project][metric_key]}')
        
        print('Best project)')
        (best_proj, best_val) = max(better_dict.items(), key=lambda x: x[1])
        org_val = org_metric[best_proj][metric_key]
        print(f'{best_proj} : {org_val} > {org_val - best_val}')

        print('Worst project)')
        (worst_proj, worst_val) = max(worse_dict.items(), key=lambda x: x[1])
        org_val = org_metric[worst_proj][metric_key]
        print(f'{worst_proj} : {org_val} > {org_val + worst_val}')

        # Python 3.7+ maintains insertion order
        # Since projects are added following GT rows, the orders are the same
        org_list = [data[metric_key] for data in org_metric.values()]
        new_list = [data[metric_key] for data in new_metric.values()]

        cost_saving = [a - b for a, b in zip(org_list, new_list)]

        plt.figure(figsize=(9, 2))
        plt.title(f"Improved {metric_key}")

        cost_saving = list(reversed(sorted(cost_saving)))

        #To confirm that the median of the differences can be assumed to be positive, we use:
        w, p = wilcoxon(cost_saving, alternative='greater')
        print("Wilcoxon signed rank test (Excluding zeros)", w, p)

        w, p = wilcoxon(cost_saving, zero_method="pratt", alternative='greater')
        print("Wilcoxon signed rank test (Including zeros)", w, p)
        N = len(cost_saving)

        plt.bar(range(0, N), cost_saving,
            color=["red" if d < 0 else "green" for d in cost_saving])
        plt.axhline(0, color="black")

        #plt.yticks(range(min(cost_saving), max(cost_saving)+1))

        plt.axvspan(-0.5, num_better - 0.5, facecolor='green', alpha=0.1)
        plt.axvspan(num_better + num_same -0.5, N-0.5, facecolor='red', alpha=0.1)

        #if reduced > 0.05:
        #    plt.text(N * reduced/2 - 0.5, max(cost_saving)-1, f"{reduced*100:.1f}%", horizontalalignment="center")
        #if same > 0.05:
        #    plt.text(N * (reduced + same/2) - 0.5, max(cost_saving)-1, f"{same*100:.1f}%", horizontalalignment="center")
        #if increased > 0.05:
        #    plt.text(N * (reduced + same + increased/2) - 0.5, max(cost_saving)-1, f"{increased*100:.1f}%", horizontalalignment="center")

        plt.xlim((0-0.5, N-0.5))
        ax = plt.gca()
        ax.get_xaxis().set_visible(False)

        plt.axhline(np.mean(cost_saving), color="black", linestyle="--", label=f"Average improved {metric_key}: {np.mean(cost_saving).round(1)}")
        print(f"Average improved {metric_key}", np.mean(cost_saving))
        plt.legend(loc="upper right")

        savepath = os.path.join(f'/root/workspace/analysis/data/{new_method}', f"{metric_key}.pdf")
        plt.savefig(savepath, bbox_inches="tight")
        print(f"Saved to {savepath}")
        #plt.show()

# Print data of each ID
def cmp_id_type(stage2='precise', \
    feature_setting={'tracker': 'git', 'diff_tool' : 'base', 'diff_type' : 'gumtree_id', 'adddel' : 'add'}, \
    vote_setting={'use_br' : True, 'classify_id' : True}):

    # Load manual data only
    all_GT = load_BIC_GT("/root/workspace/data/Defects4J/BIC_dataset")
    GT = all_GT[all_GT['provenance'].str.contains("Manual", na=False)]

    # Initialize settings
    org_setting = frozenset((base_setting | org_setting).items())
    new_setting = frozenset((base_setting | new_setting).items())

    res_dict, rank_list_dict = dict(), dict()

    # Iterate through projects
    for _, row in GT.iterrows():
        pid, vid, BIC = row.pid, row.vid, row.commit

        with open(os.path.join(DIFF_DATA_DIR, f'{pid}-{vid}b', 'feature.pkl'), 'rb') as file:
            feature_dict = pickle.load(file)

        with open(os.path.join(RESULT_DATA_DIR, f'{pid}-{vid}b', 'vote', 'bug2commit.pkl'), 'rb') as file:
            vote_dict = pickle.load(file)
        
        feature_data = feature_dict[stage2][frozenset(feature_setting.items())]
        vote_data = vote_dict[stage2][frozenset((feature_setting | vote_setting).items())]

if __name__ == "__main__":
    # Compare greedy_id using
    org_setting = frozenset({'stage2' : 'precise', 'tracker' : 'git', 'diff_tool' : 'gumtree', 'diff_type' : 'greedy_id', 'adddel' : 'add', 'use_id' : True, 'use_br' : False, 'beta' : 0.1}.items())
    new_setting = frozenset({'stage2' : 'precise', 'tracker' : 'git', 'diff_tool' : 'gumtree', 'diff_type' : 'gumtree_id', 'adddel' : 'add', 'use_id' : True, 'classify_id' : True, 'use_br' : False, 'beta' : 0.1}.items())
    compare_settings('bug2commit', 'bug2commit', org_setting, new_setting)

    # Compare GumTree ID using
    #org_setting = frozenset({'stage2' : 'precise', 'tracker' : 'git', 'diff_tool' : 'gumtree', 'diff_type' : 'gumtree_id', 'adddel' : 'all_uni', 'use_id' : False, 'classify_id' : True, 'use_br' : False, 'beta' : 0.1}.items())
    #new_setting = frozenset({'stage2' : 'precise', 'tracker' : 'git', 'diff_tool' : 'gumtree', 'diff_type' : 'gumtree_id', 'adddel' : 'all_uni', 'use_id' : True, 'classify_id' : True, 'use_br' : False, 'beta' : 0.1}.items())
    #compare_settings('bug2commit', 'bug2commit', org_setting, new_setting)

    # Compare GumTree classification
    #org_setting = frozenset({'stage2' : 'precise', 'tracker' : 'git', 'diff_tool' : 'gumtree', 'diff_type' : 'gumtree_id', 'adddel' : 'all_uni', 'use_id' : True, 'classify_id' : False, 'use_br' : False, 'beta' : 0.1}.items())
    #new_setting = frozenset({'stage2' : 'precise', 'tracker' : 'git', 'diff_tool' : 'gumtree', 'diff_type' : 'gumtree_id', 'adddel' : 'all_uni', 'use_id' : True, 'classify_id' : True, 'use_br' : False, 'beta' : 0.1}.items())
    #compare_settings('bug2commit', 'bug2commit', org_setting, new_setting)

    #org_setting = frozenset({'stage2' : 'precise'}.items())
    #new_setting = frozenset({'stage2' : 'precise', 'tracker' : 'git', 'diff_tool' : 'base', 'diff_type' : 'greedy_id', 'adddel' : 'add', 'use_id' : True, 'use_br' : False, 'beta' : 1.6}.items())
    #compare_settings('fonte', 'ensemble', org_setting, new_setting)
    
    #cmp_use_id_all()
    
    # Check how using ID affects the performance
    #cmp_use_id_all(stage2='precise', \
    #    vote_setting={'tracker': 'git', 'diff_tool' : 'base', 'diff_type' : 'greedy_id', 'adddel' : 'add', 'use_br' : True})
    
    #cmp_classify_id_all()