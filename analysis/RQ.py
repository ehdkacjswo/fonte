import sys, os
import pandas as pd
from scipy.stats import wilcoxon, spearmanr

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

sys.path.append('/root/workspace/analysis/lib/')
#from util import *
from result_gen import *
from analysis import *
from compare import *

sys.path.append('/root/workspace/lib/')
from experiment_utils import get_all_commits, load_BIC_GT

sys.path.append('/root/workspace/data_collector/lib/')
from vote_util import get_style_change_commits

RESULT_DATA_DIR = '/root/workspace/data/Defects4J/result'
CORE_DATA_DIR = '/root/workspace/data/Defects4J/core'
PLT_DIR = '/root/workspace/analysis/plot'


def RQ1_():
    # Decay analysis
    print('[Decay analysis]')
    metric_dict = get_metric_dict(method='bug2commit', mode='all')
    decay_list, MRR_list, num_iter_list = [0.0, 0.1, 0.2, 0.3, 0.4], list(), list()

    for decay in decay_list:
        MRR_list.append(metric_dict[frozenset({'stage2' : 'precise', 'tracker': 'git', \
            'diff_tool': 'gumtree', 'diff_type': 'gumtree_id', 'use_br' : False, 'classify_id' : True, \
            'use_id' : True, 'decay': decay}.items())]['MRR'])

        num_iter_list.append(metric_dict[frozenset({'stage2' : 'precise', 'tracker': 'git', \
            'diff_tool': 'gumtree', 'diff_type': 'gumtree_id', 'use_br' : False, 'classify_id' : True, \
            'use_id' : True, 'beta': 0.2, 'decay': decay}.items())]['num_iter'])
        
        print(f'Decay : {decay}) MRR : {MRR_list[-1]:.3f}, num_iter : {num_iter_list[-1]:.3f}')

    # Decay to performance plot
    for metric in ['MRR', 'Number of iterations']:
        sns.lineplot(x=decay_list, y=MRR_list if metric == 'MRR' else num_iter_list)
        
        plt.ylabel(metric)
        plt.xlabel(r'$\lambda$')
        plt.xticks(ticks=decay_list)
        plt.tight_layout()
        plt.savefig(os.path.join(PLT_DIR, f"RQ1_decay_{metric}.png"))
        plt.close()
    
    # Score distribution of Jsoup-24b (Ranked first, iteration 2 -> 5)
    all_GT = load_BIC_GT("/root/workspace/data/Defects4J/BIC_dataset")
    GT = all_GT[all_GT['provenance'].str.contains("Manual", na=False)]
    
    project, pid, vid = 'Jsoup-24b', 'Jsoup', '24'
    BIC = GT.set_index(["pid", "vid"]).loc[(pid, vid), "commit"]

    # Load Fonte metrics
    with open(os.path.join(RESULT_DATA_DIR, project, 'vote', 'fonte.pkl'), 'rb') as file:
        metric_dict = pickle.load(file)
    
    fonte_df = metric_dict['precise']

    # Load commit list in order and index of BIC
    all_commits = get_all_commits(os.path.join(CORE_DATA_DIR, project))
    style_change_commits = get_style_change_commits(os.path.join(CORE_DATA_DIR, project), stage2='precise')
    commit_list = [c for c in all_commits if c in fonte_df.index and c not in style_change_commits]
    BIC_index = commit_list.index(BIC)

    fonte_metric_dict = get_metric_dict(method='fonte', mode='project')
    fonte_score_list = [float(fonte_df.loc[c, "vote"]) for c in commit_list]
    fonte_rank = fonte_metric_dict[frozenset({'stage2' : 'precise'}.items())][project]['rank']
    fonte_iter = fonte_metric_dict[frozenset({'stage2' : 'precise'}.items())][project]['num_iter']

    # Load Bug2Commit metrics
    with open(os.path.join(RESULT_DATA_DIR, project, 'vote', 'bug2commit.pkl'), 'rb') as file:
        metric_dict = pickle.load(file)
    
    bug2commit_setting = {'tracker': 'git', 'diff_tool': 'gumtree', 'diff_type': 'gumtree_id', \
        'use_br' : False, 'classify_id' : True, 'use_id' : True, 'decay': 0.1}

    bug2commit_df = metric_dict['precise'][frozenset(bug2commit_setting.items())]['all']

    bug2commit_metric_dict = get_metric_dict(method='bug2commit', mode='project')
    bug2commit_score_list = [float(bug2commit_df.loc[c, "vote"]) for c in commit_list]
    bug2commit_rank = bug2commit_metric_dict[frozenset(({'stage2' : 'precise'} | bug2commit_setting).items())][project]['rank']
    bug2commit_iter = bug2commit_metric_dict[frozenset(({'stage2' : 'precise', 'beta' : 0.2} | bug2commit_setting).items())][project]['num_iter']

    print(f'Project {project})')
    print(f'Fonte) Rank : {fonte_rank}, #Iter : {fonte_iter}, Score std : {np.std(fonte_score_list)}')
    print(f'Bug2Commit) Rank : {bug2commit_rank}, #Iter : {bug2commit_iter}, Score std : {np.std(bug2commit_score_list)}')
    
    # Draw plot
    colors = ['lightgray'] * len(fonte_score_list)
    colors[BIC_index] = 'tomato'
    x = list(range(len(fonte_score_list)))

    for ind, score_list in enumerate([fonte_score_list, bug2commit_score_list]): 
        plt.figure(figsize=(6, 3))
        sns.barplot(x=x, y=score_list, palette=colors)

        # Cleanup
        plt.xticks([], [])          # Hide x-axis ticks and labels
        plt.xlabel('')              # Remove x-axis label
        plt.ylabel('Commit score')         # Label for y-axis
        plt.tight_layout()
        plt.savefig(os.path.join(PLT_DIR, f"RQ1_score_{project}_{'fonte' if ind == 0 else 'bug2commit'}.png"))
        plt.close()

    # Number of votes per commit
    corr_dict = {'Fonte' : list(), 'Bug2Commit' : list()}
    num_vote_dict = {'BIC' : list(), 'Non_BIC' : list()}
    num_vote_all = list()
    MRR, num = 0, 0

    for _, row in tqdm(GT.iterrows()):
        pid, vid, BIC = row.pid, row.vid, row.commit

        # Load Fonte & Bug2Commit data
        with open(os.path.join(RESULT_DATA_DIR, f'{pid}-{vid}b', 'vote', 'fonte.pkl'), 'rb') as file:
            metric_dict = pickle.load(file)
        fonte_df = metric_dict['precise']

        with open(os.path.join(RESULT_DATA_DIR, f'{pid}-{vid}b', 'vote', 'bug2commit.pkl'), 'rb') as file:
            metric_dict = pickle.load(file)
        bug2commit_df = metric_dict['precise'][frozenset((bug2commit_setting | {'decay' : 0.0}).items())]['all']

        # Number of votes
        num_vote_dict['BIC'].append(int(fonte_df.loc[BIC, 'num_vote']))
        num_vote_dict['Non_BIC'].append((int(fonte_df['num_vote'].sum()) - num_vote_dict['BIC'][-1]) / (len(fonte_df) - 1) if len(fonte_df) > 1 else 0)
        num_vote_all += fonte_df['num_vote'].tolist()

        fonte_df["num_vote_rank"] = fonte_df["num_vote"].rank(ascending=False, method="max")
        MRR += 1 / int(fonte_df.loc[BIC, 'num_vote_rank'])
        num += 1
        
        # Spearman test to check monotonicity
        if np.std(fonte_df['num_vote'].tolist()) == 0:
            continue

        corr, p = spearmanr(fonte_df['num_vote'].tolist(), fonte_df['org_vote'].tolist())
        corr_dict['Fonte'].append(corr)
        if p > 0.05:
            print(f'Corr for {pid}-{vid}b:Fonte p-value {p}')

        corr, p = spearmanr(fonte_df['num_vote'].tolist(), bug2commit_df['vote'].tolist())
        corr_dict['Our approach'].append(corr)
        if p > 0.05:
            print(f'Corr for {pid}-{vid}b:Bug2Commit p-value {p}')

    for ind, (data_dict, a, b) in enumerate([(corr_dict, 'Fonte', 'Our approach'), (num_vote_dict, 'BIC', 'Non_BIC')]):
        print('BIC vs Non_BIC' if ind > 0 else 'Correlation')
        
        #
        _, p = wilcoxon(data_dict[a], data_dict[b], alternative='greater')
        print(f'WSR p-value) {p}')

        if ind > 0:
            print(f'{a}) {sum(data_dict[a]) / len(data_dict[a])}')
            print(f'{b}) {sum(num_vote_all) / len(num_vote_all)}')
            
            rel_vote_list = [x / y if y > 0 else 1 for x, y in zip(data_dict[a], data_dict[b])]
            print(f'Average ratio) {sum(rel_vote_list) / len(rel_vote_list)}')
            print(f'MRR) {MRR / num}')

        # Plot normal points
        above_x = [x for x, y in zip(data_dict[b], data_dict[a]) if y > x]
        above_y = [y for x, y in zip(data_dict[b], data_dict[a]) if y > x]

        below_x = [x for x, y in zip(data_dict[b], data_dict[a]) if y < x]
        below_y = [y for x, y in zip(data_dict[b], data_dict[a]) if y < x]

        same_x = [x for x, y in zip(data_dict[b], data_dict[a]) if y == x]
        same_y = [y for x, y in zip(data_dict[b], data_dict[a]) if y == x]

        print(f'Better) {len(above_x)}, Worse) {len(below_x)}, Same) {len(same_x)}')

        plt.figure(figsize=(6,6))

        # Scatter points
        plt.scatter(above_x, above_y, color='green')
        plt.scatter(below_x, below_y, color='red')
        plt.scatter(same_x, same_y, color='grey')
        plt.axline((0, 0), slope=1)

        if ind > 0:
            plt.xlim(0, max(data_dict[b]))
            plt.ylim(0, max(data_dict[a]))
            
        else:
            plt.xlim(0, 1)
            plt.ylim(0, 1)
            plt.xticks([0, 0.2, 0.4, 0.6, 0.8, 1])
            plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
        
        plt.xlabel(b)
        plt.ylabel(a)
        plt.tight_layout()
        plt.savefig(os.path.join(PLT_DIR, f"RQ1_vote_{'num' if ind > 0 else 'corr'}.png"))
        plt.close()

def RQ2_():
    all_GT = load_BIC_GT("/root/workspace/data/Defects4J/BIC_dataset")
    GT = all_GT[all_GT['provenance'].str.contains("Manual", na=False)]

    # Token share ratio
    setting_list = \
        [('org', {'tracker': 'git', 'diff_tool': 'base', 'diff_type': 'gumtree_id', 'classify_id' : True}), \
        ('new', {'tracker': 'git', 'diff_tool': 'gumtree', 'diff_type': 'gumtree_id', 'classify_id' : True})]
    
    # {type_pair : {id_type : {rank : {org, new} } } }
    rank_dict, ratio_dict = dict(), dict()
    
    for (key, setting) in setting_list:
        share_ratio_dict, total_metric_dict = token_share_ratio(stage2='precise', feature_setting=setting)

        for project, bug_type_dict in total_metric_dict.items():
            pid, vid = project[:-1].split('-')
            BIC = GT.set_index(["pid", "vid"]).loc[(pid, vid), "commit"]

            for bug_type, commit_type_dict in bug_type_dict.items():
                
                # Not using bug report
                if bug_type.startswith('br'):
                    continue

                for commit_type, id_type_dict in commit_type_dict.items():

                    # Fine grained tool only affects code elements
                    if commit_type not in ['class', 'method', 'variable', 'comment']:
                        continue

                    type_pair = (commit_type, bug_type)
                    rank_dict.setdefault(type_pair, {'org' : list(), 'new' : list()})
                    rank_dict[type_pair][key].append(id_type_dict['all']['rank'])

                    ratio_dict.setdefault(type_pair, \
                        {'BIC' : {'org' : list(), 'new' : list()}, 'other' : {'org' : list(), 'new' : list()}})
                    
                    total_ratio_dict = share_ratio_dict[project][bug_type][commit_type]['all']['total']
                    del total_ratio_dict[BIC]
                    ratio_dict[type_pair]['BIC'][key].append(share_ratio_dict[project][bug_type][commit_type]['all']['BIC'])
                    ratio_dict[type_pair]['other'][key] += share_ratio_dict[project][bug_type][commit_type]['all']['total'].values()
    
    print('Token share ratio MRR')

    for type_pair, rank_dict in rank_dict.items():
        print(f'Commit) {type_pair[0]}, Bug) {type_pair[1]}')
        
        org_reverse = [1 / rank for rank in rank_dict['org']]
        new_reverse = [1 / rank for rank in rank_dict['new']]

        print(f"Rank of MRR) {sum(org_reverse) / len(org_reverse):.3f} -> {sum(new_reverse) / len(new_reverse):.3f}")
        _, p = wilcoxon(rank_dict['new'], rank_dict['org'], alternative='less')
        print(f'WSR p-value) {p:.3f}\n')
    
    # Ratio change plot
    type_pair_list = list(ratio_dict.keys())
    BIC_org, BIC_new, other_org, other_new = list(), list(), list(), list()

    for type_pair in type_pair_list:
        print(type_pair)
        
        sub_dict = ratio_dict[type_pair]
        _, BIC_p = wilcoxon(sub_dict['BIC']['new'], sub_dict['BIC']['org'], alternative='greater')
        _, other_p = wilcoxon(sub_dict['other']['new'], sub_dict['other']['org'], alternative='less')
        print(f'WSR) BIC : {BIC_p}, Others : {other_p}')

        BIC_org.append(sum(sub_dict['BIC']['org']) / len(sub_dict['BIC']['org']))
        BIC_new.append(sum(sub_dict['BIC']['new']) / len(sub_dict['BIC']['new']))
        other_org.append(sum(sub_dict['other']['org']) / len(sub_dict['other']['org']))
        other_new.append(sum(sub_dict['other']['new']) / len(sub_dict['other']['new']))
        print(f'Change ratio) BIC : {(BIC_new[-1] - BIC_org[-1]) / BIC_org[-1]}, Others : {(other_new[-1] - other_org[-1]) / other_org[-1]}')

    # Plot settings
    x = np.arange(len(type_pair_list))  # label locations
    width = 0.2  # width of each bar
    fig, ax = plt.subplots(figsize=(14, 6))

    # Plot bars
    ax.bar(x - width, BIC_org, width, label='BIC (Coarse-grained)', color='tab:blue')
    ax.bar(x, BIC_new, width, label='BIC (Fine-grained)', color='tab:cyan')

    ax.bar(x + width, other_org, width, label='Non-BIC (Coarse-grained)', color='tab:red')
    ax.bar(x + 2*width, other_new, width, label='Non-BIC (Fine-grained)', color='tab:pink')

    # Labels and formatting
    ax.set_xlabel('Commit-Bug Feature Type Pair')
    ax.set_ylabel('Mean Token Share Ratio')
    #ax.set_title('Effect of Fine-Grained Differencing on Token Share Ratio')
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(type_pair_list, rotation=45)
    ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1.0))
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig(os.path.join(PLT_DIR, "RQ2_mean_token_share_ratio.png"))
    plt.close()
    
    # Score rank
    res_dict = dict()
    
    for _, row in tqdm(GT.iterrows()):
        pid, vid, BIC = row.pid, row.vid, row.commit
        proj_dir = os.path.join(RESULT_DATA_DIR, f'{pid}-{vid}b')
        
        with open(os.path.join(proj_dir, 'vote', f'bug2commit.pkl'), 'rb') as file:
            total_vote_dict = pickle.load(file)
        
        for key, setting in setting_list:
            vote_dict = total_vote_dict['precise'][frozenset(({'use_br' : False, 'use_id' : True, 'decay' : 0.1} | setting).items())]

            for type_pair_dict, vote_df in vote_dict.items():
                if type_pair_dict == 'all':
                    continue

                type_pair = (dict(type_pair_dict)['bug'], dict(type_pair_dict)['commit'])
                res_dict.setdefault(type_pair, {'org' : list(), 'new' : list()})
                res_dict[type_pair][key].append(int(vote_df.loc[BIC, 'rank']))
    
    print('Score MRR')

    for type_pair, rank_dict in res_dict.items():
        print(f'Bug) {type_pair[0]}, Commit) {type_pair[1]}')
        
        org_reverse = [1 / rank for rank in rank_dict['org']]
        new_reverse = [1 / rank for rank in rank_dict['new']]

        print(f"Rank of MRR) {sum(org_reverse) / len(org_reverse):.3f} -> {sum(new_reverse) / len(new_reverse):.3f}")
        _, p = wilcoxon(rank_dict['new'], rank_dict['org'], alternative='less')
        print(f'WSR p-value) {p:.3f}\n')
    return

def RQ3_():
    all_GT = load_BIC_GT("/root/workspace/data/Defects4J/BIC_dataset")
    GT = all_GT[all_GT['provenance'].str.contains("Manual", na=False)]

    setting = {'tracker': 'git', 'diff_tool': 'gumtree', 'diff_type': 'gumtree_id', 'use_br' : False, 'classify_id' : True, 'use_id' : True, 'decay': 0.1}
    res_dict, rank_dict = dict(), dict()
    
    # Score distribution
    for _, row in tqdm(GT.iterrows()):
        pid, vid, BIC = row.pid, row.vid, row.commit
        proj_dir = os.path.join(RESULT_DATA_DIR, f'{pid}-{vid}b')

        with open(os.path.join(proj_dir, 'vote', f'bug2commit.pkl'), 'rb') as file:
            total_vote_dict = pickle.load(file)
        
        vote_dict = total_vote_dict['precise'][frozenset(setting.items())]

        for type_pair_dict, vote_df in vote_dict.items():
            if type_pair_dict == 'all':
                continue

            bug_type, commit_type = dict(type_pair_dict)['bug'], dict(type_pair_dict)['commit']
            if commit_type not in ['class', 'method', 'variable', 'comment']:
                continue
            
            type_pair = (commit_type, bug_type)
            res_dict.setdefault(type_pair, {'BIC' : list(), 'Others' : list()})
            rank_dict.setdefault(type_pair, list())

            for commit, commit_df in vote_df.iterrows():
                
                if commit == BIC:
                    res_dict[type_pair]['BIC'].append(float(commit_df['vote']))
                    rank_dict[type_pair].append(1 / int(commit_df['rank']))
                
                else:
                    res_dict[type_pair]['Others'].append(float(commit_df['vote']))

        
        #for commit_type, is_BIC_dict in project_score_dict.items():
        #    for is_BIC, vote in is_BIC_dict.items():
        #        res_dict[commit_type][is_BIC].append(vote if is_BIC == 'BIC' else vote / num_other_commit if num_other_commit > 0 else 0)

    type_pair_list, BIC_score_list, others_score_list = list(res_dict.keys()), list(), list()
    type_pair_list.sort()
    for type_pair in type_pair_list:
        is_BIC_dict = res_dict[type_pair]
        BIC_score_list.append(sum(is_BIC_dict["BIC"]) / len(is_BIC_dict["BIC"]))
        others_score_list.append(sum(is_BIC_dict["Others"]) / len(is_BIC_dict["Others"]))
        
        print(f'{type_pair}) BIC: {BIC_score_list[-1]}, Others: {others_score_list[-1]}, Diff) {BIC_score_list[-1] / others_score_list[-1] * 100:.2f}, MRR {sum(rank_dict[type_pair]) / len(rank_dict[type_pair])}')
    
    x = np.arange(len(type_pair_list))  # label locations
    width = 0.2  # width of each bar
    fig, ax = plt.subplots(figsize=(14, 6))

    # Plot bars
    ax.bar(x, others_score_list, width, label='Non-BIC', color='tab:cyan')
    ax.bar(x + width, BIC_score_list, width, label='BIC', color='tab:red')

    for i in x:
        BIC_score, others_score = BIC_score_list[i], others_score_list[i]
        ax.text(i, max(BIC_score, others_score) + 0.001, f'{BIC_score / others_score * 100:.2f}%', ha='center', fontsize=8)

    # Labels and formatting
    ax.set_xlabel('Commit-Bug Feature Type Pair')
    ax.set_ylabel('Mean Similarity Score')
    #ax.set_title('Effect of Fine-Grained Differencing on Token Share Ratio')
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(type_pair_list, rotation=45)
    ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1.0))
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig(os.path.join(PLT_DIR, "RQ3_mean_score.png"))
    plt.close()

    return
    # Example setup
    n_programs = 146
    program_ids = [f'P{i}' for i in range(n_programs)]
    element_types = ['class', 'method', 'variable', 'comment']

    # Simulated scores: shape (4, 146)
    np.random.seed(0)
    score_matrix = np.random.rand(len(element_types), n_programs)

    # Create heatmap
    plt.figure(figsize=(20, 4))  # Wide format
    sns.heatmap(score_matrix, cmap='YlOrRd', xticklabels=10, yticklabels=element_types)

    plt.xlabel('Target Programs')
    plt.ylabel('Code Element Types')
    plt.title('Suspiciousness Score Contribution by Code Element Type and Program')
    plt.tight_layout()
    plt.show()

def RQ4_():
    setting = {'tracker': 'git', 'diff_tool': 'gumtree', 'diff_type': 'gumtree_id', 'classify_id' : True}
    token_ratio_dict, res_dict = token_share_ratio_proj(pid='Lang', vid='47', stage2='precise', feature_setting=setting, BIC=None)


def RQ(settings):
    # Print metrics
    for (diff_type, stage2, method, set_dict) in settings:
        print(f'[{diff_type}]')
        print_metric(method=method, stage2=stage2, bisect_setting=set_dict)

    # Compare settings
    
    for ind in range(len(settings) - 1):
        print(f'Comparing {settings[ind][0]}, {settings[ind + 1][0]}')
        compare_settings(org_method=settings[ind][2], new_method=settings[ind + 1][2], \
            org_setting=(settings[ind][3] | {'stage2' : settings[ind][1]}), \
            new_setting=(settings[ind + 1][3] | {'stage2' : settings[ind + 1][1]}))
    

if __name__ == "__main__":
    RQ1 = \
        [('Bug2Commit (Bug report)', 'precise', 'bug2commit', {'tracker': 'git', 'diff_tool': None, 'use_br' : False, 'beta': 0.2, 'decay': 0.0}), \
        ('Fonte', 'precise', 'fonte', {}), \
        ('FBL-BERT', 'precise', 'fbl_bert', {'beta' : 0.2}), \
        ('New Bug2Commit', 'precise', 'bug2commit', {'tracker': 'git', 'diff_tool': 'gumtree', 'diff_type': 'gumtree_id', 'use_br' : False, 'classify_id' : True, 'use_id' : True, 'beta': 0.2, 'decay': 0.1})]

    RQ1 = \
        [('Fonte', 'precise', 'fonte', {}), \
        ('New Bug2Commit', 'precise', 'bug2commit', {'tracker': 'git', 'diff_tool': 'gumtree', 'diff_type': 'gumtree_id', 'use_br' : False, 'classify_id' : True, 'use_id' : True, 'beta': 0.2, 'decay': 0.1})]

    # Fine-grained
    """
    RQ2 = \
        [('No Diff', 'precise', 'bug2commit', {'tracker': 'git', 'diff_tool': None, 'use_br' : False, 'beta': 0.1}), \
        ('Whole File', 'precise', 'bug2commit', {'tracker': 'git', 'diff_tool': 'file', 'diff_type': 'base', 'use_br' : False, 'beta': 0.1}), \
        ('Git Diff', 'precise', 'bug2commit', {'tracker': 'git', 'diff_tool': 'base', 'diff_type': 'base', 'use_br' : False, 'beta': 0.1}), \
        ('GumTree Diff', 'precise', 'bug2commit', {'tracker': 'git', 'diff_tool': 'gumtree', 'diff_type': 'base', 'use_br' : False, 'beta': 0.1})]
    """
    RQ2 = \
        [('Git Diff', 'precise', 'bug2commit', {'tracker': 'git', 'diff_tool': 'base', 'diff_type': 'base', 'use_br' : False, 'beta': 0.2, 'decay': 0.1}), \
        ('GumTree Diff', 'precise', 'bug2commit', {'tracker': 'git', 'diff_tool': 'gumtree', 'diff_type': 'base', 'use_br' : False, 'beta': 0.2, 'decay': 0.1})]
    
    # Classify identifier
    RQ3 = \
        [('No classifying', 'precise', 'bug2commit', {'tracker': 'git', 'diff_tool': 'gumtree', 'diff_type': 'gumtree_id', 'use_br' : False, 'classify_id' : False, 'use_id' : False, 'beta': 0.2, 'decay': 0.1}), \
        ('Classifying', 'precise', 'bug2commit', {'tracker': 'git', 'diff_tool': 'gumtree', 'diff_type': 'gumtree_id', 'use_br' : False, 'classify_id' : True, 'use_id' : False, 'beta': 0.2, 'decay': 0.1})]
    
    # Use full identifier
    RQ4_greedy = \
        [('No Identifier', 'precise', 'bug2commit', {'tracker': 'git', 'diff_tool': 'gumtree', 'diff_type' : 'greedy_id', 'use_br' : False, 'classify_id' : True, 'use_id' : False, 'beta': 0.2, 'decay': 0.1}), \
        ('Use Identifier', 'precise', 'bug2commit', {'tracker': 'git', 'diff_tool': 'gumtree', 'diff_type': 'greedy_id', 'use_br' : False, 'classify_id' : True, 'use_id' : True, 'beta': 0.2, 'decay': 0.1})]
    
    RQ4_gumtree = \
        [('No Identifier', 'precise', 'bug2commit', {'tracker': 'git', 'diff_tool': 'gumtree', 'diff_type' : 'gumtree_id', 'use_br' : False, 'classify_id' : True, 'use_id' : False, 'beta': 0.2, 'decay': 0.1}), \
        ('Use Identifier', 'precise', 'bug2commit', {'tracker': 'git', 'diff_tool': 'gumtree', 'diff_type': 'gumtree_id', 'use_br' : False, 'classify_id' : True, 'use_id' : True, 'beta': 0.2, 'decay': 0.1})]
    
    Ablation = \
        [('Coarse', 'skip', 'bug2commit', {'tracker': 'git', 'diff_tool': 'base', 'diff_type': 'gumtree_id', 'use_br' : False, 'classify_id' : False, 'use_id' : False, 'beta': 0.2, 'decay': 0.1}), \
        ('Coarse + Classify', 'skip', 'bug2commit', {'tracker': 'git', 'diff_tool': 'base', 'diff_type': 'gumtree_id', 'use_br' : False, 'classify_id' : True, 'use_id' : False, 'beta': 0.2, 'decay': 0.1}), \
        ('Coarse + Full', 'skip', 'bug2commit', {'tracker': 'git', 'diff_tool': 'base', 'diff_type': 'gumtree_id', 'use_br' : False, 'classify_id' : False, 'use_id' : True, 'beta': 0.2, 'decay': 0.1}), \
        ('Coarse + Classify + Full', 'skip', 'bug2commit', {'tracker': 'git', 'diff_tool': 'base', 'diff_type': 'gumtree_id', 'use_br' : False, 'classify_id' : True, 'use_id' : True, 'beta': 0.2, 'decay': 0.1}), \
        ('Fine', 'precise', 'bug2commit', {'tracker': 'git', 'diff_tool': 'gumtree', 'diff_type': 'gumtree_id', 'use_br' : False, 'classify_id' : False, 'use_id' : False, 'beta': 0.2, 'decay': 0.1}), \
        ('Fine + Classify', 'precise', 'bug2commit', {'tracker': 'git', 'diff_tool': 'gumtree', 'diff_type': 'gumtree_id', 'use_br' : False, 'classify_id' : True, 'use_id' : False, 'beta': 0.2, 'decay': 0.1}), \
        ('Fine + Full', 'precise', 'bug2commit', {'tracker': 'git', 'diff_tool': 'gumtree', 'diff_type': 'gumtree_id', 'use_br' : False, 'classify_id' : False, 'use_id' : True, 'beta': 0.2, 'decay': 0.1}), \
        ('Fine + Classify + Full', 'precise', 'bug2commit', {'tracker': 'git', 'diff_tool': 'gumtree', 'diff_type': 'gumtree_id', 'use_br' : False, 'classify_id' : True, 'use_id' : True, 'beta': 0.2, 'decay': 0.1})]

    #RQ(RQ1)
    RQ1_()
    #RQ(Ablation)
    #RQ(RQ4_gumtree)

