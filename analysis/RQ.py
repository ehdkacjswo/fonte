import sys, os
import pandas as pd
from scipy.stats import wilcoxon

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

# Compare Fonte with new approach
def RQ1_():
    #metric_dict = get_metric_dict(method=method, mode='project')
    all_GT = load_BIC_GT("/root/workspace/data/Defects4J/BIC_dataset")
    GT = all_GT[all_GT['provenance'].str.contains("Manual", na=False)]

    bug2commit_list, fonte_list = list(), list()
    
    for _, row in tqdm(GT.iterrows()):
        pid, vid, BIC = row.pid, row.vid, row.commit
        proj_dir = os.path.join(RESULT_DATA_DIR, f'{pid}-{vid}b')
        
        with open(os.path.join(proj_dir, 'vote', f'bug2commit.pkl'), 'rb') as file:
            bug2commit_dict = pickle.load(file)['precise']\
                [frozenset({'tracker': 'git', 'diff_tool': 'gumtree', 'diff_type': 'gumtree_id', 'use_br' : False, 'classify_id' : True, 'use_id' : True, 'decay': 0.1}.items())]['all']
        
        with open(os.path.join(proj_dir, 'vote', f'fonte.pkl'), 'rb') as file:
            fonte_dict = pickle.load(file)['precise']

        bug2commit_BIC, fonte_BIC = float(bug2commit_dict['vote'].get(BIC)), float(fonte_dict['vote'].get(BIC))
        bug2commit_mean, fonte_mean = float(bug2commit_dict['vote'].mean()), float(fonte_dict['vote'].mean())
        
        bug2commit_list.append(bug2commit_BIC / bug2commit_mean if bug2commit_mean > 0 else 1)
        fonte_list.append(fonte_BIC / fonte_mean if fonte_mean > 0 else 1)
    
    _, p = wilcoxon(fonte_list, bug2commit_list, alternative='greater')
    print(p)

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

    print(f'Project {project}) Fonte){fonte_rank}, {fonte_iter}, Bug2Commit) {bug2commit_rank}, {bug2commit_iter}')
    
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
    vote_dict = {'fonte' : {'all' : list(), 'BIC' : list()}, 'bug2commit' : {'all' : list(), 'BIC' : list()}}
    num_vote_dict = {'all' : list(), 'BIC' : list()}
    BIC_idx = list()

    for _, row in tqdm(GT.iterrows()):
        pid, vid, BIC = row.pid, row.vid, row.commit

        # Load Fonte & Bug2Commit data
        with open(os.path.join(RESULT_DATA_DIR, f'{pid}-{vid}b', 'vote', 'fonte.pkl'), 'rb') as file:
            metric_dict = pickle.load(file)
        fonte_df = metric_dict['precise']

        with open(os.path.join(RESULT_DATA_DIR, f'{pid}-{vid}b', 'vote', 'bug2commit.pkl'), 'rb') as file:
            metric_dict = pickle.load(file)
        bug2commit_df = metric_dict['precise'][frozenset((bug2commit_setting | {'decay' : 0.0}).items())]['all']

        # Get vote, num_vote data
        for commit, commit_df in fonte_df.iterrows():
            vote_dict['fonte']['all'].append(float(commit_df['org_vote']))
            vote_dict['bug2commit']['all'].append(float(bug2commit_df.loc[commit, 'vote']))
            num_vote_dict['all'].append(int(commit_df['num_vote']))

            if commit == BIC:
                vote_dict['fonte']['BIC'].append(float(commit_df['org_vote']))
                vote_dict['bug2commit']['BIC'].append(float(bug2commit_df.loc[commit, 'vote']))
                num_vote_dict['BIC'].append(int(commit_df['num_vote']))
                BIC_idx.append(len(vote_dict['fonte']['all']) - 1)
    
    for method in ['fonte', 'bug2commit']:
        x, y = np.array(num_vote_dict['all']), np.array(vote_dict[method]['all'])

        highlight = np.zeros_like(x, dtype=bool)
        highlight[BIC_idx] = True

        model = LinearRegression(fit_intercept=False)
        model.fit(x.reshape(-1, 1), y)
        a = model.coef_[0]

        # Generate line for y = ax
        x_line = np.linspace(x.min(), x.max(), 100)
        y_line = a * x_line

        # Plot
        plt.figure(figsize=(8, 6))

        # Plot normal points
        plt.scatter(x[~highlight], y[~highlight], color='gray', label='Normal', s=80)

        # Plot highlighted points
        plt.scatter(x[highlight], y[highlight], color='red', label='Highlighted', s=80)

        # Fitted line
        plt.plot(x_line, y_line, color='blue', linewidth=2, label=f'$y = {a:.2f}x$')

        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Dotplot with Highlighted Indices and y = ax Fit')
        #plt.legend()
        #plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(PLT_DIR, f"RQ1_num_to_vote_{method}.png"))
        plt.close()


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
        [('No Identifier', 'precise', 'bug2commit', {'tracker': 'git', 'diff_tool': 'gumtree', 'diff_type' : 'greedy_id', 'use_br' : False, 'use_id' : False, 'beta': 0.2, 'decay': 0.1}), \
        ('Use Identifier', 'precise', 'bug2commit', {'tracker': 'git', 'diff_tool': 'gumtree', 'diff_type': 'greedy_id', 'use_br' : False, 'use_id' : True, 'beta': 0.2, 'decay': 0.1})]
    
    RQ4_gumtree = \
        [('No Identifier', 'precise', 'bug2commit', {'tracker': 'git', 'diff_tool': 'gumtree', 'diff_type' : 'gumtree_id', 'use_br' : False, 'classify_id' : True, 'use_id' : False, 'beta': 0.2, 'decay': 0.1}), \
        ('Use Identifier', 'precise', 'bug2commit', {'tracker': 'git', 'diff_tool': 'gumtree', 'diff_type': 'gumtree_id', 'use_br' : False, 'classify_id' : True, 'use_id' : True, 'beta': 0.2, 'decay': 0.1})]
    
    Ablation = \
        [('Base', 'precise', 'bug2commit', {'tracker': 'git', 'diff_tool': 'gumtree', 'diff_type': 'gumtree_id', 'use_br' : False, 'classify_id' : False, 'use_id' : False, 'beta': 0.2, 'decay': 0.1}), \
        ('Classify', 'precise', 'bug2commit', {'tracker': 'git', 'diff_tool': 'gumtree', 'diff_type': 'gumtree_id', 'use_br' : False, 'classify_id' : True, 'use_id' : False, 'beta': 0.2, 'decay': 0.1}), \
        ('Full', 'precise', 'bug2commit', {'tracker': 'git', 'diff_tool': 'gumtree', 'diff_type': 'gumtree_id', 'use_br' : False, 'classify_id' : False, 'use_id' : True, 'beta': 0.2, 'decay': 0.1}), \
        ('Classify + Full', 'precise', 'bug2commit', {'tracker': 'git', 'diff_tool': 'gumtree', 'diff_type': 'gumtree_id', 'use_br' : False, 'classify_id' : True, 'use_id' : True, 'beta': 0.2, 'decay': 0.1})]

    #RQ(RQ1)
    RQ1_()