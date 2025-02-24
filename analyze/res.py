import argparse, os, copy, itertools, pickle, sys, json, subprocess
import pandas as pd
from scipy.stats import wilcoxon
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import math

sys.path.append('/root/workspace/lib/')
from experiment_utils import load_BIC_GT

sys.path.append('/root/workspace/analyze/lib')
from result_gen import get_metric_dict

sys.path.append('/root/workspace/data_collector/lib/')
from encoder import *
from gumtree import CustomInterval

sys.path.append('/root/workspace/data_collector/tool/')
from stage2 import get_style_change_data

RESULT_DATA_DIR = '/root/workspace/data/Defects4J/result'
CORE_DATA_DIR = '/root/workspace/data/Defects4J/core'
BASE_DATA_DIR = '/root/workspace/data/Defects4J/baseline'
DIFF_DATA_DIR = '/root/workspace/data/Defects4J/diff'

# Print vocab
def abc(project, stage2, setting):
    # Load data
    with open(os.path.join(f'/root/workspace/data/Defects4J/diff/{project}', 'feature.pkl'), 'rb') as file:
        feature_dict = pickle.load(file)
    
    with open(os.path.join(f'/root/workspace/data/Defects4J/diff/{project}', 'vocab.pkl'), 'rb') as file:
        vocab_dict = pickle.load(file)
    
    GT = load_BIC_GT("/root/workspace/data/Defects4J/BIC_dataset")
    pid, vid = project[:-1].split('-')
    BIC = GT.set_index(["pid", "vid"]).loc[(pid, vid), "commit"]

    # No diff not allowed
    setting = dict(setting)
    if setting['diff_type'] == 'no_diff':
        num_feature = 3 if setting['adddel'] == 'all-sep' else 2
        setting['diff_type'] = 'git'
    else:
        num_feature = math.inf
    
    vocab_setting = copy.deepcopy(setting)
    del vocab_setting['adddel']
    
    vocab_setting = frozenset(vocab_setting.items())
    setting = frozenset(setting.items())
    
    feature_list = feature_dict[stage2][setting][BIC]
    vocab = vocab_dict[stage2][vocab_setting]
    encoder = Encoder(vocab)

    # Encode bug query
    with open(os.path.join(CORE_DATA_DIR, project, "failing_tests"), "r") as file:
        bug_query = file.read()
    
    bug_query = encoder.encode(bug_query.strip(), use_stopword=dict(setting)['use_stopword'], update_vocab=False)
    bug_query = {ind : freq for (ind, freq) in bug_query}

    BIC_dict = dict()
    for feature_ind, feature in enumerate(feature_list):
        if feature_ind >= num_feature:
            break
        for (ind, freq) in feature:
            BIC_dict[ind] = BIC_dict.get(ind, 0) + freq

    # Print
    vocab = {ind: word for word, ind in vocab.items()}

    for ind, BIC_freq in BIC_dict.items():
        bug_freq = bug_query.get(ind, None)

        if bug_freq is not None:
            print(f'Word: {vocab[ind]}, Bug: {bug_freq}, BIC: {BIC_freq}')

def check_filtering():
    GT = load_BIC_GT("/root/workspace/data/Defects4J/BIC_dataset")
    res_dict = dict()

    # Iterate through projects
    for _, row in GT.iterrows():
        #pid, vid, BIC = row.pid, row.vid, row.commit
        pid = 'Closure'
        vid = '2'
        BIC = GT.set_index(["pid", "vid"]).loc[(pid, vid), "commit"]

        # Ignore Closure-131b
        if pid == 'Closure' and vid == '131':
            continue

        # Load data
        with open(os.path.join(f'/root/workspace/data/Defects4J/diff/{pid}-{vid}b', 'feature.pkl'), 'rb') as file:
            feature_dict = pickle.load(file)
        
        with open(os.path.join(f'/root/workspace/data/Defects4J/diff/{pid}-{vid}b', 'vocab.pkl'), 'rb') as file:
            vocab_dict = pickle.load(file)
        
        git_dict = feature_dict['precise'][frozenset({'use_stopword': True, 'adddel': 'all-uni', 'diff_type': 'git'}.items())]
        gumtree_dict = feature_dict['precise'][frozenset({'use_stopword': True, 'adddel': 'all-uni', 'diff_type': 'gumtree_class'}.items())]

        vocab = vocab_dict['precise'][frozenset({'diff_type': 'git', 'use_stopword': True}.items())]

        # Encode bug query
        encoder = Encoder(vocab)
        
        with open(os.path.join(CORE_DATA_DIR, f'{pid}-{vid}b', "failing_tests"), "r") as file:
            bug_query = file.read()
        
        bug_query = encoder.encode(bug_query.strip(), use_stopword=True, update_vocab=False)
        bug_query.sort(key=lambda x: x[1], reverse=True)

        # Classify removed / remaining tokens (Consider only diff)
        # By tokenization, sometimes gumtree contains other tokens than git
        # For now, new query is empty
        git_token_set = set()
        for commit, feature_list in git_dict.items():
            git_token_set.update([x[0] for x in feature_list[2]])
            git_dict[commit] = feature_list[2]
        
        gumtree_token_set = set()
        for commit, feature_list in gumtree_dict.items():
            sum_feature = sum_encode(feature_list[2], sum_encode(feature_list[3], sum_encode(feature_list[4], feature_list[5])))
            gumtree_token_set.update([x[0] for x in sum_feature])
            gumtree_dict[commit] = sum_feature
        
        # Set of removed/remaining/new tokens
        removed_token_set = git_token_set - gumtree_token_set
        remain_token_set = git_token_set & gumtree_token_set
        new_token_set = gumtree_token_set - git_token_set

        if len(new_token_set):
            print(pid, vid)
            print(new_token_set)

        # Tokens of query removed/remaining/new
        removed_query = [x for x in bug_query if x[0] in removed_token_set]
        remain_query = [x for x in bug_query if x[0] in remain_token_set]
        new_query = [x for x in bug_query if x[0] in new_token_set]
        
        # 
        git_BIC, git_else = [], [], [], []
        git_remain_BIC, git_remain_else, gumtree_remain_BIC, gumtree_remain_else = [], [], [], []
        git_removed_BIC, git_removed_else = [], []
        commit_num = len(git_dict) - 1
        
        for commit, git_feature in git_dict.items():
            gumtree_feature = gumtree_dict[commit]

            git = {word : freq for (word, freq) in git_feature}
            gumtree = {word : freq for (word, freq) in gumtree_feature}

            if commit == BIC:
                for ind, (word, freq) in enumerate(removed_query | remain_query):
                    git_BIC.append(git.get(word, 0))

                for ind, (word, freq) in enumerate(removed_query):
                    git_removed_BIC.append(git.get(word, 0))
                
                for ind, (word, freq) in enumerate(remain_query):
                    git_remain_BIC.append(git.get(word, 0))
                    gumtree_remain_BIC.append(gumtree.get(word, 0))

            else:
                for ind, (word, freq) in enumerate(removed_query | remain_query):
                    if len(git_else) <= ind:
                        git_else.append(git.get(word, 0) / commit_num)
                    else:
                        git_else[ind] += git.get(word, 0) / commit_num

                for ind, (word, freq) in enumerate(removed_query):
                    if len(git_removed_else) <= ind:
                        git_removed_else.append(git.get(word, 0) / commit_num)
                    else:
                        git_removed_else[ind] += git.get(word, 0) / commit_num
                
                for ind, (word, freq) in enumerate(remain_query):
                    if len(git_remain_else) <= ind:
                        git_remain_else.append(git.get(word, 0) / commit_num)
                        gumtree_remain_else.apend(gumtree.get(word, 0) / commit_num)
                    else:
                        git_remain_else += git.get(word, 0) / commit_num
                        gumtree_remain_else += gumtree.get(word, 0) / commit_num
        
        break

# metric_dict = {project : {metric_key : metric_value}}
# Convert to {metric_key : {project : metric_value}}
def metric_converter(metric_dict):
    res_dict = defaultdict(dict)

    for project, metrics in metric_dict.items():
        for metric_key, metric_value in metrics.items():
            res_dict[metric_key][project] = metric_value
    
    return dict(res_dict)

def compare_settings(org_method, new_method, org_setting, new_setting):
    org_metric_dict = get_metric_dict(method=org_method, mode='project')
    org_metric = metric_converter(org_metric_dict[org_setting])

    new_metric_dict = get_metric_dict(method=new_method, mode='project')
    new_metric = metric_converter(new_metric_dict[new_setting])

    for metric_key in org_metric.keys():
        print(f'Metric) {metric_key}')

        # Project level comparison
        num_better, num_same, num_worse = 0, 0, 0
        better, worse = 0, 0
        better_dict, worse_dict = dict(), dict()

        for project, org_value in org_metric[metric_key].items():
            new_value = new_metric[metric_key][project]

            if new_value < org_value:
                num_better += 1
                better += org_value - new_value
                better_dict[project] = org_value - new_value
            
            elif new_value > org_value:
                num_worse += 1
                worse += new_value - org_value
                worse_dict[project] = new_value - org_value
            
            else:
                num_same += 1
        
        print(f'Total: {num_better + num_worse + num_same}, Better: {num_better} ({better}), Worse: {num_worse} ({worse}), Same: {num_same}')

        print('Better projects)')
        for project, value in sorted(better_dict.items(), key=lambda item: item[1], reverse=True):
            print(f'{project}) {org_metric[metric_key][project]} {new_metric[metric_key][project]}')
        
        print('Worse projects)')
        for project, value in sorted(worse_dict.items(), key=lambda item: item[1], reverse=True):
            print(f'{project}) {org_metric[metric_key][project]} {new_metric[metric_key][project]}')
        
        print('Best project)')
        (best_proj, best_val) = max(better_dict.items(), key=lambda x: x[1])
        org_val = org_metric[metric_key][best_proj]
        print(f'{best_proj} : {org_val} > {org_val - best_val}')

        print('Worst project)')
        (worst_proj, worst_val) = max(worse_dict.items(), key=lambda x: x[1])
        org_val = org_metric[metric_key][worst_proj]
        print(f'{worst_proj} : {org_val} > {org_val + worst_val}')

        # Python 3.7+ maintains insertion order
        # Since projects are added following GT rows, the orders are the same
        org_list = list(org_metric[metric_key].values())
        new_list = list(new_metric[metric_key].values())

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

        savepath = os.path.join(f'/root/workspace/analyze/data/{new_method}', f"{metric_key}.pdf")
        plt.savefig(savepath, bbox_inches="tight")
        print(f"Saved to {savepath}")
        #plt.show()

# Compare the frequencies of BIC and other commits
def check_freq(pid, vid, stage2, setting):
    GT = load_BIC_GT("/root/workspace/data/Defects4J/BIC_dataset")
    BIC = GT.set_index(["pid", "vid"]).loc[(pid, vid), "commit"]

    # Load data
    with open(os.path.join(f'/root/workspace/data/Defects4J/diff/{pid}-{vid}b', 'feature.pkl'), 'rb') as file:
        feature_dict = pickle.load(file)
    
    with open(os.path.join(f'/root/workspace/data/Defects4J/diff/{pid}-{vid}b', 'vocab.pkl'), 'rb') as file:
        vocab_dict = pickle.load(file)
    
    vocab_setting = dict(copy.deepcopy(setting))
    del vocab_setting['adddel']

    commit_dict = feature_dict[stage2][setting] # {commit : feature}
    vocab = vocab_dict[stage2][frozenset(vocab_setting.items())]

    # Encode bug query
    encoder = Encoder(vocab)
    
    with open(os.path.join(CORE_DATA_DIR, f'{pid}-{vid}b', "failing_tests"), "r") as file:
        bug_query = file.read()
    
    bug_query = encoder.encode(bug_query.strip(), use_stopword=True, update_vocab=False)

    # Count frequencies
    num_other = len(commit_dict) - 1
    print('Total commits) ', num_other + 1)
    vocab = {ind : word for word, ind in vocab.items()}

    # Type of target feature on certain index
    # Can be modified when feature gets changed for dictionary
    diff_type = vocab_setting['diff_type']
    
    if diff_type == 'git' or diff_type == 'file' or diff_type == 'gumtree_base':
        feature_type_dict = {2 : diff_type}
    
    elif diff_type == 'gumtree_class':
        feature_type_dict = {2 : 'class', 3 : 'method', 4 : 'variable', 5 : 'comment'}

    for ind, feature_type in feature_type_dict.items():
        print('Feature type) ', feature_type)
        
        word_dict = {word : {'freq' : freq, 'BIC' : 0, 'all' : 0, 'num_doc' : 0} for (word, freq) in bug_query}
        sum_len = 0

        for commit, feature_list in commit_dict.items():
            vec = []
            for ind in feature_type_dict.keys():
                vec = sum_encode(vec, feature_list[ind])

            doc_len = 0

            for (word, freq) in vec:
                doc_len += freq
                if word in word_dict:
                    if commit == BIC:
                        word_dict[word]['BIC'] = freq

                    word_dict[word]['num_doc'] += 1
                    word_dict[word]['all'] += freq
                
            if commit == BIC:
                print('BIC document length) ', doc_len)
            
            sum_len += doc_len
        
        print('Mean document length) ', sum_len / (num_other + 1))
            
        word_dict = dict(filter(lambda x: x[1]['num_doc'] > 0 and x[1]['num_doc'] * 2 < num_other + 1, word_dict.items()))
        #word_dict = dict(filter(lambda x: x[1]['num_doc'] > 0, word_dict.items()))
        word_dict = {word : data | {'all' : data['all'] / data['num_doc']} for word, data in word_dict.items()}
        
        for word, sub_dict in sorted(word_dict.items(), key=lambda x: x[1]['freq'], reverse=True):
            print(f'{vocab[word]}) {sub_dict}')
    
    word_dict = {word : {'freq' : freq, 'BIC' : 0, 'all' : 0, 'num_doc' : 0} for (word, freq) in bug_query}
    sum_len = 0

    for commit, feature_list in commit_dict.items():
        vec = []
        for ind in feature_type_dict.keys():
            vec = sum_encode(vec, feature_list[ind])

        doc_len = 0

        for (word, freq) in vec:
            doc_len += freq
            if word in word_dict:
                if commit == BIC:
                    word_dict[word]['BIC'] = freq

                word_dict[word]['num_doc'] += 1
                word_dict[word]['all'] += freq
             
        if commit == BIC:
            print('BIC document length) ', doc_len)
        
        sum_len += doc_len
    
    print('Mean document length) ', sum_len / (num_other + 1))
        
    word_dict = dict(filter(lambda x: x[1]['num_doc'] > 0 and x[1]['num_doc'] * 2 < num_other + 1, word_dict.items()))
    #word_dict = dict(filter(lambda x: x[1]['num_doc'] > 0, word_dict.items()))
    word_dict = {word : data | {'all' : data['all'] / data['num_doc']} for word, data in word_dict.items()}
    
    for word, sub_dict in sorted(word_dict.items(), key=lambda x: x[1]['freq'], reverse=True):
        print(f'{vocab[word]}) {sub_dict}')

        

# Load the feature of BIC (Commit message + Related files)
def load_BIC_feature(pid, vid, stage2, setting):
    # Checkout Defects4J project
    p = subprocess.Popen(f'sh /root/workspace/lib/checkout.sh {pid} {vid}', \
        shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out_txt, err_txt = p.communicate()

    if p.returncode != 0:
        log('[ERROR] Checkout failed', out_txt, err_txt)
        return
    
    # Change working directory
    try:
        os.chdir(f'/tmp/{pid}-{vid}b/')
    except:
        log('[ERROR] Moving directory failed')
        return

    GT = load_BIC_GT("/root/workspace/data/Defects4J/BIC_dataset")
    BIC = GT.set_index(["pid", "vid"]).loc[(pid, vid), "commit"]

    print('BIC)', BIC)

    base_data_dir = os.path.join(BASE_DATA_DIR, f'{pid}-{vid}b', 'commits')
    for filename in os.listdir(base_data_dir):
        if filename.startswith(f'c_{BIC}'):
            with open(os.path.join(base_data_dir, filename), "r") as file:
                data = json.load(file)
            print('Commit message) ', data['log'])
            break

    # Load data
    with open(os.path.join(f'/root/workspace/data/Defects4J/diff/{pid}-{vid}b', 'stage2.pkl'), 'rb') as file:
        stage2_dict = pickle.load(file)
    
    print(stage2_dict[stage2][setting][BIC])
    """for src_path, sub_dict in stage2_dict[stage2][setting][BIC]['addition'].items():
        print('Addition', src_path)
        #if len(sub_dict) == 0:
        #    continue

        p = subprocess.Popen(f'git show {BIC}:{src_path}', shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        code_txt, err_txt = p.communicate()

        # Error raised while copying file
        if p.returncode != 0:
            # Actual addition occured but failed to copy file
            print(f'[ERROR] Failed to copy file {BIC}:{src_path}')
        
        else:
            code_txt = code_txt.decode(encoding='utf-8', errors='ignore')
            savepath = os.path.join('/root/workspace/analyze/tmp', 'Addition_' + src_path.replace('/', '_'))

            with open(savepath, 'w') as file:
                file.write(code_txt)
    
    for src_path, sub_dict in stage2_dict[stage2][setting][BIC]['deletion'].items():
        print('Deletion', src_path)
        p = subprocess.Popen(f'git show {BIC}~1:{src_path}', shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        code_txt, err_txt = p.communicate()

        # Error raised while copying file
        if p.returncode != 0:
            # Actual addition occured but failed to copy file
            print(f'[ERROR] Failed to copy file {BIC}~1:{src_path}')
        
        else:
            code_txt = code_txt.decode(encoding='utf-8', errors='ignore')
            savepath = os.path.join('/root/workspace/analyze/tmp', 'Deletion_' + src_path.replace('/', '_'))

            with open(savepath, 'w') as file:
                file.write(code_txt)"""



def diff_interval(pid, vid, stage2, diff_type):
    excluded = get_style_change_data(os.path.join(CORE_DATA_DIR, f'{pid}-{vid}b/'), 'git', stage2)
    GT = load_BIC_GT("/root/workspace/data/Defects4J/BIC_dataset")
    BIC = GT.set_index(["pid", "vid"]).loc[(pid, vid), "commit"]

    diff_data_dir = os.path.join(DIFF_DATA_DIR, f'{pid}-{vid}b')

    if diff_type == 'git':
        with open(os.path.join(diff_data_dir, 'git_diff.pkl'), 'rb') as file:
            diff_dict = pickle.load(file)[BIC]
        
    for (before_src_path, after_src_path), sub_dict in diff_dict.items():
        if (before_src_path, after_src_path) in excluded:
            continue
        
        print('Before src path) ', before_src_path)
        print('After src path) ', after_src_path)

        for adddel, line_dict in sub_dict.items():
            print(adddel, ')')
            print(', '.join(map(str, line_dict.keys())))


# res_dict[stage2][(new_diff_type, use_stopword, adddel, use_br)]
if __name__ == "__main__":
    # Bug2Commit
    #org_setting = frozenset({'stage2':'precise', 'use_stopword':True, 'use_br':False, 'diff_type':'no_diff', 'adddel':'all-uni'}.items())
    #new_setting = frozenset({'stage2':'precise', 'use_stopword':True, 'use_br':False, 'diff_type':'file', 'adddel':'all-uni'}.items())

    #compare_settings(org_method='bug2commit', new_method='bug2commit', org_setting=org_setting, new_setting=new_setting)

    # Check token frequency of setting
    #setting = frozenset({'use_stopword':True, 'diff_type':'git', 'adddel':'all-uni'}.items())
    #check_freq(pid='Closure', vid='115', stage2='precise', setting=setting)

    #setting = frozenset({'use_stopword':True, 'diff_type':'gumtree_class', 'adddel':'all-uni'}.items())
    #check_freq(pid='Closure', vid='115', stage2='precise', setting=setting)

    # Load the feature of BIC (Commit message + Related files)
    #setting = frozenset({'diff_type':'git'}.items())
    #load_BIC_feature(pid='Closure', vid='2', stage2='precise', setting=setting)

    #setting = frozenset({'diff_type':'gumtree_class'}.items())
    #load_BIC_feature(pid='Closure', vid='2', stage2='precise', setting=setting)

    check_filtering()

    #diff_interval(pid='Closure', vid='115', stage2='precise', diff_type='git')

    # Fonte
    #org_setting = frozenset({'stage2':'precise'}.items())
    #org_setting = frozenset({'stage2':'precise', 'use_stopword':True, 'use_br':False, 'diff_type':'no_diff', 'adddel':'del', 'beta':1.5}.items())
    #new_setting = frozenset({'stage2':'precise', 'use_stopword':True, 'use_br':False, 'diff_type':'gumtree_class', 'adddel':'all-uni', 'beta':1.0}.items())

    #compare_settings(org_method='fonte', new_method='ensemble', org_setting=org_setting, new_setting=new_setting)

    # Load related files
    #load_diff(pid='Closure', vid='121', stage2='precise', setting=frozenset({'diff_type':'file'}.items()))

    #get_metric_dict(method='bug2commit', mode='project')

    #check_filtering()

    # Print setting, metric pairs (Bug2Commit)
    """for setting, metric in get_metric_dict(method='bug2commit', mode='all').items():
        setting_dict = dict(setting)

        if setting_dict['use_br'] == False and setting_dict['diff_type'] == 'file' and setting_dict['stage2'] == 'precise':
            print(setting, metric)"""