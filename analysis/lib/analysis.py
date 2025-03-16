import argparse, os, copy, itertools, pickle, sys, json, subprocess, html
import pandas as pd
from scipy.stats import wilcoxon
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import math
from interval import interval, inf

sys.path.append('/root/workspace/lib/')
from experiment_utils import load_BIC_GT

sys.path.append('/root/workspace/analysis/lib')
from result_gen import get_metric_dict

sys.path.append('/root/workspace/data_collector/lib/')
from encoder import *
from utils import CustomInterval, get_src_from_commit

sys.path.append('/root/workspace/analysis/lib/')
from util import *
#from stage2 import get_style_change_data

RESULT_DATA_DIR = '/root/workspace/data/Defects4J/result'
CORE_DATA_DIR = '/root/workspace/data/Defects4J/core'
BASE_DATA_DIR = '/root/workspace/data/Defects4J/baseline'
DIFF_DATA_DIR = '/root/workspace/data/Defects4J/diff'

def cmp_id_class(stage2)

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
        
        if pid == 'Jsoup' and vid == '17':
            break

# Print how using IDs affect the voting for feature pairs + 
# Vote setting doesn't have to contain 'use_id'
def cmp_use_id(pid='Cli', vid='10', stage2='precise', \
    feature_setting={'tracker': 'git', 'diff_tool' : 'base', 'diff_type' : 'gumtree_id', 'adddel' : 'add'}, \
    vote_setting={'use_br' : True, 'classify_id' : True}):
    
    GT = load_BIC_GT("/root/workspace/data/Defects4J/BIC_dataset")
    BIC = GT.set_index(["pid", "vid"]).loc[(pid, vid), "commit"]

    # Load data (Feature, encoder, bug_feature)
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

        if pid == 'Jsoup' and vid == '17':
            break
        #break

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

    #print(bug_feature_dict)


# Check how id filtering affects the document
# Setting fields = (tracker, diff_tool, doc_level, adddel)
def id_filter_print(pid, vid, stage2='precise', \
    setting=frozenset({'tracker': 'git', 'diff_tool' : 'base', 'adddel' : 'all_sep'}.items())):
    GT = load_BIC_GT("/root/workspace/data/Defects4J/BIC_dataset")
    res_dict = dict()

    org_setting = frozenset((dict(setting) | {'diff_type' : 'base'}).items())
    new_setting = frozenset((dict(setting) | {'diff_type' : 'id_all'}).items())
    adddel = dict(setting)['adddel']
    target_type_list = [('add_diff', 'add_id'), ('del_diff', 'del_id')] if adddel == 'all_sep' else [('diff', 'id')]

    BIC = GT.set_index(["pid", "vid"]).loc[(pid, vid), "commit"]

    # Load data
    with open(os.path.join(f'/root/workspace/data/Defects4J/diff/{pid}-{vid}b', 'feature.pkl'), 'rb') as file:
        feature_dict = pickle.load(file)
    
    with open(os.path.join(f'/root/workspace/data/Defects4J/diff/{pid}-{vid}b', 'vocab.pkl'), 'rb') as file:
        vocab = pickle.load(file)
    
    vocab = {ind : word for word, ind in vocab.items()}
    org_dict = feature_dict[stage2][org_setting]
    new_dict = feature_dict[stage2][new_setting]

    for commit in org_dict.keys():
        print(f'Commit) {commit}')

        org_feature, new_feature = org_dict[commit], new_dict[commit]

        for (org_type, new_type) in target_type_list:
            aaa = {ind : freq for (ind, freq) in org_feature[org_type]}
            bbb = {ind : freq for (ind, freq) in new_feature[new_type]}

            for ind, freq in aaa.items():
                if bbb.get(ind, 0) < freq:
                    print(f'Deleted token) {vocab[ind]}')
                
            for ind, freq in bbb.items():
                if ind not in aaa:
                    print(f'New token) {vocab[ind]}')

# Code txt could be possibly "None" (Failed to get code data)
def get_tokens_intvl(code_txt, intvl):
    
    # Convert index from interval to integer
    def convert_ind(ind):
        if ind == -inf:
            return 0
        if ind == inf:
            return len(code_txt)
        
        return math.floor(ind) + 1
        
    return code_txt[convert_ind(intvl[0]) : convert_ind(intvl[1])]

# Build html file that highlights components
# Interval setting fields = (tracker, diff_tool, diff_type)
def build_html(pid='Cli', vid='10', stage2='precise', \
    intvl_setting=frozenset({'tracker': 'git', 'diff_tool' : 'base', 'diff_type' : 'greedy_id'}.items())):

    # Convert index from interval to integer
    def convert_ind(ind, code_txt):
        if ind == -inf:
            return 0
        if ind == inf:
            return len(code_txt)
        
        return math.floor(ind) + 1
    
    # Escape html special character
    def html_escape(txt):
        escape_txt = html.escape(txt)
        return len(escape_txt) - len(txt), escape_txt
    
    # GumTree diff is not suitable to visualize since it's not line level based
    """if dict(intvl_setting)['diff_tool'] == 'gumtree':
        print('')
        return"""
    
    # 
    if not dict(intvl_setting)['diff_type'].endswith('id'):
        print('This function is designed to higlight components')
        return

    html_lines = ["<html><head><title>Highlighted Intervals</title></head><body>"]

    GT = load_BIC_GT("/root/workspace/data/Defects4J/BIC_dataset")
    BIC = GT.set_index(["pid", "vid"]).loc[(pid, vid), "commit"]

    org_setting = frozenset((dict(intvl_setting) | {'diff_type' : 'base'}).items())

    # Checkout Defects4J project
    p = subprocess.Popen(['sh', '/root/workspace/lib/checkout.sh', pid, vid], \
        stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out_txt, err_txt = p.communicate()

    if p.returncode != 0:
        return
    
    # Change working directory to target Defects4J project
    try:
        os.chdir(f'/tmp/{pid}-{vid}b/')
    except:
        return

    # Load data
    with open(os.path.join(f'/root/workspace/data/Defects4J/diff/{pid}-{vid}b', 'total_intvl.pkl'), 'rb') as file:
        intvl_dict = pickle.load(file)
    
    org_intvl_dict = intvl_dict[stage2][org_setting]
    new_intvl_dict = intvl_dict[stage2][intvl_setting]

    # Colors of components to be highlighted
    color_dict = {'class' : 'red', 'method' : 'green', 'variable' : 'purple', 'comment' : 'blue', \
        'id' : 'red', 'non_id' : 'blue'}

    for commit, org_commit_dict in org_intvl_dict.items():
        html_lines.append(f"<h2>Commit: {commit}{'(BIC)' if commit == BIC else ''}</h2>")

        for adddel, org_adddel_dict in org_commit_dict.items():
            html_lines.append(f"<h3>{adddel}:</h3><ul>")

            for src_path, org_src_dict in org_adddel_dict.items():
                org_intvl = org_src_dict['diff']
                
                # Skip when line diff is empty
                if org_intvl.is_empty():
                    continue

                html_lines.append(f"<li><strong>{src_path}:</strong><br>")
                code_txt = get_src_from_commit(commit + ('' if adddel == 'addition' else '~1'), src_path)

                # Token intervals 
                for sub_org_intvl in org_intvl:
                    org_start, org_end = convert_ind(sub_org_intvl[0], code_txt), convert_ind(sub_org_intvl[1], code_txt)
                    diff = ''.join(code_txt[org_start : org_end])
                    id_list = list()

                    # Intervals of components
                    for id_type, id_intvl in new_intvl_dict[commit][adddel][src_path].items():
                        new_id_intvl = CustomInterval(org_start, org_end - 1) & id_intvl

                        for sub_id_intvl in new_id_intvl:
                            id_start, id_end = convert_ind(sub_id_intvl[0], code_txt) - org_start, convert_ind(sub_id_intvl[1], code_txt) - org_start
                            id_list.append((id_start, id_end, id_type))
                    
                    id_list.sort(key=lambda x : x[0])
                    offset, last_ind = 0, 0

                    # Highlight the components
                    for (id_start, id_end, id_type) in id_list:
                        start_off, start_txt = html_escape("".join(diff[last_ind : offset + id_start]))
                        mid_off, mid_txt = html_escape("".join(diff[offset + id_start : offset + id_end]))

                        diff = "".join(diff[:last_ind]) + \
                            start_txt + \
                            f'<span style="color: {color_dict[id_type]}; font-weight: bold;">{mid_txt}</span>' + \
                            "".join(diff[offset + id_end:])
                        
                        last_ind += len(f'<span style="color: {color_dict[id_type]}; font-weight: bold;">') + len('</span>') + len(start_txt) + len(mid_txt)
                        offset += len(f'<span style="color: {color_dict[id_type]}; font-weight: bold;">') + len('</span>') + start_off + mid_off

                    html_lines.append(f"<p>{diff}</p></li>")
                html_lines.append(f"</li>")
            html_lines.append("</ul>")

    html_lines.append("</body></html>")
    with open('/root/workspace/analysis/ereer.html', 'w') as file:
        file.write('\n'.join(html_lines))

# 
def print_vote(pid, vid, stage2, setting):
    GT = load_BIC_GT("/root/workspace/data/Defects4J/BIC_dataset")
    BIC = GT.set_index(["pid", "vid"]).loc[(pid, vid), "commit"]

    with open(os.path.join(f'/root/workspace/data/Defects4J/result/{pid}-{vid}b', 'vote', 'bug2commit.pkl'), 'rb') as file:
        vote_dict = pickle.load(file)
    
    for vote_type, vote_df in vote_dict[stage2][setting].items():
        print(vote_type)
        print(vote_df.loc[BIC])

# Compare the settings
def compare_settings(org_method, new_method, org_setting, new_setting):
    org_metric_dict = get_metric_dict(method=org_method, mode='project')
    #org_metric = metric_converter(org_metric_dict[org_setting])

    new_metric_dict = get_metric_dict(method=new_method, mode='project')
    #new_metric = metric_converter(new_metric_dict[new_setting])

    # Bug2Commit의 경우, num_iter에만 beta가 있다

    for metric_key in ['rank', 'num_iter']:
        print(f'Metric) {metric_key}')

        org_metric_setting, new_metric_setting = org_setting, new_setting

        # Bug2Commit uses extra setting (beta) for num_iter
        if metric_key == 'rank':
            if org_method == 'bug2commit':
                org_metric_setting = dict(org_metric_setting)
                org_metric_setting.pop('beta', None)
                org_metric_setting = frozenset(org_metric_setting.items())
            
            if new_method == 'bug2commit':
                new_metric_setting = dict(new_metric_setting)
                new_metric_setting.pop('beta', None)
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

# Print metrics in order (best > worst) for given method
# Fix the setting for given 'fix' dictionary
def metric_by_setting(method='bug2commit', fix={'stage2' : 'precise', 'use_br' : False}):
    all_GT = load_BIC_GT("/root/workspace/data/Defects4J/BIC_dataset")
    GT = all_GT[all_GT['provenance'].str.contains("Manual", na=False)]
    metric_dict = get_metric_dict(method=method, mode='all')

    res_list = list()
    for setting, setting_dict in metric_dict.items():
        
        # Consider settings with target fixed settings
        is_target = True
        for key, value in fix.items():
            if not (key in dict(setting) and dict(setting)[key] == value):
                is_target = False
                break
        
        if not is_target:
            continue
        
        res_list.append((setting, setting_dict))
    
    #print(res_list)
    
    for metric in ['MRR', 'num_iter']:
        print('Target metric : ', metric)
        new_list = [data for data in res_list if metric in dict(data[1])] # Bug2Commit uses different settings for MRR and num_iter
        new_list.sort(key=lambda x: x[1][metric], reverse=(metric == 'MRR')) # Bigger MRR, Smaller num_iter is better
        
        for (setting, setting_dict) in new_list:
            print(setting, setting_dict[metric])

# res_dict[stage2][(new_diff_type, use_stopword, adddel, use_br)]
if __name__ == "__main__":
    
    # 
    #(stage2='precise', \
    #    base_setting={'tracker' : 'git', 'diff_tool' : 'base', 'diff_type' : 'greedy_id', 'use_br' : False}, \
    #    org_setting = {'adddel' : 'add', 'use_id' : False}, \
    #    new_setting = {'adddel' : 'add', 'use_id' : True})
    
    #cmp_type_pair(stage2='precise', \
    #    base_setting={'tracker' : 'git', 'diff_tool' : 'gumtree', 'diff_type' : 'gumtree_id', 'use_br' : False, 'classify_id' : True}, \
    #    org_setting = {'adddel' : 'add', 'use_id' : False}, \
    #    new_setting = {'adddel' : 'add', 'use_id' : True})

    cmp_use_id()
    #metric_by_setting()
    #org_setting = frozenset({'tracker': 'git', 'stage2' : 'precise', 'use_br' : False, 'diff_tool' : 'base', 'diff_type' : 'greedy_id', 'adddel' : 'add', 'use_id' : False, 'beta' : 0.1}.items())
    #new_setting = frozenset({'tracker': 'git', 'stage2' : 'precise', 'use_br' : False, 'diff_tool' : 'base', 'diff_type' : 'greedy_id', 'adddel' : 'add', 'use_id' : True, 'beta' : 0.1}.items())
    #compare_settings(org_method='bug2commit', new_method='bug2commit', org_setting=org_setting, new_setting=new_setting)
    #id_count()
    #abc()
    #build_html()
    #cmp_type_pair()
    #print_metric()
    #print_metric(method='ensemble', fix={'stage2' : 'precise', 'use_br' : False})
    #org_setting = frozenset({'tracker': 'git', 'stage2' : 'precise', 'use_br' : False, 'diff_tool' : 'gumtree', 'diff_type' : 'id_all', 'adddel' : 'all_uni', 'doc_level' : 'commit', 'beta' : 0.1}.items())
    #new_setting = frozenset({'tracker': 'git', 'stage2' : 'precise', 'use_br' : False, 'diff_tool' : 'gumtree', 'diff_type' : 'id', 'adddel' : 'all_uni', 'doc_level' : 'commit', 'beta' : 0.1}.items())
    #compare_settings(org_method='bug2commit', new_method='bug2commit', org_setting=org_setting, new_setting=new_setting)

    #print_vote('Csv', '7', 'precise', frozenset({'tracker': 'git', 'use_br' : False, 'diff_tool' : 'gumtree', 'diff_type' : 'id', 'adddel' : 'all_uni', 'doc_level' : 'commit'}.items()))
    #print_vote('Csv', '7', 'precise', frozenset({'tracker': 'git', 'use_br' : False, 'diff_tool' : 'gumtree', 'diff_type' : 'id_all', 'adddel' : 'all_uni', 'doc_level' : 'commit'}.items()))

    #aaa(pid='Cli', vid='37', stage2='precise', setting=frozenset({'tracker': 'git', 'diff_tool' : 'base', 'doc_level' : 'commit'}.items()))