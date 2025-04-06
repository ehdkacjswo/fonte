import argparse, os, copy, itertools, pickle, sys, json, subprocess, html
import pandas as pd
from scipy.stats import wilcoxon, kruskal
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

sys.path.append('/root/workspace/analysis/lib/')
from util import *
#from stage2 import get_style_change_data

RESULT_DATA_DIR = '/root/workspace/data/Defects4J/result'
CORE_DATA_DIR = '/root/workspace/data/Defects4J/core'
BASE_DATA_DIR = '/root/workspace/data/Defects4J/baseline'
DIFF_DATA_DIR = '/root/workspace/data/Defects4J/diff'

#def cmp_id_class(stage2)

# Check the number of each identifiers
# Doesn't work for all_sep (all_sep voting is not working correctly too)
# Bug, Commit > {feature type : {project : {id / non_id : token frequency} } }
# feature_setting={'tracker': 'git', 'diff_tool' : 'base', 'diff_type' : 'greedy_id', 'adddel' : 'add'}
# feature_setting={'tracker': 'git', 'diff_tool' : 'gumtree', 'diff_type' : 'gumtree_id', 'adddel' : 'add'}
def id_dist(stage2='precise', \
    feature_setting={'tracker': 'git', 'diff_tool' : 'base', 'diff_type' : 'greedy_id', 'adddel' : 'add'}):
    
    all_GT = load_BIC_GT("/root/workspace/data/Defects4J/BIC_dataset")
    GT = all_GT[all_GT['provenance'].str.contains("Manual", na=False)]
    
    bug_dict, commit_dict = dict(), dict()
    
    for _, row in tqdm(GT.iterrows()):
        pid, vid, BIC = row.pid, row.vid, row.commit

        # Load feature, encoder and bug feature
        feature_dict, encoder_dict, bug_feature_dict = load_feature_data(pid, vid)
        
        encoder_setting = feature_setting.copy()
        del encoder_setting['adddel']

        feature_dict = feature_dict[stage2][frozenset(feature_setting.items())]
        encoder = encoder_dict[stage2][frozenset(encoder_setting.items())]

        # Evaluate the ratio of common identifiers
        # {bug_type : {commit_type : {id_type : {BIC, total}}}}
        token_ratio_dict = dict()
        
        for bug_type, bug_feature in bug_feature_dict.items():
            token_ratio_dict[bug_type] = dict()
            
            # Encode bug feature
            bug_feature_id, bug_feature_non_id = encoder.encode(bug_feature, update_vocab=False, mode='code' if bug_type == 'test_code' else 'text')

            for commit, commit_feature_dict in feature_dict.items():
                for commit_type, commit_feature_vec in commit_feature_dict.items():
                    
                    # Initialize token ratio dictionaries
                    token_ratio_dict[bug_type].setdefault(commit_type, {'id' : {'total' : list()}, 'non_id' : {'total' : list()}, 'all' : {'total' : list()}})
                    
                    # Count total, common identifiers
                    for id_type in ['id', 'non_id', 'all']:
                        total_token_cnt, common_token_cnt = 0, 0

                        if id_type == 'id':
                            commit_vec, bug_vec = commit_feature_vec['id'], bug_feature_id
                        elif id_type == 'non_id':
                            commit_vec, bug_vec = commit_feature_vec['non_id'], bug_feature_non_id
                        elif id_type == 'all':
                            commit_vec, bug_vec = commit_feature_vec['id'] + commit_feature_vec['non_id'], bug_feature_id + bug_feature_non_id

                        for word, freq in commit_vec.items():
                            total_token_cnt += freq
                            if word in bug_vec:
                                common_token_cnt += freq
                    
                        # Ratio of common identifiers
                        token_ratio = common_token_cnt / total_token_cnt if total_token_cnt > 0 else 0

                        token_ratio_dict[bug_type][commit_type][id_type]['total'].append(token_ratio)
                        if commit == BIC:
                            token_ratio_dict[bug_type][commit_type][id_type]['BIC'] = token_ratio

# Check the number of each identifiers
# Doesn't work for all_sep (all_sep voting is not working correctly too)
# {bug feature : {commit feature : {vote, rank, } } }
# feature_setting={'tracker': 'git', 'diff_tool' : 'base', 'diff_type' : 'greedy_id', 'adddel' : 'add'}
# feature_setting={'tracker': 'git', 'diff_tool' : 'gumtree', 'diff_type' : 'gumtree_id', 'adddel' : 'add'}
def id_dist(classify_id=True, stage2='precise', \
    feature_setting={'tracker': 'git', 'diff_tool' : 'base', 'diff_type' : 'greedy_id', 'adddel' : 'add'}):
    classify_id &= (feature_setting.get('diff_type', None) == 'gumtree_id')
    
    all_GT = load_BIC_GT("/root/workspace/data/Defects4J/BIC_dataset")
    GT = all_GT[all_GT['provenance'].str.contains("Manual", na=False)]
    
    # {bug_type : {commit_type : {id_type : {rank, BIC_ratio_list, total_ratio_list}}}}
    res_dict = dict()
    
    for _, row in tqdm(GT.iterrows()):
        pid, vid, BIC = row.pid, row.vid, row.commit

        # Load feature, encoder and bug feature
        feature_dict, encoder_dict, bug_feature_dict = load_feature_data(pid, vid)
        
        encoder_setting = feature_setting.copy()
        del encoder_setting['adddel']

        feature_dict = feature_dict[stage2][frozenset(feature_setting.items())]
        encoder = encoder_dict[stage2][frozenset(encoder_setting.items())]

        # Evaluate the ratio of common identifiers
        # {bug_type : {commit_type : {id_type : {BIC, total}}}}
        token_ratio_dict = dict()
        
        for bug_type, bug_feature in bug_feature_dict.items():
            token_ratio_dict[bug_type] = dict()
            
            # Encode bug feature
            bug_feature_id, bug_feature_non_id = encoder.encode(bug_feature, update_vocab=False, mode='code' if bug_type == 'test_code' else 'text')

            for commit, commit_feature_dict in feature_dict.items():
                for commit_type, commit_feature_vec in commit_feature_dict.items():
                    
                    # Initialize token ratio dictionaries
                    token_ratio_dict[bug_type].setdefault(commit_type, {'id' : {'total' : list()}, 'non_id' : {'total' : list()}, 'all' : {'total' : list()}})
                    
                    # Count total, common identifiers
                    for id_type in ['id', 'non_id', 'all']:
                        total_token_cnt, common_token_cnt = 0, 0

                        if id_type == 'id':
                            commit_vec, bug_vec = commit_feature_vec['id'], bug_feature_id
                        elif id_type == 'non_id':
                            commit_vec, bug_vec = commit_feature_vec['non_id'], bug_feature_non_id
                        elif id_type == 'all':
                            commit_vec, bug_vec = commit_feature_vec['id'] + commit_feature_vec['non_id'], bug_feature_id + bug_feature_non_id

                        for word, freq in commit_vec.items():
                            total_token_cnt += freq
                            if word in bug_vec:
                                common_token_cnt += freq
                    
                        # Ratio of common identifiers
                        token_ratio = common_token_cnt / total_token_cnt if total_token_cnt > 0 else 0

                        token_ratio_dict[bug_type][commit_type][id_type]['total'].append(token_ratio)
                        if commit == BIC:
                            token_ratio_dict[bug_type][commit_type][id_type]['BIC'] = token_ratio
        
        # Aggregate data
        for bug_type, commit_type_dict in token_ratio_dict.items():
            res_dict.setdefault(bug_type, dict())

            for commit_type, id_type_dict in commit_type_dict.items():
                res_dict[bug_type].setdefault(commit_type, \
                    {'id' : {'rank' : list(), 'BIC_ratio' : list(), 'total_ratio' : list()}, \
                    'non_id' : {'rank' : list(), 'BIC_ratio' : list(), 'total_ratio' : list()}, \
                    'all' : {'rank' : list(), 'BIC_ratio' : list(), 'total_ratio' : list()}})
                
                for id_type, sub_ratio_dict in id_type_dict.items():
                    total_token_ratio, BIC_token_ratio = sub_ratio_dict['total'], sub_ratio_dict['BIC']
                    total_token_ratio.sort(reverse=True)

                    # Rank of common identifier ratio
                    BIC_ratio_rank = len(total_token_ratio) - total_token_ratio[::-1].index(BIC_token_ratio)
                    res_dict[bug_type][commit_type][id_type]['rank'].append(BIC_ratio_rank)

                    # Common identifier ratio
                    res_dict[bug_type][commit_type][id_type]['BIC_ratio'].append(BIC_token_ratio)
                    res_dict[bug_type][commit_type][id_type]['total_ratio'].append(sum(total_token_ratio) / len(total_token_ratio))
    
    # Analyze and print results
    for bug_type, commit_type_dict in res_dict.items():
        print(f'Bug) {bug_type}')

        # Compare between identifiers
        if classify_id:
            classify_id_dict = {'rank' : dict(), 'ratio' : dict()}

        for commit_type, id_type_dict in commit_type_dict.items():
            print(f'Commit) {commit_type}\n')
            rank_list_dict, total_ratio_list_dict, BIC_ratio_list_dict = dict(), dict(), dict()

            for id_type, sub_dict in id_type_dict.items():
                print(f'ID type) {id_type}')

                BIC_rank_list = res_dict[bug_type][commit_type][id_type]['rank']
                BIC_ratio_list = res_dict[bug_type][commit_type][id_type]['BIC_ratio']
                total_ratio_list = res_dict[bug_type][commit_type][id_type]['total_ratio']

                # Evaluate rank metrics
                MRR = sum(1 / rank for rank in BIC_rank_list) / len(BIC_rank_list)
                print(f'MRR) {MRR:.3f}')

                for n in [1, 2, 3, 5, 10]:
                    acc_n = sum(1 for rank in BIC_rank_list if rank <= n)
                    print(f'acc@{n}) {acc_n}')

                # Handle ratio data
                mean_BIC_ratio = sum(BIC_ratio_list) / len(BIC_ratio_list)
                mean_total_ratio = sum(total_ratio_list) / len(total_ratio_list)
                ratio = sum(BIC_ratio / total_ratio if total_ratio > 0 else 1 \
                    for (BIC_ratio, total_ratio) in zip(BIC_ratio_list, total_ratio_list)) / len(BIC_ratio_list)
                
                print(f'BIC ratio) {mean_BIC_ratio:.3f}, Total ratio) {mean_total_ratio:.3f}')
                print(f'BIC/total Ratio) {ratio:.3f}')
                
                # Wilcoxon signed rank test
                try:
                    _, better_p = wilcoxon(BIC_ratio_list, total_ratio_list, alternative='greater')
                    _, worse_p = wilcoxon(BIC_ratio_list, total_ratio_list, alternative='less')
                    print(f'Ratio P-value) Better : {better_p:.3f}, Worse : {worse_p:.3f}\n')
                except:
                    print('Identical ratio\n')
                
                # Save data for extra analysis
                rank_list_dict[id_type] = BIC_rank_list.copy()
                BIC_ratio_list_dict[id_type] = BIC_ratio_list.copy()
                total_ratio_list_dict[id_type] = [BIC_ratio / total_ratio if total_ratio > 0 else 0 for (BIC_ratio, total_ratio) in zip(BIC_ratio_list, total_ratio_list)]
                
                if classify_id and id_type == 'id' and commit_type in ['class', 'method', 'variable']: # Not considering all_sep
                    classify_id_dict['rank'][commit_type] = BIC_rank_list.copy()
                    classify_id_dict['ratio'][commit_type] = \
                        [BIC_ratio / total_ratio if total_ratio > 0 else 0 for (BIC_ratio, total_ratio) in zip(BIC_ratio_list, total_ratio_list)]
            
            # Compare rank/ratio of ID/non_ID tokens
            print('[INFO] Non_ID/All comparison')

            try:
                _, better_p = wilcoxon(rank_list_dict['all'], rank_list_dict['non_id'], alternative='less')
                _, worse_p = wilcoxon(rank_list_dict['all'], rank_list_dict['non_id'], alternative='greater')
                print(f'Rank P-value) Better : {better_p:.3f}, Worse : {worse_p:.3f}')
            except:
                print('Identical rank')
            
            try:
                _, better_p = wilcoxon(BIC_ratio_list_dict['all'], BIC_ratio_list_dict['non_id'], alternative='greater')
                _, worse_p = wilcoxon(BIC_ratio_list_dict['all'], BIC_ratio_list_dict['non_id'], alternative='less')
                print(f'BIC ratio P-value) Better : {better_p:.3f}, Worse : {worse_p:.3f}')
            except:
                print('Identical BIC ratio')

            try:
                _, better_p = wilcoxon(total_ratio_list_dict['all'], total_ratio_list_dict['non_id'], alternative='greater')
                _, worse_p = wilcoxon(total_ratio_list_dict['all'], total_ratio_list_dict['non_id'], alternative='less')
                print(f'Ratio P-value) Better : {better_p:.3f}, Worse : {worse_p:.3f}\n')
            except:
                print('Identical ratio\n')
    
        # Compare the different types of identifiers
        # Not considering all_sep
        if classify_id:
            print('[INFO] ID type comparison')

            for metric, class_dict in classify_id_dict.items():
                
                # Perfom Kruskal-Willis test
                try:
                    _, kruskal_p = kruskal(class_dict['class'], class_dict['method'], class_dict['variable'])
                    print(f'[{metric}] Kruskal P-value) {kruskal_p}')
                except:
                    print(f'Identical {metric}?')

                id_class_list = ['class', 'method', 'variable']

                for ind1, id_class1 in enumerate(id_class_list):
                    for ind2 in range(ind1 + 1, 3):
                        id_class2 = id_class_list[ind2]
                        try:
                            _, better_p = wilcoxon(class_dict[id_class1], class_dict[id_class2], alternative='less' if metric == 'rank' else 'greater')
                            _, worse_p = wilcoxon(class_dict[id_class1], class_dict[id_class2], alternative='greater' if metric == 'rank' else 'less')
                            print(f'[{metric}]{id_class1}-{id_class2} P-value) Better : {better_p:.3f}, Worse : {worse_p:.3f}')
                        except:
                            print(f'[{metric}] {id_class1}-{id_class2} Identical')


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
def build_html(pid='Lang', vid='49', stage2='precise', \
    intvl_setting=frozenset({'tracker': 'git', 'diff_tool' : 'gumtree', 'diff_type' : 'base'}.items())):

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
    #if not dict(intvl_setting)['diff_type'].endswith('id'):
    #    print('This function is designed to higlight components')
    #    return

    html_lines = ["<html><head><title>Highlighted Intervals</title></head><body>"]

    GT = load_BIC_GT("/root/workspace/data/Defects4J/BIC_dataset")
    BIC = GT.set_index(["pid", "vid"]).loc[(pid, vid), "commit"]

    org_setting = frozenset((dict(intvl_setting) | {'diff_type' : 'base', 'diff_tool' : 'base'}).items())

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
        'id' : 'red', 'non_id' : 'blue', 'diff' : 'red'}

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
def print_metric(method, stage2, bisect_setting):
    metric_dict = get_metric_dict(method=method, mode='all')
    bisect_setting['stage2'] = stage2

    sub_metric_dict = metric_dict[frozenset(bisect_setting.items())]
    for metric_key, metric in sub_metric_dict.items():
        print(f'{metric_key}) {metric}')

    if method == 'bug2commit':
        vote_setting = bisect_setting.copy()
        del vote_setting['beta']

        sub_metric_dict = metric_dict[frozenset(vote_setting.items())]
        for metric_key, metric in sub_metric_dict.items():
            print(f'{metric_key}) {metric}')

    

# Print metrics in order (best > worst) for given method
# Fix the setting for given 'fix' dictionary
# fix={'stage2' : 'precise', 'use_br' : False, 'diff_type' : 'gumtree_id', 'use_id' : True, 'diff_tool' : 'gumtree'}
def metric_by_setting(method='bug2commit', fix={'stage2' : 'precise', 'use_br' : False, 'diff_type' : 'gumtree_id', 'use_id' : False, 'diff_tool' : 'gumtree'}):
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
    build_html()
    #id_dist()
    
    #metric_by_setting()

    # No diff, File, Git, GumTreee
    #print_metric(method='bug2commit', stage2='precise', bisect_setting={'tracker': 'git', 'diff_tool': None, 'adddel': 'all_uni', 'use_br' : False, 'beta': 0.1})
    #print_metric(method='bug2commit', stage2='precise', bisect_setting={'tracker': 'git', 'diff_tool': 'file', 'diff_type': 'base', 'adddel': 'all_uni', 'use_br' : False, 'beta': 0.1})
    #print_metric(method='bug2commit', stage2='precise', bisect_setting={'tracker': 'git', 'diff_tool': 'base', 'diff_type': 'base', 'adddel': 'all_uni', 'use_br' : False, 'beta': 0.1})
    #print_metric(method='bug2commit', stage2='precise', bisect_setting={'tracker': 'git', 'diff_tool': 'gumtree', 'diff_type': 'base', 'adddel': 'all_uni', 'use_br' : False, 'beta': 0.1})

    # Greedy
    #print_metric(method='bug2commit', stage2='precise', bisect_setting={'beta': 0.1, 'use_id': False, 'diff_tool': 'base', 'use_br' : False, 'adddel': 'add', 'diff_type' : 'greedy_id', 'tracker' : 'git'})
    #print_metric(method='bug2commit', stage2='precise', bisect_setting={'beta': 0.1, 'use_id': False, 'diff_tool': 'base', 'use_br' : False, 'adddel': 'all_uni', 'diff_type' : 'greedy_id', 'tracker' : 'git'})
    
    # GumTree
    #print_metric(method='bug2commit', stage2='precise', bisect_setting={'beta': 0.1, 'use_id': False, 'diff_tool': 'gumtree', 'use_br' : False, 'adddel': 'add', 'diff_type' : 'gumtree_id', 'tracker' : 'git', 'classify_id' : True})
    #print_metric(method='bug2commit', stage2='precise', bisect_setting={'beta': 0.1, 'use_id': False, 'diff_tool': 'gumtree', 'use_br' : False, 'adddel': 'all_uni', 'diff_type' : 'gumtree_id', 'tracker' : 'git', 'classify_id' : True})

    #print_metric(method='bug2commit', stage2='precise', bisect_setting={'beta': 0.1, 'use_id': True, 'diff_tool': 'gumtree', 'use_br' : False, 'adddel': 'add', 'diff_type' : 'gumtree_id', 'tracker' : 'git', 'classify_id' : False})
    #print_metric(method='bug2commit', stage2='precise', bisect_setting={'beta': 0.1, 'use_id': True, 'diff_tool': 'gumtree', 'use_br' : False, 'adddel': 'all_uni', 'diff_type' : 'gumtree_id', 'tracker' : 'git', 'classify_id' : False})