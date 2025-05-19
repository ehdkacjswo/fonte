import argparse, os, copy, itertools, pickle, sys, json, html, subprocess
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from scipy.stats import wilcoxon, ttest_rel
from statistics import mean
from typing import Literal
import numpy
import pandas as pd
import csv
from interval import interval, inf

sys.path.append('/root/workspace/lib/')
from experiment_utils import *

sys.path.append('/root/workspace/data_collector/lib/')
from encoder import *
from utils import CustomInterval, get_src_from_commit

RESULT_DATA_DIR = '/root/workspace/data/Defects4J/result'
DIFF_DATA_DIR = '/root/workspace/data/Defects4J/diff'

# Build html file that highlights components
# Interval setting fields = (tracker, diff_tool, diff_type)
def build_html(pid='Cli', vid='2', stage2='precise', \
    intvl_setting=frozenset({'tracker': 'git', 'diff_tool' : 'gumtree', 'diff_type' : 'gumtree_id'}.items())):

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


def token_share_ratio_proj(pid, vid, stage2, feature_setting, BIC=None):
    if BIC is None:
        GT = load_BIC_GT("/root/workspace/data/Defects4J/BIC_dataset")
        BIC = GT.set_index(["pid", "vid"]).loc[(pid, vid), "commit"]

    res_dict = dict()

    # Load feature, and bug feature
    with open(os.path.join(DIFF_DATA_DIR, f'{pid}-{vid}b', 'feature.pkl'), 'rb') as file:
        feature_dict = pickle.load(file)

    with open(os.path.join(DIFF_DATA_DIR, f'{pid}-{vid}b', 'bug_feature.pkl'), 'rb') as file:
        bug_feature_dict = pickle.load(file)
    
    feature_dict = feature_dict[stage2][frozenset(feature_setting.items())]
    bug_feature_dict = bug_feature_dict[stage2][frozenset(feature_setting.items())]

    # Evaluate the ratio of common identifiers
    # {bug_type : {commit_type : {id_type : {BIC, total}}}}
    token_ratio_dict = dict()
    
    for bug_type, sub_bug_feature_dict in bug_feature_dict.items():
        token_ratio_dict[bug_type] = dict()
        bug_feature_id, bug_feature_non_id = sub_bug_feature_dict['id'], sub_bug_feature_dict['non_id']

        for commit, commit_feature_dict in feature_dict.items():
            for commit_type, commit_feature_vec in commit_feature_dict.items():

                # Initialize token ratio dictionaries
                token_ratio_dict[bug_type].setdefault(commit_type, \
                    {'id' : {'total' : list()}, 'non_id' : {'total' : list()}, 'all' : {'total' : list()}})
                
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
            res_dict[bug_type] = dict()

            for commit_type, id_type_dict in commit_type_dict.items():
                res_dict[bug_type][commit_type] = dict()
                
                for id_type, sub_ratio_dict in id_type_dict.items():
                    res_dict[bug_type][commit_type][id_type] = dict()
                    total_token_ratio, BIC_token_ratio = sub_ratio_dict['total'], sub_ratio_dict['BIC']
                    total_token_ratio.sort(reverse=True)

                    # Rank of common token ratio
                    BIC_ratio_rank = len(total_token_ratio) - total_token_ratio[::-1].index(BIC_token_ratio)
                    res_dict[bug_type][commit_type][id_type]['rank'] = BIC_ratio_rank

                    # Common token ratio
                    res_dict[bug_type][commit_type][id_type]['BIC_ratio'] = BIC_token_ratio
                    res_dict[bug_type][commit_type][id_type]['avg_ratio'] =  sum(total_token_ratio) / len(total_token_ratio)
    
    return res_dict

# Check the number of each identifiers
# Doesn't work for all_sep (all_sep voting is not working correctly too)

# {project : {bug type : {commit type : 
# {id / non_id / all : 
# rank / BIC_ratio / rel_BIC_ratio} } } }
def token_share_ratio(stage2, feature_setting):
    res_dict = dict()
    all_GT = load_BIC_GT("/root/workspace/data/Defects4J/BIC_dataset")
    GT = all_GT[all_GT['provenance'].str.contains("Manual", na=False)]
    
    for _, row in tqdm(GT.iterrows()):
        pid, vid, BIC = row.pid, row.vid, row.commit
        res_dict[f'{pid}-{vid}b'] = token_share_ratio_proj(pid=pid, vid=vid, stage2=stage2, feature_setting=feature_setting, BIC=BIC)
    
    return res_dict

def BIC_info(pid='Jsoup', vid='46'):
    GT = load_BIC_GT("/root/workspace/data/Defects4J/BIC_dataset")
    BIC = GT.set_index(["pid", "vid"]).loc[(pid, vid), "commit"]
    print(BIC)

    with open(os.path.join(DIFF_DATA_DIR, f'{pid}-{vid}b', 'total_intvl.pkl'), 'rb') as file:
        total_intvl_dict = pickle.load(file)
    
    print(total_intvl_dict['precise'][frozenset({'tracker' : 'git'}.items())][BIC])

# Get metric dictionary for bug2commit
# method : fonte, bug2commit, ensemble
# mode : all, project
# metric : mean rank, mean number of iterations
def get_metric_dict(method: Literal['fonte', 'bug2commit', 'ensemble', 'bert'], mode: Literal['all', 'project']):
    savepath = f"/root/workspace/analysis/data/{method}/metric_{mode}.pkl"

    # If file already exists, read it
    if os.path.isfile(savepath):
        with open(savepath, 'rb') as file:
            return pickle.load(file)
    
    # Load manual data only
    all_GT = load_BIC_GT("/root/workspace/data/Defects4J/BIC_dataset")
    GT = all_GT[all_GT['provenance'].str.contains("Manual", na=False)]

    res_dict = dict()

    # Iterate through projects
    for _, row in GT.iterrows():
        pid, vid, BIC = row.pid, row.vid, row.commit
        proj_dir = os.path.join(RESULT_DATA_DIR, f'{pid}-{vid}b')

        # Get rank
        with open(os.path.join(proj_dir, 'vote', f'{method}.pkl'), 'rb') as file:
            vote_dict = pickle.load(file)
        
        for stage2, value in vote_dict.items():
            if method == 'fonte': # Fonte doesn't have extra setting
                rank = value['rank'].get(BIC)
                setting_key = frozenset({'stage2' : stage2}.items())

                if mode == 'project':
                    if setting_key not in res_dict:
                        res_dict[setting_key] = dict()

                    res_dict[setting_key][f'{pid}-{vid}b'] = {'rank' : rank}

                else:
                    if setting_key not in res_dict:
                        res_dict[setting_key] = {'MRR': 0, 'acc@1': 0, 'acc@2': 0, 'acc@3': 0, 'acc@5': 0, 'acc@10': 0, 'num_iter': 0}

                    res_dict[setting_key]['MRR'] += 1 / (rank * len(GT))
                    res_dict[setting_key]['acc@1'] += 1 if rank <= 1 else 0
                    res_dict[setting_key]['acc@2'] += 1 if rank <= 2 else 0
                    res_dict[setting_key]['acc@3'] += 1 if rank <= 3 else 0
                    res_dict[setting_key]['acc@5'] += 1 if rank <= 5 else 0
                    res_dict[setting_key]['acc@10'] += 1 if rank <= 10 else 0

            else: # Bug2Commit and Ensemble have extra settings
                for setting, vote_df in value.items():
                    rank = vote_df['all']['rank'].get(BIC) if method == 'bug2commit' else vote_df['rank'].get(BIC)
                    #vote = vote_df['vote'].get(BIC)
                    
                    setting_key = frozenset((dict(setting) | {'stage2' : stage2}).items())

                    #if vote == 0:
                    #    print('0 score BIC', pid, vid, setting_key)

                    if mode == 'project':
                        res_dict.setdefault(setting_key, dict())
                        res_dict[setting_key][f'{pid}-{vid}b'] = {'rank' : rank}

                    else:
                        if setting_key not in res_dict:
                            res_dict[setting_key] = {'MRR': 0, 'acc@1': 0, 'acc@2': 0, 'acc@3': 0, 'acc@5': 0, 'acc@10': 0}

                        res_dict[setting_key]['MRR'] += 1 / (rank * len(GT))
                        res_dict[setting_key]['acc@1'] += 1 if rank <= 1 else 0
                        res_dict[setting_key]['acc@2'] += 1 if rank <= 2 else 0
                        res_dict[setting_key]['acc@3'] += 1 if rank <= 3 else 0
                        res_dict[setting_key]['acc@5'] += 1 if rank <= 5 else 0
                        res_dict[setting_key]['acc@10'] += 1 if rank <= 10 else 0
        
        # Get iteration
        with open(os.path.join(proj_dir, 'iteration', f'{method}.pkl'), 'rb') as file:
            iter_dict = pickle.load(file)
    
        for stage2, value in iter_dict.items():
            if method == 'fonte': # Fonte doesn't have extra setting
                setting_key = frozenset({'stage2' : stage2}.items())

                if mode == 'project':
                    res_dict[setting_key][f'{pid}-{vid}b']['num_iter'] = value
                else:
                    res_dict[setting_key]['num_iter'] += value / len(GT)
            
            else: #Ensemble has extra settings
                for setting, num_iter in value.items():
                    setting_key = frozenset((dict(setting) | {'stage2' : stage2}).items())
                    res_dict.setdefault(setting_key, dict())

                    if mode == 'project':
                        res_dict[setting_key].setdefault(f'{pid}-{vid}b', dict())
                        res_dict[setting_key][f'{pid}-{vid}b']['num_iter'] = num_iter
                    else:
                        res_dict[setting_key]['num_iter'] = res_dict[setting_key].get('num_iter', 0) + num_iter / len(GT)

    # Save & return the dictionary
    #os.makedirs(savepath, exist_ok=True)
    with open(savepath, 'wb') as file:
        pickle.dump(res_dict, file)

    return res_dict

# Create csv file for ART ANOVA
def metrics_to_csv(method: Literal['fonte', 'bug2commit', 'ensemble']):
    savepath = f"/root/workspace/analysis/data/{method}/metrics.csv"

    """if os.path.isfile(savepath):
        print(f'{savepath} already exists!')
        return"""
    
    GT = load_BIC_GT("/root/workspace/data/Defects4J/BIC_dataset")
    metric_dict = get_metric_dict(method, mode='project')
    
    setting_key_list = list(dict(next(iter(metric_dict))).keys())
    field = ['project'] + setting_key_list + ['DependentName', 'DependentValue']

    with open(savepath, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(field)

        for setting, proj_dict in metric_dict.items():
            setting_dict = dict(setting)
            setting_row = [setting_dict[key] for key in setting_key_list]

            for project, sub_dict in proj_dict.items():
                writer.writerow([project] + setting_row + ['rank', sub_dict['rank']])
                
                if method != 'bug2commit': # Bug2Commit doesn't have iteration data
                    writer.writerow([project] + setting_row + ['num_iter', sub_dict['num_iter']])

if __name__ == "__main__":
    build_html(pid='Csv', vid='12', stage2='precise', intvl_setting=frozenset({'tracker': 'git', 'diff_tool' : 'gumtree', 'diff_type' : 'gumtree_id'}.items()))
    #BIC_info()