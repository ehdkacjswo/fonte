import argparse, os, copy, itertools, pickle, sys, json, subprocess
import pandas as pd
from scipy.stats import wilcoxon
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import math
from interval import interval, inf

sys.path.append('/root/workspace/lib/')
from experiment_utils import load_BIC_GT

sys.path.append('/root/workspace/analyze/lib')
from result_gen import get_metric_dict

sys.path.append('/root/workspace/data_collector/lib/')
from encoder import *
from utils import CustomInterval, get_src_from_commit

sys.path.append('/root/workspace/data_collector/tool/')
#from stage2 import get_style_change_data

RESULT_DATA_DIR = '/root/workspace/data/Defects4J/result'
CORE_DATA_DIR = '/root/workspace/data/Defects4J/core'
BASE_DATA_DIR = '/root/workspace/data/Defects4J/baseline'
DIFF_DATA_DIR = '/root/workspace/data/Defects4J/diff'

# Check how id filtering affects the document
def id_filter_print(stage2='precise', setting=frozenset({'tracker': 'git', 'diff_tool' : 'base', 'doc_level' : 'commit', 'adddel' : 'all_sep'}.items())):
    GT = load_BIC_GT("/root/workspace/data/Defects4J/BIC_dataset")
    res_dict = dict()

    org_setting = frozenset((dict(setting) | {'diff_type' : 'base'}).items())
    new_setting = frozenset((dict(setting) | {'diff_type' : 'id_all'}).items())
    adddel = dict(setting)['adddel']
    target_type_list = [('add_diff', 'add_id'), ('del_diff', 'del_id')] if adddel == 'all_sep' else [('diff', 'id')]

    # Iterate through projects
    for _, row in GT.iterrows():
        #pid, vid, BIC = row.pid, row.vid, row.commit
        pid = 'Closure'
        vid = '2'
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
        
        break

# Code txt could be possibly "None" (Failed to get code data)
def get_tokens_intvl(code_txt, intvl):
    
    # Convert index from interval to integer
    def convert_ind(ind):
        if ind == -inf:
            return 0
        if ind == inf:
            return len(code_txt)
        
        return math.floor(ind) + 1
        
    return ''.join(code_txt[convert_ind(intvl[0]) : convert_ind(intvl[1])])

# Convert index from interval to integer
def convert_ind(ind, code_txt):
    if ind == -inf:
        return 0
    if ind == inf:
        return len(code_txt)
    
    return math.floor(ind) + 1

def aaa(pid='Cli', vid='10', stage2='precise', setting=frozenset({'tracker': 'git', 'diff_tool' : 'base', 'doc_level' : 'commit'}.items())):
    html_lines = ["<html><head><title>Highlighted Intervals</title></head><body>"]

    org_setting = frozenset((dict(setting) | {'diff_type' : 'base'}).items())
    new_setting = frozenset((dict(setting) | {'diff_type' : 'id'}).items())

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
    new_intvl_dict = intvl_dict[stage2][new_setting]

    color_dict = {'class' : 'red', 'method' : 'green', 'variable' : 'blue', 'comment' : 'purple'}

    for commit, org_commit_dict in org_intvl_dict.items():
        html_lines.append(f"<h2>Commit: {commit}</h2>")

        for adddel, org_adddel_dict in org_commit_dict.items():
            html_lines.append(f"<h3>{adddel}:</h3><ul>")

            for src_path, org_src_dict in org_adddel_dict.items():
                org_intvl = org_src_dict['diff']
                
                # Ignore when line diff is empty
                if org_intvl.is_empty():
                    continue

                html_lines.append(f"<li><strong>{src_path}:</strong><br>")
                code_txt = get_src_from_commit(commit + ('' if adddel == 'addition' else '~1'), src_path)

                # Token intervals 
                for sub_org_intvl in org_intvl:
                    org_start, org_end = convert_ind(sub_org_intvl[0], code_txt), convert_ind(sub_org_intvl[1], code_txt)
                    diff = ''.join(code_txt[org_start : org_end])
                    id_list = list()

                    # Intervals of identifiers
                    for id_type, id_intvl in new_intvl_dict[commit][adddel][src_path].items():
                        new_id_intvl = CustomInterval(org_start, org_end - 1) & id_intvl

                        for sub_id_intvl in new_id_intvl:
                            id_start, id_end = convert_ind(sub_id_intvl[0], code_txt) - org_start, convert_ind(sub_id_intvl[1], code_txt) - org_start
                            id_list.append((id_start, id_end, id_type))
                    
                    id_list.sort(key=lambda x : x[0])
                    offset = 0

                    for (id_start, id_end, id_type) in id_list:
                        diff = "".join(diff[:offset + id_start]) + \
                            f'<span style="color: {color_dict[id_type]}; font-weight: bold;">{"".join(diff[offset + id_start : offset + id_end])}</span>' + \
                            "".join(diff[offset + id_end:])
                        
                        offset += len(f'<span style="color: {color_dict[id_type]}; font-weight: bold;">') + len('</span>')

                    html_lines.append(f"<p>{diff}</p></li>")

                #
                html_lines.append(f"</li>")
            
            html_lines.append("</ul>")

    html_lines.append("</body></html>")
    with open('/root/workspace/analyze/ereer.html', 'w') as file:
        file.write('\n'.join(html_lines))

# res_dict[stage2][(new_diff_type, use_stopword, adddel, use_br)]
if __name__ == "__main__":
    aaa()