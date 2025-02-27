import os, sys, json, argparse, pickle, itertools, subprocess
import numpy as np
import pandas as pd
from spiral import ronin
from collections import Counter
from tqdm import tqdm

import copy
sys.path.append('/root/workspace/data_collector/lib/')
from encoder import sum_encode
from gumtree import CustomInterval

CORE_DATA_DIR = "/root/workspace/data/Defects4J/core"
BIC_GT_DIR = "/root/workspace/data/Defects4J/BIC_dataset"
BASELINE_DATA_DIR = "/root/workspace/data/Defects4J/baseline"
DIFF_DATA_DIR = '/root/workspace/data/Defects4J/diff'

diff_tool_list = ['base', 'gumtree'] # Tool to get diff (Tracker, Tracker + GumTree)
diff_type_list = ['base', 'id'] # Type of diff data (Pure code, Identifier)
doc_level_list = ['commit'] # Level of document unit (Per commit, Per src, Per method)
#doc_level_list = ['commit', 'src', 'method']
# Currently method only has line range (May add data later (method name, signature available))

def log(txt, out_txt=None, err_txt=None):
    with open('/root/workspace/data_collector/log/get_feature.log', 'a') as file:
        file.write(txt + '\n')

        if out_txt is not None:
            file.write('[ERROR] OUT\n' + out_txt.decode(encoding='utf-8', errors='ignore') + '\n')
        
        if err_txt is not None:
            file.write('[ERROR] ERR\n' + err_txt.decode(encoding='utf-8', errors='ignore') + '\n')

# Convert line level range to character level range
def line_to_char_intvl(commit, src_path, line_intvl)
    
    # Empty character interval for empty line interval
    if line_intvl.is_empty():
        return CustomInterval()
    
    # Get the target file text
    p = subprocess.Popen(['git', 'show', f'{commit}:{src_path}'], \
    stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    code_txt, err_txt = p.communicate()

    # Error raised while copying file
    if p.returncode != 0:
        log(f'[ERROR] Failed to get file {commit}:{src_path}', code_txt, err_txt)
        return None
    
    code_txt = code_txt.decode(encoding='utf-8', errors='ignore')
    lines = code_txt.splitlines(True) # Split the line while preserving new lines

    char_intvl, char_cnt = CustomInterval(), 0
    
    # Add range for line in given range
    for line_cnt, line in enumerate(lines):
        next_char_cnt = char_cnt + len(line)

        if line_cnt in line_intvl:
            char_intvl |= CustomInterval(char_cnt, next_char_cnt - 1)
        
        char_cnt = next_char_cnt

    return char_intvl


# data : message, src_path, diff
def get_intvl(track_intvl, gumtree_intvl, diff_tool, diff_type, doc_level):
    ret_dict = dict()

    # Get line range diff (Maybe different tracker has different level)
    for src_path, src_dict in track_intvl.items():
        for method_info, method_dict in src_dict.items():
            for commit, commit_dict in method_dict.items():
                ret_dict.setdefault(commit, dict())

                for path_tup, intvl_dict in commit_dict.items():
                    #if doc_level == 'commit':
                    ret_dict[commit].setdefault(path_tup, {'addition' : CustomInterval(), 'deletion' : CustomInterval()})
                
                    ret_dict[commit][path_tup]['addition'] |= intvl_dict['addition']['line_diff']
                    ret_dict[commit][path_tup]['deletion'] |= intvl_dict['deletion']['line_diff']
    
    # Line range diff to char range diff
    #if doc_level == 'commit':
    for commit, commit_dict in ret_dict.items():
        for path_tup, intvl_dict in commit_dict.items():
            intvl_dict['addition'] = line_to_char_intvl(commit, path_tup[1], intvl_dict['addition'])
            intvl_dict['deletion'] = line_to_char_intvl(commit + '~1', path_tup[0], intvl_dict['deletion'])
    
            if diff_tool == 'gumtree':
                intvl_dict['addition'] |= gumtree_intvl[commit][path_tup].get('addition', {}).get('char_diff', CustomInterval())
                intvl_dict['deletion'] |= gumtree_intvl[commit][path_tup].get('deletion', {}).get('char_diff', CustomInterval())
            
            if diff_type == 'id':
                intvl_dict['addition'] |= gumtree_intvl[commit][path_tup].get('addition', {}).get('char_id', CustomInterval())
                intvl_dict['deletion'] |= gumtree_intvl[commit][path_tup].get('deletion', {}).get('char_id', CustomInterval())
        
    return res_dict

def main(pid, vid):
    log(f'Working on project {pid}-{vid}b')

    # Checkout Defects4J project
    p = subprocess.Popen(['sh', '/root/workspace/lib/checkout.sh', pid, vid], \
        stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out_txt, err_txt = p.communicate()

    if p.returncode != 0:
        log('[ERROR] Checkout failed', out_txt, err_txt)
        return
    
    # Change working directory to target Defects4J project
    try:
        os.chdir(f'/tmp/{pid}-{vid}b/')
    except:
        log('[ERROR] Moving directory failed')
        return
    
    # Load interval data
    diff_data_dir = os.path.join(DIFF_DATA_DIR, f'{pid}-{vid}b')
    os.makedirs(diff_data_dir, exist_ok=True)

    with open(os.path.join(savedir, 'track_intvl.pkl'), 'rb') as file:
        track_intvl_dict = pickle.load(file)
    
    with open(os.path.join(savedir, 'gumtree_intvl.pkl'), 'rb') as file:
        gumtree_intvl = pickle.load(file)
    
    res_dict = dict()
    param_list = list(itertools.product(diff_tool_list, diff_type_list, doc_level_list))

    for tracker_setting, track_intvl in track_intvl_dict.items():
        setting_dict = dict(tracker_setting)
        tracker = setting_dict['tracker']

        for (diff_tool, diff_type, doc_level) in param_list:
            res_dict[frozenset((setting_dict | \
                {'diff_tool' : diff_tool, 'diff_type' : diff_type, 'doc_level' : doc_level}).items())] = \
                gen_intvl(track_intvl, gumtree_intvl_dict[tracker_setting], diff_tool, diff_type, doc_level)

    # Load the previous result if possible
    feature_save_path = os.path.join(diff_data_dir, f'feature.pkl')

    """if os.path.isfile(feature_save_path):
        with open(feature_save_path, 'rb') as file:
            res_dict = pickle.load(file)
    
    else:
        res_dict = dict()"""
    
    """res_dict = dict()

    for stage2, sub_dict in encode_dict.items():
        res_dict[stage2] = dict()

        for setting, encode_data in sub_dict.items():
            setting_dict = dict(setting)
            feature_dict = get_feature(encode_data=encode_data, setting_dict=setting_dict)

            for adddel, feature_data in feature_dict.items():
                new_setting = frozenset((setting_dict | {'adddel' : adddel}).items())
                res_dict[stage2][new_setting] = feature_data"""
    
    with open(os.path.join(diff_data_dir, 'feature.pkl'), 'rb') as file:
        res_dict = pickle.load(file)
    
    for stage2, sub_dict in res_dict:
        for setting, feature_data in sub_dict.items():
            setting_dict = dict(setting)

            if setting_dict['diff_type'] != 'gumtree_class':
                continue

            new_setting = frozenset((setting_dict | {'diff_type' : 'gumtree_identifier'}).items())
            new_feautre = copy.deepcopy(feature_data)

            ind = 3 if setting_dict['adddel'] == 'all_sep' else 2
            sub_feature = []
            for i in range(ind, len(new_feature)):
                sub_feature = sum_encode(sub_feature, new_feature[i])
            
            new_feature = new_feature[0:i] + [sub_feature]
            res_dict[new_setting]
    
    with open(os.path.join(diff_data_dir, 'feature.pkl'), 'wb') as file:
        pickle.dump(res_dict, file)
