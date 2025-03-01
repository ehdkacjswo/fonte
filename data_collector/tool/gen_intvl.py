import os, sys, json, argparse, pickle, itertools, subprocess, time
import numpy as np
import pandas as pd
from spiral import ronin
from collections import Counter
from tqdm import tqdm
import copy

sys.path.append('/root/workspace/data_collector/lib/')
from utils import *

CORE_DATA_DIR = "/root/workspace/data/Defects4J/core"
BIC_GT_DIR = "/root/workspace/data/Defects4J/BIC_dataset"
BASELINE_DATA_DIR = "/root/workspace/data/Defects4J/baseline"
DIFF_DATA_DIR = '/root/workspace/data/Defects4J/diff'

diff_tool_list = ['base', 'gumtree', 'file'] # Tool to get diff (Tracker, Tracker + GumTree, Whole file)
diff_type_list = ['base', 'id'] # Type of diff data (Pure code, Identifier)
doc_level_list = ['commit'] # Level of document unit (Per commit, Per src, Per method)
#doc_level_list = ['commit', 'src', 'method']
# Currently method only has line range (May add data later (method name, signature available))

# Convert line interval to character interval
def line_to_char_intvl(commit, src_path, line_intvl):
    
    # Empty character interval for empty line interval
    if line_intvl.is_empty():
        return CustomInterval()
    
    # Get the target file text
    code_txt = get_src_from_commit(commit, src_path)

    # Error raised while copying file
    if code_txt is None:
        log('gen_intvl', f'[ERROR] Failed to get file {commit}:{src_path}')
        return None
    
    # Split the line while preserving new lines
    lines = code_txt.splitlines(True)
    
    # Convert line interval to token interval
    char_intvl, char_cnt = CustomInterval(), 0

    for line_cnt, line in enumerate(lines):
        next_char_cnt = char_cnt + len(line)
        if line_cnt in line_intvl:
            char_intvl |= CustomInterval(char_cnt, next_char_cnt - 1)
        char_cnt = next_char_cnt

    return char_intvl

# data : message, src_path, diff
# Works for "git" tracker, May not be the same for others
def gen_intvl(track_intvl, gumtree_intvl, diff_tool, diff_type, doc_level):
    ret_dict = dict()

    # Get line range diff 
    for src_path, src_dict in track_intvl.items():
        for method_info, method_dict in src_dict.items():
            for commit, commit_dict in method_dict.items():
                ret_dict.setdefault(commit, dict())

                for path_tup, intvl_dict in commit_dict.items():
                    #if doc_level == 'commit':
                    if diff_tool == 'file': # Whole file doesn't need line difference
                        ret_dict[commit].setdefault(path_tup, {'addition' : CustomInterval(-inf, inf), 'deletion' : CustomInterval(-inf, inf)})
                    
                    else: 
                        ret_dict[commit].setdefault(path_tup, {'addition' : CustomInterval(), 'deletion' : CustomInterval()})
                
                        ret_dict[commit][path_tup]['addition'] |= intvl_dict['addition']['line_diff']
                        ret_dict[commit][path_tup]['deletion'] |= intvl_dict['deletion']['line_diff']
    
    # Line range diff to char range diff
    #if doc_level == 'commit':
    for commit, commit_dict in ret_dict.items():
        for path_tup, intvl_dict in commit_dict.items():
            if diff_tool == 'file':
                add_intvl, del_intvl = CustomInterval(-inf, inf), CustomInterval(-inf, inf)

            else:
                add_intvl = line_to_char_intvl(commit, path_tup[1], intvl_dict['addition'])
                del_intvl = line_to_char_intvl(commit + '~1', path_tup[0], intvl_dict['deletion'])
    
                # Apply GumTree differencing
                if diff_tool == 'gumtree':
                    add_intvl |= gumtree_intvl[commit][path_tup].get('addition', {}).get('char_diff', CustomInterval())
                    del_intvl |= gumtree_intvl[commit][path_tup].get('deletion', {}).get('char_diff', CustomInterval())
            
            # Get identifiers
            if diff_type == 'id':
                id_add_intvl = gumtree_intvl[commit][path_tup].get('addition', {}).get('char_id', {'class':CustomInterval(), 'method':CustomInterval(), 'variable':CustomInterval(), 'comment':CustomInterval()})
                id_del_intvl = gumtree_intvl[commit][path_tup].get('deletion', {}).get('char_id', {'class':CustomInterval(), 'method':CustomInterval(), 'variable':CustomInterval(), 'comment':CustomInterval()})

                intvl_dict['addition'] = dict()
                intvl_dict['deletion'] = dict()

                for id_type in id_add_intvl.keys():
                    intvl_dict['addition'][id_type] = add_intvl & id_add_intvl[id_type]
                    intvl_dict['deletion'][id_type] = del_intvl & id_del_intvl[id_type]
            
            else:
                intvl_dict['addition'] = {'diff' : add_intvl}
                intvl_dict['deletion'] = {'diff' : del_intvl}
        
    return ret_dict

def main(pid, vid):
    log('gen_intvl', f'Working on project {pid}-{vid}b')

    # Checkout Defects4J project
    p = subprocess.Popen(['sh', '/root/workspace/lib/checkout.sh', pid, vid], \
        stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out_txt, err_txt = p.communicate()

    if p.returncode != 0:
        log('gen_intvl', '[ERROR] Checkout failed', out_txt, err_txt)
        return
    
    # Change working directory to target Defects4J project
    try:
        os.chdir(f'/tmp/{pid}-{vid}b/')
    except:
        log('gen_intvl', '[ERROR] Moving directory failed')
        return
    
    # Load interval data
    diff_data_dir = os.path.join(DIFF_DATA_DIR, f'{pid}-{vid}b')
    os.makedirs(diff_data_dir, exist_ok=True)

    with open(os.path.join(diff_data_dir, 'track_intvl.pkl'), 'rb') as file:
        track_intvl_dict = pickle.load(file)
    
    with open(os.path.join(diff_data_dir, 'gumtree_intvl.pkl'), 'rb') as file:
        gumtree_intvl_dict = pickle.load(file)
    
    # Gather interval data
    res_dict = dict()
    param_list = list(itertools.product(diff_tool_list, diff_type_list, doc_level_list))

    start_time = time.time()
    for tracker_setting, track_intvl in track_intvl_dict.items():
        setting_dict = dict(tracker_setting)
        tracker = setting_dict['tracker']

        for (diff_tool, diff_type, doc_level) in param_list:
            res_dict[frozenset((setting_dict | \
                {'diff_tool' : diff_tool, 'diff_type' : diff_type, 'doc_level' : doc_level}).items())] = \
                gen_intvl(track_intvl, gumtree_intvl_dict[tracker_setting], diff_tool, diff_type, doc_level)
    
    end_time = time.time()
    log('gen_intvl', f'({tracker}) : {time_to_str(start_time, end_time)}')

    with open(os.path.join(diff_data_dir, 'total_intvl.pkl'), 'wb') as file:
        pickle.dump(res_dict, file)
