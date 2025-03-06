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
stage2_list = ['skip', 'precise']
#doc_level_list = ['commit', 'src', 'method']
# Currently method only has line range (May add data later (method name, signature available))

# Get style change data list [(commit, before_src_path, after_src_path)]
def get_excluded(coredir, tool='git', stage2='skip'):
    if stage2 == 'skip':
        return []

    elif stage2 == 'precise':
        val_df = pd.read_csv(
            os.path.join(coredir, tool, f"precise_validation_noOpenRewrite.csv"), 
            header=None,
            names=["commit", "before_src_path", "after_src_path", "AST_diff"])

        unchanged_df = val_df[val_df["AST_diff"] == "U"]
        return list(zip(unchanged_df["commit"], unchanged_df["before_src_path"], unchanged_df["after_src_path"]))

# Convert line interval to character interval
# Problem : file level > Full intvl, but file may not exists
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

# Aggregate line intervals
def aggr_line_intvl(track_intvl, doc_level, excluded):
    # Get line interval data
    ret_dict = dict()

    for src_path, src_dict in track_intvl.items():
        for method_info, method_dict in src_dict.items():
            for commit, commit_dict in method_dict.items():
                for path_tup, intvl_dict in commit_dict.items():
                    # Ignore style changes
                    if (commit, path_tup[0], path_tup[1]) in excluded:
                        continue

                    #if doc_level == 'commit':
                    # Add commit only when at least one change is not style change
                    ret_dict.setdefault(commit, dict())
                    ret_dict[commit].setdefault(path_tup, {'addition' : CustomInterval(), 'deletion' : CustomInterval()})
            
                    ret_dict[commit][path_tup]['addition'] |= intvl_dict['addition']['line_diff']
                    ret_dict[commit][path_tup]['deletion'] |= intvl_dict['deletion']['line_diff']
    
    return ret_dict

# data : message, src_path, diff
# Works for "git" tracker, May not be the same for others
def gen_intvl(line_intvl_dict, gumtree_intvl, diff_tool, diff_type, doc_level):
    # Line range diff to char range diff
    #if doc_level == 'commit':
    ret_dict = dict()

    for commit, commit_dict in line_intvl_dict.items():
        ret_dict.setdefault(commit, {'addition' : dict(), 'deletion' : dict()})

        for path_tup, path_tup_dict in commit_dict.items():
            for adddel, line_intvl in path_tup_dict.items():
                # For file level differencing, use full file only when actual 
                # 이거 line level이 아니라 method track level로 해야되는거 아니냐?
                if not line_intvl.is_empty():
                    src_path = path_tup[1 if adddel == 'addition' else 0]
                    ret_dict[commit][adddel].setdefault(src_path, dict())

                    if diff_tool == 'file':
                        token_intvl = CustomInterval(-inf, inf)
                    else:
                        token_intvl = line_to_char_intvl(commit + ('' if adddel == 'addition' else '~1'), src_path, line_intvl)

                        # Failed to get tokens
                        if token_intvl is None:
                            return None
            
                        # Apply GumTree differencing
                        if diff_tool == 'gumtree':
                            token_intvl &= gumtree_intvl[commit][path_tup].get(adddel, {}).get('char_diff', CustomInterval())
                    
                    # Get identifiers
                    if diff_type == 'id':
                        id_intvl_dict = gumtree_intvl[commit][path_tup].get(adddel, {}).get('char_id', {})

                        ret_dict[commit][adddel][src_path] = \
                            {id_type : ret_dict[commit][adddel][src_path].get(id_type, CustomInterval()) | \
                            (token_intvl & id_intvl) for id_type, id_intvl in id_intvl_dict.items()}
                    
                    else:
                        ret_dict[commit][adddel][src_path]['diff'] = \
                            ret_dict[commit][adddel][src_path].get('diff', CustomInterval()) | token_intvl
        
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
    param_list = list(itertools.product(diff_tool_list, diff_type_list))

    start_time = time.time()
    for setting, track_intvl in track_intvl_dict.items():
        setting_dict = dict(setting)
        tracker = setting_dict['tracker']

        for stage2 in stage2_list:
            res_dict.setdefault(stage2, dict())
            excluded = get_excluded(os.path.join(CORE_DATA_DIR, f'{pid}-{vid}b/'), tracker, stage2)

            for doc_level in doc_level_list:
                line_intvl_dict = aggr_line_intvl(track_intvl, doc_level, excluded)

                for (diff_tool, diff_type) in param_list:
                    res_dict[stage2][frozenset((setting_dict | \
                        {'diff_tool' : diff_tool, 'diff_type' : diff_type, 'doc_level' : doc_level}).items())] = \
                        gen_intvl(line_intvl_dict, gumtree_intvl_dict[setting], diff_tool, diff_type, doc_level)
    
    end_time = time.time()
    log('gen_intvl', f'{time_to_str(start_time, end_time)}')

    with open(os.path.join(diff_data_dir, 'total_intvl.pkl'), 'wb') as file:
        pickle.dump(res_dict, file)
