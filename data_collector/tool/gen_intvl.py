import os, sys, json, argparse, pickle, itertools, subprocess, time
import numpy as np
import pandas as pd
from spiral import ronin
from collections import Counter
from tqdm import tqdm
import copy
from interval import inf

sys.path.append('/root/workspace/data_collector/lib/')
from utils import *

CORE_DATA_DIR = "/root/workspace/data/Defects4J/core"
BIC_GT_DIR = "/root/workspace/data/Defects4J/BIC_dataset"
BASELINE_DATA_DIR = "/root/workspace/data/Defects4J/baseline"
DIFF_DATA_DIR = '/root/workspace/data/Defects4J/diff'

diff_tool_list = ['base', 'gumtree', 'file'] # Tool to get diff (Tracker, Tracker + GumTree, Whole file)
diff_type_list = ['base', 'gumtree_id', 'greedy_id'] # Type of diff data (Pure code, Identifier)
stage2_list = ['skip', 'precise']
#doc_level_list = ['commit', 'src', 'method'] # Level of document unit (Per commit, Per src, Per method)
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

# Aggregate line intervals from git
# {commit : {adddel : {src_path : interval} } }
def aggr_git_intvl(intvl_dict, excluded):
    res_dict = dict()

    for bug_intvl_dict in intvl_dict.values():
        for commit_dict in bug_intvl_dict.values():
            for commit, path_dict in commit_dict.items():
                for path_tup, adddel_dict in path_dict.items():
                    # Ignore style changes
                    if (commit, path_tup[0], path_tup[1]) in excluded:
                        continue
                    
                    res_dict.setdefault(commit, {'addition' : dict(), 'deletion' : dict()})

                    for ind, adddel in enumerate(['deletion', 'addition']):
                        # Ignore /dev/null
                        if path_tup[ind] != '/dev/null':
                            res_dict[commit][adddel][path_tup[ind]] = \
                                res_dict[commit][adddel].get(path_tup[ind], CustomInterval()) | adddel_dict[adddel]['line_diff']
    
    # Git contains line intervals, Convert them to character interval
    for commit, adddel_dict in res_dict.items():
        for adddel, path_dict in adddel_dict.items():
            target_commit = commit + ('' if adddel == 'addition' else '~1')

            for src_path, line_intvl in path_dict.items():
                char_intvl = line_to_char_intvl(target_commit, src_path, line_intvl)

                # Failed to convert line interval to character interval
                if char_intvl is None:
                    return None
                
                else:
                    path_dict[src_path] = char_intvl
    
    return res_dict

# Aggregate diff intervals from GumTree
# {commit : {adddel : {src_path : interval} } }
def aggr_gumtree_diff(intvl_dict, excluded):
    res_dict = dict()

    for commit, path_dict in intvl_dict.items():
        for path_tup, adddel_dict in path_dict.items():
            # Ignore style changes
            if (commit, path_tup[0], path_tup[1]) in excluded:
                continue
            
            res_dict.setdefault(commit, {'addition' : dict(), 'deletion' : dict()})

            for ind, adddel in enumerate(['deletion', 'addition']):
                # Ignore /dev/null
                if path_tup[ind] != '/dev/null':
                    res_dict[commit][adddel][path_tup[ind]] = \
                        res_dict[commit][adddel].get(path_tup[ind], CustomInterval()) | adddel_dict[adddel]
    
    return res_dict

# {setting : {commit : {adddel : {src_path : interval} } } }
def gen_diff_tool_intvl(track_intvl, gumtree_diff_intvl):
    res_dict = dict()

    for diff_tool in diff_tool_list:
        setting = frozenset({'diff_tool' : diff_tool}.items())
        res_dict[setting] = dict()

        for commit, adddel_dict in track_intvl.items():
            res_dict[setting][commit] = {'addition' : dict(), 'deletion' : dict()}

            for adddel, path_dict in adddel_dict.items():
                for src_path, base_intvl in path_dict.items():
                    # Use interval from tracker
                    if diff_tool == 'base':
                        res_dict[setting][commit][adddel][src_path] = base_intvl
                    
                    # Use interval from tracker + GumTree
                    elif diff_tool == 'gumtree':
                        res_dict[setting][commit][adddel][src_path] = \
                            base_intvl & gumtree_diff_intvl[commit][adddel][src_path]

                    # Use whole file (No diff)
                    elif diff_tool == 'file':
                        res_dict[setting][commit][adddel][src_path] = CustomInterval(-inf, inf)
    
    return res_dict

# data : message, src_path, diff
# Works for "git" tracker, May not be the same for others
def gen_diff_type_intvl(diff_tool_intvl, id_intvl_dict):
    res_dict = dict()

    for setting, commit_dict in diff_tool_intvl.items():
        res_dict[setting] = dict()

        for commit, adddel_dict in commit_dict.items():
            res_dict[setting][commit] = {'addition' : dict(), 'deletion' : dict()}

            for adddel, path_dict in adddel_dict.items():
                for src_path, intvl in path_dict.items():
                    # Not using identifier
                    if id_intvl_dict is None:
                        res_dict[setting][commit][adddel][src_path] = {'diff' : intvl}
                    
                    else:
                        res_dict[setting][commit][adddel][src_path] = dict()

                        for id_type, id_intvl in id_intvl_dict[commit][adddel][src_path].items():
                            #print(id_intvl)
                            res_dict[setting][commit][adddel][src_path][id_type] = id_intvl & intvl
        
    return res_dict

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
    
    with open(os.path.join(DIFF_DATA_DIR, f'{pid}-{vid}b', 'greedy_intvl.pkl'), 'rb') as file:
        greedy_id_dict = pickle.load(file)
    
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

            # Aggregate diff intervals from tracker (Convert to character level interval if neccessary)
            if tracker == 'git':
                track_intvl_aaa = aggr_git_intvl(track_intvl, excluded)
                if track_intvl_aaa is None:
                    return None
            
            # Aggregate diff intervals from gumtree & Get intervals based on diff tool
            gumtree_diff = aggr_gumtree_diff(gumtree_intvl_dict['char_diff'][setting], excluded)
            diff_tool_intvl = gen_diff_tool_intvl(track_intvl_aaa, gumtree_diff)

            for diff_type in diff_type_list:
                # Don't use identifiers
                if diff_type == 'base':
                    id_intvl = None

                # Use GumTree based identifiers
                elif diff_type == 'gumtree_id':
                    id_intvl = gumtree_intvl_dict['char_id'][setting]
                
                # Use greedy identifiers
                elif diff_type == 'greedy_id':
                    id_intvl = greedy_id_dict[setting]

                diff_type_intvl = gen_diff_type_intvl(diff_tool_intvl, id_intvl)

                for sub_setting, asdf in diff_type_intvl.items():
                    res_dict[stage2][frozenset((dict(sub_setting) | {'diff_type' : diff_type}).items())] = asdf
    
    end_time = time.time()
    log('gen_intvl', f'{time_to_str(start_time, end_time)}')
    print(res_dict)

    with open(os.path.join(diff_data_dir, 'total_intvl.pkl'), 'wb') as file:
        pickle.dump(res_dict, file)
