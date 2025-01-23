import os, json, argparse, pickle, sys, itertools, subprocess, logging
import pandas as pd
from tqdm import tqdm
from interval import interval

sys.path.append('/root/workspace/diff_util/lib/')
from diff import *
from gumtree import *

DIFF_DATA_DIR = '/root/workspace/data/Defects4J/diff'
CORE_DATA_DIR = '/root/workspace/data/Defects4J/core'
BASE_DATA_DIR = '/root/workspace/data/Defects4J/baseline'

# Get the unchanged file data
# [(commit, src_path)]
def get_style_change_data(coredir, tool='git', with_Rewrite=True):
    if with_Rewrite == 'skip':
        return []
    #postfix = "" if with_Rewrite else "_noOpenRewrite"
    #postfix = "" if with_Rewrite else ""
    postfix = ""

    if with_Rewrite == 'precise':
        val_df = pd.read_csv(
            os.path.join(coredir, tool, f"precise_validation_noOpenRewrite.csv"), 
            header=None,
            names=["commit", "before_src_path", "after_src_path", "AST_diff"])

        unchanged_df = val_df[val_df["AST_diff"] == "U"]
        return list(zip(unchanged_df["commit"], unchanged_df["before_src_path"], unchanged_df["after_src_path"]))

    else:
        val_df = pd.read_csv(
            os.path.join(coredir, tool, f"validation_noOpenRewrite.csv"), 
            header=None,
            names=["commit", "src_path", "AST_diff"])

        unchanged_df = val_df[val_df["AST_diff"] == "U"]
        return list(zip(unchanged_df["commit"], unchanged_df["src_path"], unchanged_df["src_path"]))

def git_stage2(pid, vid, stage2):
    # Load related diff data
    with open(os.path.join(DIFF_DATA_DIR, f'{pid}-{vid}b', 'stage1', 'git_diff.pkl'), 'rb') as file:
        git_diff = pickle.load(file)
    
    # Get list of style change commits
    excluded = get_style_change_data(os.path.join(CORE_DATA_DIR, f'{pid}-{vid}b/'), 'git', stage2)

    # Merge the diff data
    res_dict = dict()

    for commit_hash, commit_diff in git_diff.items(): # Iterate through commits
        addition_dict = dict()
        deletion_dict = dict()

        for (before_src_path, after_src_path), src_diff in commit_diff.items(): # Iterate through source files editted by commit
            if (commit_hash, before_src_path, after_src_path) in excluded: # Exclude style change
                continue

            addition_content_dict = addition_dict.get(after_src_path, dict())
            deletion_content_dict = deletion_dict.get(before_src_path, dict())

            for line, content in src_diff['addition'].items():
                if line not in addition_content_dict:
                    addition_content_dict[line] = content
                
                elif content != addition_content_dict[line]: # Different diff data for same source file
                    with open('/root/workspace/error.txt', 'a') as file:
                        file.write(f'Different addition content!!! {commit_hash} {after_src_path} {line}')
            
            for line, content in src_diff['deletion'].items():
                if line not in deletion_content_dict:
                    deletion_content_dict[line] = content
                
                elif content != deletion_content_dict[line]: # Different diff data for same source file
                    with open('/root/workspace/error.txt', 'a') as file:
                        file.write(f'Different deletion content!!! {commit_hash} {before_src_path} {line}')
            
            addition_dict[after_src_path] = {'diff' : list(addition_content_dict.values())}
            deletion_dict[before_src_path] = {'diff' : list(deletion_content_dict.values())}
        
        res_dict[commit_hash] = {'addition' : addition_dict, 'deletion' : deletion_dict}
    
    return res_dict

# Helper function encoding tokens
def gumtree_stage2_commit(commit_hash, commit_interval, modify, classify_token):
    res_dict = dict()

    for src_path, src_interval in commit_interval[modify].items():
        # Empty interval
        if src_interval.is_empty():
            res_dict[src_path] = dict()
            continue

        if modify == 'addition':
            p = subprocess.Popen(f'git show {commit_hash}:{src_path}', shell=True, stdout=subprocess.PIPE)
        elif modify == 'deletion':
            p = subprocess.Popen(f'git show {commit_hash}~1:{src_path}', shell=True, stdout=subprocess.PIPE)

        code_txt, _ = p.communicate()

        # Error raised while copying file
        if p.returncode != 0:
            # Actual addition occured but failed to copy file
            with open('/root/workspace/error.txt', 'a') as file:
                file.write(f'Failed to copy file {commit_hash}:{src_path}')
            res_dict[src_path] = dict()
            continue
        
        else:
            code_txt = code_txt.decode(encoding='utf-8', errors='ignore')

            # Don't classify tokens
            if not classify_token:
                res_dict[src_path] = {'diff':[]}
                for token_interval in src_interval:
                    if token_interval[0] != token_interval[1]: # Ignore empty interval
                        res_dict[src_path]['diff'] += [''.join(code_txt[int(token_interval[0]) + 1 : int(token_interval[1]) + 1])]
                continue

            with open('/root/workspace/tmp/tmp.java', 'w') as file:
                file.write(code_txt)
        
        res_gumtree_parse = gumtree_parse('tmp.java', src_interval)
        
        # Error while parsing
        if res_gumtree_parse is None:
            res_dict[src_path] = dict()
            continue
        
        res_dict[src_path] = res_gumtree_parse
    
    return res_dict

def gumtree_stage2(pid, vid, stage2, classify_token):
    # Load related diff data
    diff_data_dir = os.path.join(DIFF_DATA_DIR, f'{pid}-{vid}b/stage1')
    #diff_data_dir = os.path.join(DIFF_DATA_DIR, f'{pid}-{vid}b')
    
    with open(os.path.join(DIFF_DATA_DIR, f'{pid}-{vid}b', 'stage1', 'gumtree_diff_interval.pkl'), 'rb') as file:
        gumtree_diff = pickle.load(file)

    # Get list of style change commits
    excluded = get_style_change_data(os.path.join(CORE_DATA_DIR, f'{pid}-{vid}b/'), 'git', stage2)

    # Join the intervals
    # {commit_hash : {addition / deletion : {src_path : interval}}}
    gumtree_interval = dict()
    
    for commit_hash, commit_diff in gumtree_diff.items(): # Iterate through commits
        addition_interval = dict()
        deletion_interval = dict()

        for (before_src_path, after_src_path), src_interval in commit_diff.items(): # Iterate through source files editted by commit
            if (commit_hash, before_src_path, after_src_path) in excluded: # Exclude style change
                continue

            if after_src_path != '/dev/null':
                addition_interval[after_src_path] = \
                    addition_interval.get(after_src_path, CustomInterval()) | src_interval['addition']
            
            if before_src_path != '/dev/null':
                deletion_interval[before_src_path] = \
                    deletion_interval.get(before_src_path, CustomInterval()) | src_interval['deletion']

        gumtree_interval[commit_hash] = {'addition' : dict(), 'deletion' : dict()}

        for src_path, src_interval in addition_interval.items():
            gumtree_interval[commit_hash]['addition'][src_path] = src_interval
        
        for src_path, src_interval in deletion_interval.items():
            gumtree_interval[commit_hash]['deletion'][src_path] = src_interval
    
    # {commit_hash : {addition / deletion : {src_path : {src_path, token_types}}}}
    res_dict = dict()

    for commit_hash, commit_interval in gumtree_interval.items():
        res_dict[commit_hash] = dict()
        res_dict[commit_hash]['addition'] = gumtree_stage2_commit(commit_hash, commit_interval, 'addition', classify_token)
        res_dict[commit_hash]['deletion'] = gumtree_stage2_commit(commit_hash, commit_interval, 'deletion', classify_token)

    return res_dict

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute commit scores")
    parser.add_argument('--project', '-p', type=str, default="Closure",
        help="Target project name (default: Closure)")
    parser.add_argument('--version', '-v', type=int, default=21,
        help="Target buggy version (default: 21)")
    args = parser.parse_args()

    #logging.basicConfig(filename="/root/workspace/diff_util/collector/", level=logging.INFO)

    print('Working on {}_{}b'.format(args.project, args.version))
    with open('/root/workspace/error.txt', 'a') as file:
        file.write('Working on {}_{}b\n'.format(args.project, args.version))

    diff_type_list = ['git', 'gumtree_base', 'gumtree_class']
    stage2_list = ['skip'] # ['skip', True, False] Skip stage or use OpenRewrite or not
    param_list = list(itertools.product(diff_type_list, stage2_list))

    gumtree_dir = os.path.join(DIFF_DATA_DIR, f'{args.project}-{args.version}b')
    os.makedirs(gumtree_dir, exist_ok=True)
    
    res_dict = dict()
    for stage2 in stage2_list:
        res_dict[stage2] = dict()
        
        for diff_type in diff_type_list:
            if diff_type == 'git':
                res_dict[stage2][diff_type] = git_stage2(args.project, args.version, stage2)
            else:
                res_dict[stage2][diff_type] = gumtree_stage2(args.project, args.version, stage2, diff_type.endswith('class'))

    with open(os.path.join(gumtree_dir, f'stage2.pkl'), 'wb') as file:
        pickle.dump(res_dict, file)
    
    print(res_dict)