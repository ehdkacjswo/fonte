import os, json, argparse, pickle, sys, itertools, subprocess
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

# Helper function encoding tokens
def encode_token(commit_hash, src_interval_dict, modify, classify_token):
    src_token_dict = dict()

    for src_path, src_interval in src_interval_dict[modify].items():
        # Empty interval
        if len(src_interval) == 0:
            src_encode_dict[src_path] = dict()
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
            src_token_dict[src_path] = dict()
            continue
        
        else:
            code_txt = code_txt.decode(encoding='utf-8', errors='ignore')

            if not classify_token:
                src_token_dict[src_path] = {'token':[]}
                for token_interval in src_interval:
                    src_token_dict[src_path]['token'] += [''.join(code_txt[int(token_interval.inf) : int(token_interval.sup) + 1])]
                continue

            with open('/root/workspace/tmp/tmp.java', 'w') as file:
                file.write(code_txt)
        
        token_dict = gumtree_parse('tmp.java', src_interval)
        
        # Error while parsing
        if token_dict is None:
            src_token_dict[src_path] = dict()
            continue
        
        src_token_dict[src_path] = token_dict
    
    return src_token_dict

def encode_gumtree(pid, vid, stage2, classify_token):
    # Load related diff data
    diff_data_dir = os.path.join(DIFF_DATA_DIR, f'{pid}-{vid}b/stage1')
    #diff_data_dir = os.path.join(DIFF_DATA_DIR, f'{pid}-{vid}b')
    
    with open(os.path.join(diff_data_dir, 'gumtree_diff_interval.pkl'), 'rb') as file:
        gumtree_diff = pickle.load(file)
    
    # Get list of style change commits
    excluded = get_style_change_data(os.path.join(CORE_DATA_DIR, f'{pid}-{vid}b/'), 'git', stage2)

    # Join the intervals
    # {commit_hash : {addition / deletion : {src_path : interval}}}
    total_interval = dict()
    
    for commit_hash, commit_diff in gumtree_diff.items(): # Iterate through commits
        addition_interval_dict = dict()
        deletion_interval_dict = dict()

        for (before_src_path, after_src_path), diff_interval in commit_diff.items(): # Iterate through source files editted by commit
            if (commit_hash, before_src_path, after_src_path) in excluded: # Exclude style change
                continue

            if after_src_path != '/dev/null' and len(diff_interval['addition']) > 0:
                addition_interval_dict[after_src_path] = \
                    addition_interval_dict.get(after_src_path, interval()) | diff_interval['addition']
            
            if before_src_path != '/dev/null' and len(diff_interval['deletion']):
                deletion_interval_dict[before_src_path] = \
                    deletion_interval_dict.get(before_src_path, interval()) | diff_interval['deletion']

        total_interval[commit_hash] = {'addition' : dict(), 'deletion' : dict()}

        for src_path, src_interval in addition_interval_dict.items():
            total_interval[commit_hash]['addition'][src_path] = src_interval
        
        for src_path, src_interval in deletion_interval_dict.items():
            total_interval[commit_hash]['deletion'][src_path] = src_interval
    
    # {commit_hash : {addition / deletion : {src_path : {src_path, token_types}}}}
    total_token = dict()

    for commit_hash, commit_interval in total_interval.items():
        total_token[commit_hash] = dict()
        total_token[commit_hash]['addition'] = encode_token(commit_hash, commit_interval, 'addition', classify_token)
        total_token[commit_hash]['deletion'] = encode_token(commit_hash, commit_interval, 'deletion', classify_token)
    
    print(total_token)
    return total_token

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute commit scores")
    parser.add_argument('--project', '-p', type=str, default="Closure",
        help="Target project name (default: Closure)")
    parser.add_argument('--version', '-v', type=int, default=21,
        help="Target buggy version (default: 21)")
    args = parser.parse_args()

    print('Working on {}_{}b'.format(args.project, args.version))
    with open('/root/workspace/error.txt', 'a') as file:
        file.write('Working on {}_{}b\n'.format(args.project, args.version))

    stage2_list = ['skip'] # ['skip', True, False] Skip stage or use OpenRewrite or not
    classify_token_list = [True, False] # [True, False] Use stopword or not    
    param_list = list(itertools.product(stage2_list, classify_token_list))

    gumtree_dir = os.path.join(DIFF_DATA_DIR, f'{args.project}-{args.version}b/stage2/gumtree')
    os.makedirs(gumtree_dir, exist_ok=True)
    
    for (stage2, classify_token) in param_list:
        token_dict = encode_gumtree(args.project, args.version, stage2, classify_token)

        with open(os.path.join(gumtree_dir, f'{stage2}_{classify_token}.pkl'), 'wb') as file:
            pickle.dump(token_dict, file)