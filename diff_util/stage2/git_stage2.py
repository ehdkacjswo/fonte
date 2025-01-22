import os, json, argparse, pickle, sys, itertools, subprocess
import pandas as pd
from tqdm import tqdm
from interval import interval

sys.path.append('/root/workspace/diff_util/lib/')
from diff import *
from encoder import *
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

# Encode the raw diff data
# skip_stage_2 = Excluding style change diff, with_Rewrite = , use_stopword
# Encoded data : {commit_hash : [addition_list, deletion_list, msg_encode]}
# addition/deletion dict : [(before/after_src_path_encode, content_encode_sum)]

# Data has form of {commit_hash : {type : token_list}}
def encode_git(pid, vid, stage2):
    # Load related diff data
    diff_data_dir = os.path.join(DIFF_DATA_DIR, f'{pid}-{vid}b', 'stage1')
    with open(os.path.join(diff_data_dir, 'git_diff.pkl'), 'rb') as file:
        git_diff = pickle.load(file)
    
    # Get list of style change commits
    excluded = get_style_change_data(os.path.join(CORE_DATA_DIR, f'{pid}-{vid}b/'), 'git', stage2)

    # Merge the diff data
    diff_dict = dict()

    for commit_hash, commit_diff in git_diff.items(): # Iterate through commits
        addition_dict = dict()
        deletion_dict = dict()

        for (before_src_path, after_src_path), git_diff_dict in commit_diff.items(): # Iterate through source files editted by commit
            if (commit_hash, before_src_path, after_src_path) in excluded: # Exclude style change
                continue

            addition_content_dict = addition_dict.get(after_src_path, dict())
            deletion_content_dict = deletion_dict.get(before_src_path, dict())

            for line, content in git_diff_dict['addition'].items():
                if line not in addition_content_dict:
                    addition_content_dict[line] = content
                
                elif content != addition_content_dict[line]: # Different diff data for same source file
                    with open('/root/workspace/error.txt', 'a') as file:
                        file.write(f'Different addition content!!! {commit_hash} {after_src_path} {line}')
            
            for line, content in git_diff_dict['deletion'].items():
                if line not in deletion_content_dict:
                    deletion_content_dict[line] = content
                
                elif content != deletion_content_dict[line]: # Different diff data for same source file
                    with open('/root/workspace/error.txt', 'a') as file:
                        file.write(f'Different deletion content!!! {commit_hash} {before_src_path} {line}')
            
            addition_dict[after_src_path] = {'diff' : list(addition_content_dict.values())}
            deletion_dict[before_src_path] = {'diff' : list(deletion_content_dict.values())}
        
        diff_dict[commit_hash] = {'addition' : addition_dict, 'deletion' : deletion_dict}
    
    return diff_dict

if __name__ == "__main__":
    stage2_list = ['skip'] # ['skip', True, False] Skip stage or use OpenRewrite or not
    
    for project_dir in tqdm(os.listdir(DIFF_DATA_DIR)):
        with open('/root/workspace/error.txt', 'a') as file:
            file.write(f'Working on project {project_dir}')
        print(f'Working on project {project_dir}')
        [pid, vid] = project_dir[:-1].split("-")

        git_stage2_dir = os.path.join(DIFF_DATA_DIR, project_dir, 'stage2', 'git')
        os.makedirs(git_stage2_dir, exist_ok=True)

        for stage2 in ['skip']:
            git_stage2 = encode_git(pid, vid, stage2)

            with open(os.path.join(git_stage2_dir, f'{stage2}.pkl'), 'wb') as file:
                pickle.dump(git_stage2, file)