import io
import json
import logging
import os
import re
import argparse
import subprocess
import sys
import pickle

import pandas as pd
import git # GitPython
from tabulate import tabulate
from nltk.corpus import stopwords

sys.path.append('/root/workspace/diff_util/lib/')
from diff import Diff_commit

CORE_DATA_DIR = '/root/workspace/data/Defects4J/core/'

# Get range of suspicious parts
# Return = {src_path : set(line_start, line_end)}
def get_range_dict(pid, vid, tool='git'):
    commit_path = os.path.join(CORE_DATA_DIR, f'{pid}-{vid}b', tool, 'commits.pkl')
    com_df = pd.read_pickle(commit_path)

    # Drop duplicates
    com_df.drop_duplicates(subset=['src_path', 'begin_line', 'end_line'], inplace=True)
    range_dict = dict()

    for _, row in com_df.iterrows():
        src_path = row['src_path']
        range_val = range_dict.get(src_path, [])
        range_val.append((row['begin_line'], row['end_line']))
        range_dict[src_path] = range_val

    return range_dict
    
# Parse the diff text
# Return format : [[commit, before_src_path, after_src_path, line, content]]
def parse_diff(diff_txt, diff_commit, src_path):
    # Regex to find info
    commit_regex = r'commit (\w{40})'
    file_path_regex = r'^diff --git a/(.*) b/(.*)$'
    diff_block_regex = r'@@ -(\d+),?\d* \+(\d+),?\d* @@'

    commit_hash = None
    old_file_path = None
    new_file_path = None
    cur_old_line = None
    new_old_line = None

    #commit_set = set() # Check multiple commits in one command
    diff_lines = diff_txt.splitlines()

    for line in diff_lines:
        # Match commit info
        commit_match = re.match(commit_regex, line)
        if commit_match:
            commit = commit_match.group(1)
            commit_hash = commit[:7]
            diff_commit.add_commit(commit_hash, src_path)
            #print(f'Adding commit {commit_hash}:{src_path}')

            old_file_path = None
            new_file_path = None
            cur_old_line = None
            new_old_line = None
            continue

        # Commit has to be identified
        if commit is None:
            continue
        
        # Match file paths
        file_match = re.match(file_path_regex, line)
        if file_match:
            old_file_path = file_match.group(1)
            new_file_path = file_match.group(2)
            cur_old_line = None
            new_old_line = None

            # Consider only java files
            if old_file_path.endswith('.java') and new_file_path.endswith('.java'):
                diff_commit.add_file_info(commit_hash, src_path, old_file_path, new_file_path)
            else:
                old_file_path = None
                new_file_path = None
            continue
        
        # Ignore non-java file
        if old_file_path is None or new_file_path is None:
            continue

        # Match line numbers
        line_match = re.match(diff_block_regex, line)
        if line_match:
            cur_old_line = int(line_match.group(1))
            cur_new_line = int(line_match.group(2))
            continue
        
        # Ignore meaningless line (Ex index...)
        if cur_old_line is None or cur_new_line is None:
            continue
                
        # Deleted line
        if line.startswith('-'):
            if line.startswith('---'):
                continue
    
            line = line[1:].strip()

            diff_commit.add_diff(commit_hash, src_path, old_file_path, new_file_path, cur_old_line, line, 'del')
            cur_old_line += 1
            
        # Added line
        elif line.startswith('+'):
            if line.startswith('+++'):
                continue
            
            line = line[1:].strip()

            diff_commit.add_diff(commit_hash, src_path, old_file_path, new_file_path, cur_new_line, line, 'add')
            cur_new_line += 1
                
        # Unchanged line
        else:
            cur_old_line += 1
            cur_new_line += 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute commit scores")
    parser.add_argument('--project', '-p', type=str, default="Closure",
        help="Target project name (default: Closure)")
    parser.add_argument('--version', '-v', type=int, default=21,
        help="Target buggy version (default: 21)")
    args = parser.parse_args()

    print('Working on {}_{}b'.format(args.project, args.version))
    with open('/root/workspace/eror.txt', 'a') as file:
        file.write('Working on {}_{}b\n'.format(args.project, args.version))

    range_dict = get_range_dict(args.project, args.version)
    COMMIT_LOG_CMD = 'git log -M -C -L {0},{1}:{2}'
    diff_commit = Diff_commit()

    # For each change info, run git log and parse the result
    for src_path, ranges in range_dict.items():
        for begin_line, end_line in ranges:
            #print(f'Parsing {src_path}:{begin_line},{end_line}\n')
            cmd = COMMIT_LOG_CMD.format(begin_line, end_line, src_path)
            p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
            stdout, _ = p.communicate()

            try:
                parse_diff(stdout.decode(encoding='utf-8', errors='ignore'), diff_commit, src_path)
            except UnicodeDecodeError as e:
                print(cmd)
                raise e
        
    # Save the parsed result
    savedir = f'/root/workspace/data/Defects4J/diff/{args.project}-{args.version}b'
    os.makedirs(savedir, exist_ok=True)

    with open(os.path.join(savedir, 'diff.pkl'), 'wb') as file:
        pickle.dump(diff_commit, file)