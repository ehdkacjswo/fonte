import io
import json
import logging
import os
import re
import argparse
import subprocess
import sys
import pickle
import tempfile

import pandas as pd
from intervaltree import IntervalTree
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

class Diff:
    def __init__(self, diff_dict=dict()):
        self.diff_dict = diff_dict
        self.cur_commit_hash = None
    
    # Set current commit hash
    def set_commit_hash(self, commit_hash):
        self.cur_commit_hash = commit_hash
        self.before_src_path = None
        self.after_src_path = None
        self.cur_before_line = None
        self.cur_after_line = None

        if commit_hash not in self.diff_dict:
            self.diff_dict[commit_hash] = [dict(), dict()]
    
    # Set path info (only java file allowed)
    def set_src_path(self, before_src_path, after_src_path):
        if before_src_path.endswith('.java') and after_src_path.endswith('.java'):
            self.before_src_path = before_src_path
            self.after_src_path = after_src_path
        else:
            self.before_src_path = None
            self.after_src_path = None

        self.cur_before_line = None
        self.cur_after_line = None
    
    # Set line info
    def set_line_num(self, cur_before_line, num_before_line, cur_after_line, num_after_line):
        self.cur_before_line = cur_before_line
        self.num_before_line = num_before_line
        self.cur_after_line = cur_after_line
        self.num_after_line = num_after_line

        # Add range info
        if num_before_line > 0:
            
            
    # Parse the diff text
    # Return format : [[commit, before_src_path, after_src_path, line, content]]
    def parse_diff(self, diff_txt):
        # Regex to find info
        commit_regex = r'commit (\w{40})'
        file_path_regex = r'^diff --git a/(.*) b/(.*)$'
        diff_block_regex = r'@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@'

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
                self.set_commit_hash(commit[:7])
                continue

            # Commit has to be identified
            if self.cur_commit_hash is None:
                continue
            
            # Match file paths
            file_match = re.match(file_path_regex, line)
            if file_match:
                before_src_path = file_match.group(1)
                after_src_path = file_match.group(2)
                self.set_src_path(before_src_path, after_src_path)
                continue
            
            # Ignore non-java file
            if self.before_src_path is None or self.after_src_path is None:
                continue

            # Match line numbers
            line_match = re.match(diff_block_regex, line)
            if line_match:
                cur_before_line = int(line_match.group(1))
                num_before_line = 1 if line_match.group(2) is None else int(line_match.group(2))
                cur_after_line = int(line_match.group(3))
                num_after_line = 1 if line_match.group(4) is None else int(line_match.group(4))
                self.set_line_num(cur_before_line, num_before_line, cur_after_line, num_after_line)
                continue
            
            # Ignore meaningless line (Ex index...)
            if cur_old_line is None or cur_new_line is None:
                continue
                    
            # Deleted line
            if line.startswith('-'):
                if line.startswith('---'):
                    continue
        
                line = line[1:].strip()

                #diff_commit.add_diff(commit_hash, src_path, old_file_path, new_file_path, cur_old_line, line, 'del')
                print('del')
                cur_old_line += 1
                
            # Added line
            elif line.startswith('+'):
                if line.startswith('+++'):
                    continue
                
                line = line[1:].strip()

                #diff_commit.add_diff(commit_hash, src_path, old_file_path, new_file_path, cur_new_line, line, 'add')
                print('add')
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

    # For each change info, run git log and parse the result
    for src_path, ranges in range_dict.items():
        for begin_line, end_line in ranges:
            cmd = COMMIT_LOG_CMD.format(begin_line, end_line, src_path)
            p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
            stdout, _ = p.communicate()

            try:
                parse_diff(stdout.decode(encoding='utf-8', errors='ignore'))
            except UnicodeDecodeError as e:
                print(cmd)
                raise e
        
    # Save the parsed result
    """savedir = f'/root/workspace/data/Defects4J/diff/{args.project}-{args.version}b'
    os.makedirs(savedir, exist_ok=True)

    with open(os.path.join(savedir, 'diff.pkl'), 'wb') as file:
        pickle.dump(diff_commit, file)"""