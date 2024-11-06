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

"""sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from diff_parser import Diff"""

CORE_DATA_DIR = '/root/workspace/data/Defects4J/core/'

# Return = {src_path : set(line_start, line_end)}
def get_range_dict(pid, vid, tool='git'):
    commit_path = os.path.join(CORE_DATA_DIR, f'{pid}-{vid}b', tool, 'commits.pkl')
    com_df = pd.read_pickle(commit_path)
    com_df.drop_duplicates(subset=['src_path', 'begin_line', 'end_line'], inplace=True)
    range_dict = dict()

    for _, row in com_df.iterrows():
        src_path = row['src_path']
        range_val = range_dict.get(src_path, [])
        range_val.append((row['begin_line'], row['end_line']))
        range_dict[src_path] = range_val

    return range_dict

class Diff_commit: # Class containg diff data of commit
    class Diff_src: # Class containg diff data of source file
        def __init__(self):
            self.diff_dict = dict()
        
        def add_file_info(self, before_src_path, after_src_path):
            file_info_key = (before_src_path, after_src_path)
            if (file_info_key) not in self.diff_dict:
                self.diff_dict[file_info_key] = [dict(), dict()] # Addition, Deletion {line : content}
            
        def add_diff(self, before_src_path, after_src_path, line, content, adddel='add'):
            file_info_key = (before_src_path, after_src_path)
            dict_idx = 0 if adddel == 'add' else 1 # Select addition, deletion
            if line in self.diff_dict[file_info_key][dict_idx]:
                if self.diff_dict[file_info_key][dict_idx][line] != content: # Same line, but different content
                    with open('/root/workspace/eror.txt', 'a') as file:
                        file.write(f'Different diff content for same line num: {before_src_path},{after_src_path}, {line}\n')
            else:
                self.diff_dict[file_info_key][dict_idx][line] = content
        
        def self_print(self):
            for (before_src_path, after_src_path) in self.diff_dict.keys():
                print(f'Before path : {before_src_path}, After path : {after_src_path}')

                addition = self.diff_dict[(before_src_path, after_src_path)][0]
                deletion = self.diff_dict[(before_src_path, after_src_path)][1]

                print('Addition)')
                for line, content in addition.items():
                    print(line, content)
                print('Deletion)')
                for line, content in deletion.items():
                    print(line, content)


    def __init__(self):
        self.diff_dict = dict()

    def add_commit(self, commit, src_path):
        if commit not in self.diff_dict:
            self.diff_dict[commit] = dict()

        if src_path not in self.diff_dict[commit]:
            self.diff_dict[commit][src_path] = self.Diff_src()
    
    def add_file_info(self, commit, src_path, before_src_path, after_src_path):
        self.diff_dict[commit][src_path].add_file_info(before_src_path, after_src_path)
    
    def add_diff(self, commit, src_path, before_src_path, after_src_path, line, content, adddel='add'):
        self.diff_dict[commit][src_path].add_diff(before_src_path, after_src_path, line, content, adddel)

    def self_print(self):
        for commit in self.diff_dict.keys():
            print(f'Commit : {commit}')

            for src_path in self.diff_dict[commit].keys():
                print(f'src_path : {src_path}')
                self.diff_dict[commit][src_path].self_print()
    
# Parse the diff text
# Return format : [[commit, before_src_path, after_src_path, line, content]]
def parse_diff(diff_txt, dif_commit, scr_path):
    rows = []

    # Regex to find info
    commit_regex = r'commit (\w{40})'
    file_path_regex = r'^diff --git a/(.*) b/(.*)$'
    diff_block_regex = r'@@ -(\d+),?\d* \+(\d+),?\d* @@'

    commit = None
    old_file_path = None
    new_file_path = None
    cur_old_line = None
    new_old_line = None

    commit_set = set() # Check multiple commits in one command
    diff_lines = diff_txt.splitlines()

    for line in diff_lines:
        # Match commit info
        commit_match = re.match(commit_regex, line)
        if commit_match:
            commit = commit_match.group(1)
            old_file_path = None
            new_file_path = None
            cur_old_line = None
            new_old_line = None
            if commit in commit_set: # Duplicate commit in on log cmd
                with open('/root/workspace/eror.txt', 'a') as file:
                    file.write(f'Double commit in one cmd: {commit}\n')
            else:
                commit_set.add(commit)
            diff_commit.add_commit(commit, src_path)
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
                diff_commit.add_file_info(commit, src_path, old_file_path, new_file_path)
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

            # Ignore meaningless content and path info
            """if not any(char.isalpha() or char.isdigit() for char in line) or line.startswith('---'):
                continue"""

            diff_commit.add_diff(commit, src_path, old_file_path, new_file_path, cur_old_line, line, 'del')
            cur_old_line += 1
            #rows.append([False, old_file_path, new_file_path, cur_old_line - 1, line[1:]])
            
        # Added line
        elif line.startswith('+'):
            if line.startswith('+++'):
                continue
            
            line = line[1:].strip()

            # Ignore meaningless content and path info
            """if not any(char.isalpha() or char.isdigit() for char in line) or line.startswith('+++'):
                continue"""
            diff_commit.add_diff(commit, src_path, old_file_path, new_file_path, cur_new_line, line, 'add')
            cur_new_line += 1
            #rows.append([True, old_file_path, new_file_path, cur_new_line - 1, line])
                
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
    os.makedirs(f'/root/workspace/data/Defects4J/diff/{args.project}-{args.version}b', exist_ok=True)

    for src_path, ranges in range_dict.items():
        for begin_line, end_line in ranges:
            cmd = COMMIT_LOG_CMD.format(begin_line, end_line, src_path)
            print(cmd)
            p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
            stdout, _ = p.communicate()
            try:
                parse_diff(stdout.decode(encoding='utf-8', errors='ignore'), diff_commit, src_path)
            except UnicodeDecodeError as e:
                with open('/root/workspace/eror.txt', 'a') as file:
                    file.write('Decoding error\n')
                raise e
    with open(os.path.join(f'/root/workspace/data/Defects4J/diff/{args.project}-{args.version}b', 'log.pkl'), 'wb') as file:
        pickle.dump(diff_commit, file)

    """cov_df = pd.read_pickle(f'/root/workspace/data/Defects4J/core/{args.project}-{args.version}b/git/commits.pkl')
    is_passing = cov_df["result"].values.astype(bool) # test results
    cov_df.drop("result", axis=1, inplace=True)
    covered_by_failure_only = True
    if covered_by_failure_only:
        cov_df = cov_df.loc[:, cov_df.loc[~is_passing].any(axis=0)]
    
    for a, b in cov_df.iterrows():
        print(b)

    # Print DataFrame as a formatted table
    print(tabulate(cov_df, headers='keys', tablefmt='grid'))"""

    """# Current working directory has to be the directory of corresponding argument
    # It's automatically set when it's called by collect_diff.sh
    repo = git.Repo.init('./')

    # Index of panda dataframe
    index = ['is_addition', 'old_file_path', 'new_file_path', 'line_num', 'content']
    output_dir = '/root/workspace/data/Defects4J/diff/{}-{}b/'.format(args.project, args.version)

    # Iterate through commits in reverse order
    cnt = 0
    commit = repo.head.commit
    commit_set = set()

    while commit:
        commit_set.add(commit.hexsha)
        if commit.parents:
            commit = commit.parents[0]
        else:
            commit = None
    
    commit_set2 = set()
    for filename in os.listdir(f'/root/workspace/data/Defects4J/baseline/{args.project}-{args.version}b/commits'):
        commit_set2.add(filename[2:-7])
    
    com_df = pd.read_pickle(f'/root/workspace/data/Defects4J/core/{args.project}-{args.version}b/git/commits.pkl')
    for _, row in com_df.iterrows():
        print(row)
    
    print(com_df.columns.values.tolist())"""
    #print(commit_set - commit_set2)
    #print(commit_set2 - commit_set)
    """for commit in repo.iter_commits('HEAD'):
        # Create target directory if there is none
        commit_dir = os.path.join(output_dir, commit.hexsha)
        os.makedirs(commit_dir, exist_ok=True)

        # Write message
        msg_path = os.path.join(commit_dir, 'message.txt')
        if not os.path.exists(msg_path):
            with open(os.path.join(commit_dir, 'message.txt'), 'w') as file:
                file.write(commit.message)

        # For initial commit, give empty repo as parent
        if len(commit.parents) == 0:
            parent = '4b825dc642cb6eb9a060e54bf8d69288fbee4904'
        # Else, select first parent
        else:
            parent = commit.parents[0]
        
        for parent in commit.parents:
            csv_path = os.path.join(commit_dir, f'{parent}.csv')
            if os.path.exists(csv_path):
                continue

            try:
                diff = repo.git.diff(parent, commit.hexsha)
                diff_rows = parse_diff(diff)
            
                diff_df = pd.DataFrame(diff_rows,columns=index)
                diff_df = diff_df.set_index(index)
                diff_df.to_csv(os.path.join(commit_dir, f'{parent}.csv'), errors='ignore')
            
            except:
                with open('/root/workspace/error_list.txt', 'a') as file:
                    file.write(f'{args.project}-{arg.version}) {commit.hexsha}...{parent}')"""