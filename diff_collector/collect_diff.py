import io
import json
import logging
import os
import re
import argparse
import subprocess
import sys

import pandas as pd
import git # GitPython

"""sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from diff_parser import Diff"""

# Parse the diff text
# Return format : [[is_addition, old_file_path, new_file_path, modified line number, modified content]]
def parse_diff(diff_txt):
    rows = []

    # Regex to find info
    file_path_regex = r'^diff --git a/(.*) b/(.*)$'
    diff_block_regex = r'@@ -(\d+),?\d* \+(\d+),?\d* @@'

    old_file_path = None
    new_file_path = None
    cur_old_line = None
    new_old_line = None

    diff_lines = diff_txt.splitlines()

    for line in diff_lines:
        # Match file paths
        file_match = re.match(file_path_regex, line)
        if file_match:
            old_file_path = file_match.group(1)
            new_file_path = file_match.group(2)
            cur_old_line = None
            new_old_line = None

            # Consider only java files
            if not old_file_path.endswith('.java') or not new_file_path.endswith('.java'):
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
            cur_old_line += 1
            line = line[1:].strip()

            # Ignore meaningless content and path info
            if not any(char.isalpha() or char.isdigit() for char in line) or line.startswith('---'):
                continue
            rows.append([False, old_file_path, new_file_path, cur_old_line - 1, line[1:]])
            
        # Added line
        elif line.startswith('+'):
            cur_new_line += 1
            line = line[1:].strip()

            # Ignore meaningless content and path info
            if not any(char.isalpha() or char.isdigit() for char in line) or line.startswith('+++'):
                continue
            rows.append([True, old_file_path, new_file_path, cur_new_line - 1, line])
                
        # Unchanged line
        else:
            cur_old_line += 1
            cur_new_line += 1
    
    return rows

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute commit scores")
    parser.add_argument('--project', '-p', type=str, default="Closure",
        help="Target project name (default: Closure)")
    parser.add_argument('--version', '-v', type=int, default=21,
        help="Target buggy version (default: 21)")
    args = parser.parse_args()

    print('Working on {}_{}b'.format(args.project, args.version))

    # Current working directory has to be the directory of corresponding argument
    # It's automatically set when it's called by collect_diff.sh
    repo = git.Repo.init('./')

    # Index of panda dataframe
    index = ['is_addition', 'old_file_path', 'new_file_path', 'line_num', 'content']
    output_dir = '/root/workspace/data/Defects4J/diff/{}-{}b/'.format(args.project, args.version)

    # Iterate through commits in reverse order
    cnt = 0
    for commit in repo.iter_commits('HEAD'):
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
                    file.write(f'{args.project}-{arg.version}) {commit.hexsha}...{parent}')