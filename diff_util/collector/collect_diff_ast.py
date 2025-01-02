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
import gumtree

CORE_DATA_DIR = '/root/workspace/data/Defects4J/core/'
commit_regex = r'commit (\w{40})'
file_path_regex = r'^diff --git a/(.*) b/(.*)$'
diff_block_regex = r'@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@'

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
        self.range_dict = dict()
        self.gumtree_dict = dict()
        self.git_dict = dict()
        
        self.diff_dict = diff_dict
        self.commit_hash = None
        self.path_tup = None
        self.cur_before_line = None
    
    # Set current commit
    def set_commit_hash(self, line):
        commit_match = re.match(commit_regex, line) # Match commit info
        if commit_match:
            commit = commit_match.group(1)
            self.commit_hash = commit[:7]
        else:
            return False

        # Initialize info
        self.path_tup = None
        self.cur_before_line = None

        # Add commit
        if self.commit_hash not in self.range_dict:
            self.range_dict[self.commit_hash] = dict()
            self.gumtree_dict[self.commit_hash] = dict()
            self.git_dict[self.commit_hash] = dict()
        
        return True
    
    # Set path info (only java file allowed)
    def set_src_path(self, line):
        if self.commit_hash is None: # commit has to be defined
            return True
        
        # Match file paths
        file_match = re.match(file_path_regex, line)
        if file_match:
            before_src_path = file_match.group(1)
            after_src_path = file_match.group(2)
        else:
            return False

        self.cur_before_line = None


        # Ignore non-java / empty file
        if not (before_src_path.endswith('.java') or before_src_path != '/dev/null') \
            or not (after_src_path.endswith('.java') or after_src_path != '/dev/null'):
            self.path_tup = None
            return True

        # Set before/after path info
        self.path_tup = (before_src_path, after_src_path)

        if self.path_tup not in self.range_dict[self.commit_hash]:
            self.range_dict[self.commit_hash][self.path_tup] = {'addition' : IntervalTree(), 'deletion' : IntervalTree()}
            self.gumtree_dict[self.commit_hash][self.path_tup] = {'addition' : dict(), 'deletion' : dict()}
            self.git_dict[self.commit_hash][self.path_tup] = {'addition' : dict(), 'deletion' : dict()}
        
        return True
    
    # Set line info
    def set_line_num(self, line):
        if self.path_tup is None: # Before/after path has to be defined
            return True
        
        # Match line info
        line_match = re.match(diff_block_regex, line)
        if line_match:
            self.cur_before_line = int(line_match.group(1))
            self.num_before_line = 1 if line_match.group(2) is None else int(line_match.group(2))
            self.cur_after_line = int(line_match.group(3))
            self.num_after_line = 1 if line_match.group(4) is None else int(line_match.group(4))
        else:
            return False

        # Add range info
        if self.num_before_line > 0:
            self.range_dict[self.commit_hash][self.path_tup]['deletion'].addi(self.cur_before_line, self.cur_before_line + self.num_before_line)

        if self.num_after_line > 0:
            self.range_dict[self.commit_hash][self.path_tup]['addition'].addi(self.cur_after_line, self.cur_after_line + self.num_after_line)
        
        return True
    
    def get_git_diff(self, line):
        if self.cur_before_line is None: # Line info has to be defined
            return
                
        # Deleted line
        if line.startswith('-'):
            if line.startswith('---'): # File path
                return
    
            line = line[1:].strip()

            self.git_dict[self.commit_hash][self.path_tup]['deletion'][self.cur_before_line] = line
            self.cur_before_line += 1
            
        # Added line
        elif line.startswith('+'):
            if line.startswith('+++'): # File path
                return
            
            line = line[1:].strip()

            self.git_dict[self.commit_hash][self.path_tup]['addition'][self.cur_after_line] = line
            self.cur_after_line += 1
                
        # Unchanged line
        else:
            self.cur_before_line += 1
            self.cur_after_line += 1
        
        return
    
    def parse_gumtree(self):
        for commit_hash in self.range_dict.keys():
            for (before_src_path, after_src_path) in self.range_dict[commit_hash].keys():
                addition_range = self.range_dict[commit_hash][(before_src_path, after_src_path)]['addition']
                deletion_range = self.range_dict[commit_hash][(before_src_path, after_src_path)]['deletion']

                # Copy after file
                if after_src_path == '/dev/null': # No after file
                    no_after_src = True
                else:
                    p = subprocess.Popen(f'git show {commit_hash}:{after_src_path}', shell=True, stdout=subprocess.PIPE)
                    after_code, _ = p.communicate()

                    # Error raised but 
                    if p.returncode != 0 and not addition_range.is_empty():
                        with open('/root/workspace/error.txt', 'a') as file:
                            file.write(f'Failed to copy file {commit_hash}:{after_src_path}')
                        continue
                    
                    after_code = after_code.decode(encoding='utf-8', errors='ignore')
                    with open('/root/workspace/tmp/after.java', 'w') as file:
                        file.write(after_code)
                    
                    no_after_src = False

                # Copy before file
                if before_src_path == '/dev/null': # No before file
                    no_before_src = True
                else:
                    p = subprocess.Popen(f'git show {commit_hash}~1:{before_src_path}', shell=True, stdout=subprocess.PIPE)
                    before_code, _ = p.communicate()
                    
                    if p.returncode != 0 and not deletion_range.is_empty():
                        with open('/root/workspace/error.txt', 'a') as file:
                            file.write(f'Failed to copy file {commit_hash}~1:{before_src_path}')
                        continue
                    
                    before_code = before_code.decode(encoding='utf-8', errors='ignore')
                    with open('/root/workspace/tmp/before.java', 'w') as file:
                        file.write(before_code)
                    
                    no_before_src = False
                
                gumtree.diff_json(no_after_src=no_after_src, no_before_src=no_before_src, addition_range=addition_range, deletion_range=deletion_range)
                

            
    # Parse the diff text
    # Return format : [[commit, before_src_path, after_src_path, line, content]]
    def parse_diff(self, diff_txt):
        diff_lines = diff_txt.splitlines()

        for line in diff_lines:
            if self.set_commit_hash(line): # Match commit info
                continue
            
            if self.set_src_path(line):
                continue
            
            if self.set_line_num(line):
                continue

            self.get_git_diff(line)
        
        self.parse_gumtree()
            
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

    range_dict = get_range_dict(args.project, args.version)
    COMMIT_LOG_CMD = 'git log -M -C -L {0},{1}:{2}'

    diff = Diff()

    # For each change info, run git log and parse the result
    for src_path, ranges in range_dict.items():
        for begin_line, end_line in ranges:
            cmd = COMMIT_LOG_CMD.format(begin_line, end_line, src_path)
            p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
            stdout, _ = p.communicate()

            try:
                diff.parse_diff(stdout.decode(encoding='utf-8', errors='ignore'))
            except UnicodeDecodeError as e:
                print(cmd)
                raise e
    
    print(diff.range_dict)
    #print(diff.git_dict)
        
    # Save the parsed result
    """savedir = f'/root/workspace/data/Defects4J/diff/{args.project}-{args.version}b'
    os.makedirs(savedir, exist_ok=True)

    with open(os.path.join(savedir, 'diff.pkl'), 'wb') as file:
        pickle.dump(diff_commit, file)"""