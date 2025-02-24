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
from interval import interval

sys.path.append('/root/workspace/data_collector/lib/')
from gumtree import *

CORE_DATA_DIR = '/root/workspace/data/Defects4J/core/'
commit_regex = r'commit (\w{40})'
file_path_regex = r'^diff --git a/(.*) b/(.*)$'
diff_block_regex = r'@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@'

def log(txt, out_txt=None, err_txt=None):
    with open('/root/workspace/data_collector/log/collect_diff.log', 'a') as file:
        file.write(txt + '\n')

        if out_txt is not None:
            file.write('[ERROR] OUT\n' + out_txt.decode(encoding='utf-8', errors='ignore') + '\n')
        
        if err_txt is not None:
            file.write('[ERROR] ERR\n' + err_txt.decode(encoding='utf-8', errors='ignore') + '\n')

# Get line range of suspicious parts
# Return = {src_path : set(line_start, line_end)}
def get_diff_interval(pid, vid, tool='git'):
    commit_path = os.path.join(CORE_DATA_DIR, f'{pid}-{vid}b', tool, 'commits.pkl')
    com_df = pd.read_pickle(commit_path)

    # Drop duplicates
    com_df.drop_duplicates(subset=['src_path', 'begin_line', 'end_line'], inplace=True)
    diff_interval = dict()

    for _, row in com_df.iterrows():
        src_path = row['src_path']
        range_val = diff_interval.get(src_path, [])
        range_val.append((row['begin_line'], row['end_line']))
        diff_interval[src_path] = range_val

    return diff_interval

class Diff:
    def __init__(self):
        # Interval of lines modified
        # {commit : {(before, after src path) : before, after line interval} }
        self.diff_interval = dict()

        # Code data on given position
        # {commit : {file : {position : encoded content} } }
        self.git_diff = dict() # line > code
        self.gumtree_interval = dict() # pos > code

        self.commit_hash = None
        self.path_tup = None
        self.before_line = None
    
    # Set current commit
    def set_commit_hash(self, line):
        commit_match = re.match(commit_regex, line) # Match commit info
        if commit_match:
            commit = commit_match.group(1)
            self.commit_hash = commit[:7]
        else:
            return False

        # Initialization
        self.path_tup = None
        self.before_line = None

        # Add commit
        if self.commit_hash not in self.diff_interval:
            self.diff_interval[self.commit_hash] = dict()
            self.git_diff[self.commit_hash] = dict()
            self.gumtree_interval[self.commit_hash] = dict()
            
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

        # Initialization
        self.before_line = None

        # Allow only java/null file
        if not (before_src_path.endswith('.java') or before_src_path == '/dev/null') \
            or not (after_src_path.endswith('.java') or after_src_path == '/dev/null'):
            self.path_tup = None
            return True

        # Set before/after path info
        self.path_tup = (before_src_path, after_src_path)

        if self.path_tup not in self.diff_interval[self.commit_hash]:
            self.diff_interval[self.commit_hash][self.path_tup] = {'addition' : CustomInterval(), 'deletion' : CustomInterval()}

            self.git_diff[self.commit_hash][self.path_tup] = {'addition' : {}, 'deletion' : {}}
            self.gumtree_interval[self.commit_hash][self.path_tup] = {'addition' : CustomInterval(), 'deletion' : CustomInterval()}
            
        return True
    
    # Set line info
    def set_line_num(self, line):
        if self.path_tup is None: # Before/after path has to be defined
            return True
        
        # Match line info
        line_match = re.match(diff_block_regex, line)
        if line_match:
            self.before_line = int(line_match.group(1))
            self.num_before_line = 1 if line_match.group(2) is None else int(line_match.group(2))
            self.after_line = int(line_match.group(3))
            self.num_after_line = 1 if line_match.group(4) is None else int(line_match.group(4))
        else:
            return False

        # Add range info
        if self.num_before_line > 0:
            self.diff_interval[self.commit_hash][self.path_tup]['deletion'] |= CustomInterval(self.before_line - 1, self.before_line + self.num_before_line - 2)

        if self.num_after_line > 0:
            self.diff_interval[self.commit_hash][self.path_tup]['addition'] |= CustomInterval(self.after_line - 1, self.after_line + self.num_after_line - 2)
        
        return True
    
    # Get actual diff
    def get_git_diff(self, line):
        if self.before_line is None: # Line info has to be defined
            return
                
        # Deleted line
        if line.startswith('-'):
            if line.startswith('---'): # Deleted file path
                return
            
            # File was created but deletion occured (Need extra handling?)
            if self.path_tup[0] == '/dev/null':
                log(f'[ERROR] Deletion on empty file detected {commit_hash}:{before_src_path}')

            self.git_diff[self.commit_hash][self.path_tup]['deletion'][self.before_line] = line[1:].strip()
            self.before_line += 1
            
        # Added line
        elif line.startswith('+'):
            if line.startswith('+++'): # Added file path (Need extra handling?)
                return
            
            # File was deleted but addition occured
            if self.path_tup[1] == '/dev/null':
                log(f'[ERROR] Addition on empty file detected {commit_hash}:{after_src_path}')

            self.git_diff[self.commit_hash][self.path_tup]['addition'][self.after_line] = line[1:].strip()
            self.after_line += 1
                
        # Unchanged line
        else:
            self.before_line += 1
            self.after_line += 1
    
    # Perform gumtree parsing
    def parse_gumtree(self):
        for commit_hash in self.diff_interval.keys():
            for (before_src_path, after_src_path) in self.diff_interval[commit_hash].keys():
                addition_interval = self.diff_interval[commit_hash][(before_src_path, after_src_path)]['addition']
                deletion_interval = self.diff_interval[commit_hash][(before_src_path, after_src_path)]['deletion']

                if after_src_path == '/dev/null': # No after file
                    no_after_src = True
                else: # Copy after file
                    p = subprocess.Popen(['git', 'show', f'{commit_hash}:{after_src_path}'], \
                        stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                    after_code, err_txt = p.communicate()

                    # Error raised while copying file
                    if p.returncode != 0:
                        # Actual addition occured but failed to copy file
                        if len(self.git_diff[commit_hash][(before_src_path, after_src_path)]['addition']) > 0:
                            log(f'[ERROR] Failed to copy file {commit_hash}:{after_src_path}', after_code, err_txt)
                            return
                        
                        else:
                            no_after_src = True
                    
                    else:
                        after_code = after_code.decode(encoding='utf-8', errors='ignore')
                        with open('/root/workspace/tmp/after.java', 'w') as file:
                            file.write(after_code)
                        
                        no_after_src = False

                if before_src_path == '/dev/null': # No before file
                    no_before_src = True
                else: # Copy before file
                    p = subprocess.Popen(['git', 'show', f'{commit_hash}~1:{before_src_path}'], \
                        stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                    before_code, err_txt = p.communicate()
                    
                    # Error raised while copying file
                    if p.returncode != 0:
                        # Actual deletion occured but failed to copy file
                        if len(self.git_diff[commit_hash][(before_src_path, after_src_path)]['deletion']) > 0:
                            log(f'[ERROR] Failed to copy file {commit_hash}~1:{before_src_path}', before_code, err_txt)
                            return
                        
                        else:
                            no_before_src = True
                        
                    else:
                        before_code = before_code.decode(encoding='utf-8', errors='ignore')
                        with open('/root/workspace/tmp/before.java', 'w') as file:
                            file.write(before_code)
                        
                        no_before_src = False
                
                addition_token_interval, deletion_token_interval = gumtree_diff_token_range(\
                    no_after_src=no_after_src, no_before_src=no_before_src, \
                    addition_interval=addition_interval, deletion_interval=deletion_interval)
                
                if addition_token_interval is None or deletion_token_interval is None:
                    log(f'[ERROR] GumTree token interval retrieval failed')
                else:
                    self.gumtree_interval[commit_hash][(before_src_path, after_src_path)]['addition'] |= addition_token_interval
                    self.gumtree_interval[commit_hash][(before_src_path, after_src_path)]['deletion'] |= deletion_token_interval
                
    # Parse the diff text
    # Return format : [[commit, before_src_path, after_src_path, line, content]]
    def parse_diff(self, diff_txt):
        diff_lines = diff_txt.splitlines()

        for line in diff_lines:
            if self.set_commit_hash(line): # Match commit info
                continue
            
            if self.set_src_path(line): # Match source path info
                continue
            
            if self.set_line_num(line): # Match line number info
                continue

            self.get_git_diff(line) # Match actual diff data info
            
def main(pid, vid):
    log(f'Working on {pid}_{vid}b'.format(pid, vid))
    savedir = f'/root/workspace/data/Defects4J/diff/{pid}-{vid}b'

    #if os.path.isdir(savedir):
    #    return
    
    # Checkout Defects4J project
    p = subprocess.Popen(['sh', '/root/workspace/lib/checkout.sh', pid, vid], \
        stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out_txt, err_txt = p.communicate()

    if p.returncode != 0:
        log('[ERROR] Checkout failed', out_txt, err_txt)
        return
    
    # Change working directory
    try:
        os.chdir(f'/tmp/{pid}-{vid}b/')
    except:
        log('[ERROR] Moving directory failed')
        return

    diff_interval = get_diff_interval(pid, vid)
    COMMIT_LOG_CMD = 'git log -M -C -L {0},{1}:{2}'

    diff = Diff()

    # For each change info, run git log and parse the result
    for src_path, ranges in diff_interval.items():
        for begin_line, end_line in ranges:
            p = subprocess.Popen(['git', 'log', '-M', '-C', '-L', f'{begin_line},{end_line}:{src_path}'], \
                stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            stdout, stderr = p.communicate()

            if p.returncode != 0:
                log('[ERROR] git log failed', out_txt, err_txt)
                continue

            diff.parse_diff(stdout.decode(encoding='utf-8', errors='ignore'))

    diff.parse_gumtree()

    # Save the parsed result
    savedir = f'/root/workspace/data/Defects4J/diff/{pid}-{vid}b'
    os.makedirs(savedir, exist_ok=True)

    with open(os.path.join(savedir, 'git_diff.pkl'), 'wb') as file:
        pickle.dump(diff.git_diff, file)

    with open(os.path.join(savedir, 'gumtree_diff_interval.pkl'), 'wb') as file:
        pickle.dump(diff.gumtree_interval, file)
