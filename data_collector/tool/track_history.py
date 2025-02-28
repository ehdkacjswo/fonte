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
import time

import pandas as pd
from interval import interval

sys.path.append('/root/workspace/data_collector/lib/')
from gumtree import *

CORE_DATA_DIR = '/root/workspace/data/Defects4J/core/'
commit_regex = r'commit (\w{40})'
file_path_regex = r'^diff --git a/(.*) b/(.*)$'
diff_block_regex = r'@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@'

def time_to_str(start_time, end_time):
    hour, remainder = divmod(int(end_time - start_time), 3600)
    minute, second = divmod(remainder, 60)
    ms = int((end_time - start_time) * 1000) % 1000

    return f'{hour}h {minute}m {second}s {ms}ms'

def log(txt, out_txt=None, err_txt=None):
    with open('/root/workspace/data_collector/log/track_history.log', 'a') as file:
        file.write(txt + '\n')

        if out_txt is not None:
            file.write('[ERROR] OUT\n' + out_txt.decode(encoding='utf-8', errors='ignore') + '\n')
        
        if err_txt is not None:
            file.write('[ERROR] ERR\n' + err_txt.decode(encoding='utf-8', errors='ignore') + '\n')

# Get line range of suspicious methods
# Return = {src_path : set(line_start, line_end)}
# Contains data for every addition/deletion + method_track/line_diff even if they are empty (+ No file)
def get_suspicious_methods(pid, vid, tool='git'):
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

class GitParser:
    def reset(self):
        # Line range interval of tracked method
        # {commit : {(before, after src path) : before, after tracked line interval} }
        self.intvl_dict = dict()

        # Line interval of modified lines
        # {commit : {(before, after src path) : before, after modified line interval} }
        self.diff_intvl = dict() # line > code

        self.commit_hash = None
        self.path_tup = None
        self.before_line = None

    def __init__(self):
        self.reset()
    
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
        if self.commit_hash not in self.intvl_dict:
            self.intvl_dict[self.commit_hash] = dict()
            
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

        if self.path_tup not in self.intvl_dict[self.commit_hash]:
            self.intvl_dict[self.commit_hash][self.path_tup] = \
                {'addition' : {'method_track' : CustomInterval(), 'line_diff' : CustomInterval()}, 'deletion' : {'method_track' : CustomInterval(), 'line_diff' : CustomInterval()}}
            
        return True
    
    # Set line info
    def set_line_num(self, line):
        if self.path_tup is None: # Before/after path has to be defined
            return True
        
        # Match line info (Zero based)
        line_match = re.match(diff_block_regex, line)
        if line_match:
            self.before_line = int(line_match.group(1)) - 1
            self.num_before_line = 1 if line_match.group(2) is None else int(line_match.group(2))
            self.after_line = int(line_match.group(3)) - 1
            self.num_after_line = 1 if line_match.group(4) is None else int(line_match.group(4))
        else:
            return False

        # Add range info
        if self.num_before_line > 0:
            self.intvl_dict[self.commit_hash][self.path_tup]['deletion']['method_track'] |= CustomInterval(self.before_line, self.before_line + self.num_before_line - 1)

        if self.num_after_line > 0:
            self.intvl_dict[self.commit_hash][self.path_tup]['addition']['method_track'] |= CustomInterval(self.after_line, self.after_line + self.num_after_line - 1)
        
        return True
    
    # Get actual diff
    def parse_diff_line(self, line):
        if self.before_line is None: # Line info has to be defined
            return
                
        # Deleted line
        if line.startswith('-'):
            if line.startswith('---'): # Deleted file path (Need extra handling?)
                return

            self.intvl_dict[self.commit_hash][self.path_tup]['deletion']['line_diff'] |= CustomInterval(self.before_line)
            self.before_line += 1
            
        # Added line
        elif line.startswith('+'):
            if line.startswith('+++'): # Added file path (Need extra handling?)
                return

            self.intvl_dict[self.commit_hash][self.path_tup]['addition']['line_diff'] |= CustomInterval(self.after_line)
            self.after_line += 1
                
        # Unchanged line
        else:
            self.before_line += 1
            self.after_line += 1
            
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

            self.parse_diff_line(line) # Match actual diff data info

        return self.intvl_dict

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
    
    # Change working directory to target Defects4J project
    try:
        os.chdir(f'/tmp/{pid}-{vid}b/')
    except:
        log('[ERROR] Moving directory failed')
        return

    #
    tracker_list = ['git']
    res_dict = dict()

    for tracker in tracker_list:
        setting = frozenset({'tracker' : tracker}.items())
        res_dict[setting] = dict()

        if tracker == 'git':
            start_time = time.time()

            git_parser = GitParser()
            suspicious_methods = get_suspicious_methods(pid, vid)

            # For each change info, run git log and parse the result
            # {setting : {src_path : {(begin_line, end_line) : {commit : {path_tup : {adddel : {intvl_type : interval} } } } } } }
            for src_path, ranges in suspicious_methods.items():
                res_dict[setting][src_path] = dict()

                for begin_line, end_line in ranges:
                    p = subprocess.Popen(['git', 'log', '-M', '-C', '-L', f'{begin_line},{end_line}:{src_path}'], \
                        stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                    stdout, stderr = p.communicate()

                    if p.returncode != 0:
                        log('[ERROR] git log failed', out_txt, err_txt)
                        continue

                    res_dict[setting][src_path][(begin_line, end_line)] = git_parser.parse_diff(stdout.decode(encoding='utf-8', errors='ignore'))
            
            # Check elapsed time
            end_time = time.time()
            log(f'({tracker}) : {time_to_str(start_time, end_time)}')
            
    # Save the parsed result
    savedir = f'/root/workspace/data/Defects4J/diff/{pid}-{vid}b'
    os.makedirs(savedir, exist_ok=True)

    with open(os.path.join(savedir, 'track_intvl.pkl'), 'wb') as file:
        pickle.dump(res_dict, file)
