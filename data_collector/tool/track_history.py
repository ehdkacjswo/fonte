import io
import json
import logging
import os
import re
import argparse
import subprocess
import sys
import pickle
import time

import pandas as pd

sys.path.append('/root/workspace/data_collector/lib/')
from utils import log, time_to_str, CustomInterval

CORE_DATA_DIR = '/root/workspace/data/Defects4J/core/'

commit_regex = r'commit (\w{40})'
src_path_regex = r'(?P<adddel>[+-]{3}) (?:/dev/null|[ab]/(?P<src_path>\S+))'
diff_block_regex = r'@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@'

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

        self.commit_hash, self.before_src_path, self.path_tup, self.after_src_path = None, None, None, None

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
        self.before_src_path, self.path_tup, self.before_line, self.after_line = None, None, None, None

        # Add commit
        if self.commit_hash not in self.intvl_dict:
            self.intvl_dict[self.commit_hash] = dict()
            
        return True
    
    # Set path info (only java file allowed)
    def set_src_path(self, line):
        if self.commit_hash is None: # commit has to be defined
            return True
        
        path_match = re.match(src_path_regex, line) # Match path info

        if path_match:
            adddel = path_match.group('adddel') # '---' or '+++'
            src_path = path_match.group('src_path') if path_match.group('src_path') else '/dev/null'

            # File on parent commit
            if adddel == '---':
                self.path_tup = None

                # Allow java file or /dev/null only
                if src_path.endswith('.java') or src_path == '/dev/null':
                    self.before_src_path = src_path
                else:
                    self.before_src_path = None
            
            # File on current commit
            elif adddel == '+++':
                if self.before_src_path is None: # Before src path has to be defined
                    self.path_tup = None
                    return True

                # Allow java file or /dev/null only
                if src_path.endswith('.java') or src_path == '/dev/null':
                    self.path_tup = (self.before_src_path, src_path)

                    if self.path_tup not in self.intvl_dict[self.commit_hash]:
                        self.intvl_dict[self.commit_hash].setdefault(self.path_tup, \
                            {'addition' : {'method_track' : CustomInterval(), 'line_diff' : CustomInterval()}, \
                            'deletion' : {'method_track' : CustomInterval(), 'line_diff' : CustomInterval()}})

                else:
                    self.before_src_path, self.path_tup = None, None
                
            return True
        
        else:
            return False
    
    # Set line info
    def set_line_num(self, line):
        if self.path_tup is None: # Before/after path has to be defined
            return True
        
        # Match line info (Zero based)
        line_match = re.match(diff_block_regex, line)

        if line_match:
            self.before_line = int(line_match.group(1)) - 1
            self.num_before_line = 0 if line_match.group(2) is None else int(line_match.group(2))
            self.after_line = int(line_match.group(3)) - 1
            self.num_after_line = 0 if line_match.group(4) is None else int(line_match.group(4))

            # Add method track info
            if self.num_before_line > 0:
                self.intvl_dict[self.commit_hash][self.path_tup]['deletion']['method_track'] |= CustomInterval(self.before_line, self.before_line + self.num_before_line - 1)

            if self.num_after_line > 0:
                self.intvl_dict[self.commit_hash][self.path_tup]['addition']['method_track'] |= CustomInterval(self.after_line, self.after_line + self.num_after_line - 1)
        
        else:
            return False
    
    # Get actual diff
    def parse_diff_line(self, line):
        if self.before_line is None or self.after_line is None: # Line info has to be defined
            return
                
        # Deleted line
        if line.startswith('-'):
            self.intvl_dict[self.commit_hash][self.path_tup]['deletion']['line_diff'] |= CustomInterval(self.before_line)
            self.before_line += 1
            
        # Added line
        elif line.startswith('+'):
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
    log('track_history', f'Working on {pid}_{vid}b'.format(pid, vid))
    savedir = f'/root/workspace/data/Defects4J/diff/{pid}-{vid}b'

    #if os.path.isdir(savedir):
    #    return
    
    # Checkout Defects4J project
    p = subprocess.Popen(['sh', '/root/workspace/lib/checkout.sh', pid, vid], \
        stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out_txt, err_txt = p.communicate()

    if p.returncode != 0:
        log('track_history', '[ERROR] Checkout failed', out_txt, err_txt)
        return
    
    # Change working directory to target Defects4J project
    try:
        os.chdir(f'/tmp/{pid}-{vid}b/')
    except:
        log('track_history', '[ERROR] Moving directory failed')
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
                        log('track_history', '[ERROR] git log failed', out_txt, err_txt)
                        continue

                    #print(stdout.decode(encoding='utf-8', errors='ignore'))
                    res_dict[setting][src_path][(begin_line, end_line)] = git_parser.parse_diff(stdout.decode(encoding='utf-8', errors='ignore'))
            
            # Check elapsed time
            end_time = time.time()
            log('track_history', f'({tracker}) : {time_to_str(start_time, end_time)}')
            
    # Save the parsed result
    savedir = f'/root/workspace/data/Defects4J/diff/{pid}-{vid}b'
    os.makedirs(savedir, exist_ok=True)

    #print(res_dict)

    with open(os.path.join(savedir, 'track_intvl.pkl'), 'wb') as file:
        pickle.dump(res_dict, file)
