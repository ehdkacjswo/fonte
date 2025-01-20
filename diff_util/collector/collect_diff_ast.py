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
import git # GitPython
from tabulate import tabulate
from nltk.corpus import stopwords

sys.path.append('/root/workspace/diff_util/lib/')
from gumtree import *
from encoder import *

CORE_DATA_DIR = '/root/workspace/data/Defects4J/core/'
commit_regex = r'commit (\w{40})'
file_path_regex = r'^diff --git a/(.*) b/(.*)$'
diff_block_regex = r'@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@'

# Get line range of suspicious parts
# Return = {src_path : set(line_start, line_end)}
def get_line_interval_dict(pid, vid, tool='git'):
    commit_path = os.path.join(CORE_DATA_DIR, f'{pid}-{vid}b', tool, 'commits.pkl')
    com_df = pd.read_pickle(commit_path)

    # Drop duplicates
    com_df.drop_duplicates(subset=['src_path', 'begin_line', 'end_line'], inplace=True)
    line_interval_dict = dict()

    for _, row in com_df.iterrows():
        src_path = row['src_path']
        range_val = line_interval_dict.get(src_path, [])
        range_val.append((row['begin_line'], row['end_line']))
        line_interval_dict[src_path] = range_val

    return line_interval_dict

class Diff:
    def __init__(self):
        # Interval of lines modified
        # {commit : {(before, after src path) : before, after line interval} }
        self.line_interval_dict = dict()

        # Code data on given position
        # {commit : {file : {position : encoded content} } }
        self.git_dict = dict() # line > code
        self.gumtree_dict = dict() # pos > code

        self.commit_hash = None
        self.path_tup = None
        self.cur_before_line = None

        # Encoder for git/gumtree diff
        self.git_encoder = Encoder()
    
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
        self.cur_before_line = None

        # Add commit
        if self.commit_hash not in self.line_interval_dict:
            self.line_interval_dict[self.commit_hash] = dict()
            self.git_dict[self.commit_hash] = dict()
            self.gumtree_dict[self.commit_hash] = dict()
            
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
        self.cur_before_line = None

        # Allow only java/null file
        if not (before_src_path.endswith('.java') or before_src_path == '/dev/null') \
            or not (after_src_path.endswith('.java') or after_src_path == '/dev/null'):
            self.path_tup = None
            return True

        # Set before/after path info
        self.path_tup = (before_src_path, after_src_path)

        if self.path_tup not in self.line_interval_dict[self.commit_hash]:
            self.line_interval_dict[self.commit_hash][self.path_tup] = {'addition' : interval(), 'deletion' : interval()}

            # Encode 
            #git_path_tup = (self.git_encoder.encode(before_src_path, use_stopword=True, update_vocab=True), \
            #    self.git_encoder.encode(after_src_path, use_stopword=True, update_vocab=True))

            self.git_dict[self.commit_hash][self.path_tup] = {'addition' : {}, 'deletion' : {}}
            self.gumtree_dict[self.commit_hash][self.path_tup] = {'addition' : interval(), 'deletion' : interval()}
            
        
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
            self.line_interval_dict[self.commit_hash][self.path_tup]['deletion'] |= interval[self.cur_before_line, self.cur_before_line + self.num_before_line - 1]

        if self.num_after_line > 0:
            self.line_interval_dict[self.commit_hash][self.path_tup]['addition'] |= interval[self.cur_after_line, self.cur_after_line + self.num_after_line - 1]
        
        return True
    
    # Get actual diff
    def get_git_diff(self, line):
        if self.cur_before_line is None: # Line info has to be defined
            return
                
        # Deleted line
        if line.startswith('-'):
            if line.startswith('---'): # Deleted file path
                return
            
            # File was created but deletion occured
            if self.path_tup[0] == '/dev/null':
                with open('/root/workspace/error.txt', 'a') as file:
                    file.write(f'Deletion on empty file detected {commit_hash}:{before_src_path}')
                return
    
            line = line[1:].strip()
            #encoded_line = self.git_encoder.encode(line, use_stopword=True, update_vocab=True)

            self.git_dict[self.commit_hash][self.path_tup]['deletion'][self.cur_before_line] = line
            self.cur_before_line += 1
            
        # Added line
        elif line.startswith('+'):
            if line.startswith('+++'): # Added file path
                return
            
            # File was deleted but addition occured
            if self.path_tup[1] == '/dev/null':
                with open('/root/workspace/error.txt', 'a') as file:
                    file.write(f'Addition on empty file detected {commit_hash}:{after_src_path}')
                return
            
            line = line[1:].strip()
            #encoded_line = self.git_encoder.encode(line, use_stopword=True, update_vocab=True)

            self.git_dict[self.commit_hash][self.path_tup]['addition'][self.cur_after_line] = line
            self.cur_after_line += 1
                
        # Unchanged line
        else:
            self.cur_before_line += 1
            self.cur_after_line += 1
        
        return
    
    # Perform gumtree parsing
    def parse_gumtree(self):
        for commit_hash in self.line_interval_dict.keys():
            for (before_src_path, after_src_path) in self.line_interval_dict[commit_hash].keys():
                #(before_src_path, after_src_path)
                addition_range = self.line_interval_dict[commit_hash][(before_src_path, after_src_path)]['addition']
                deletion_range = self.line_interval_dict[commit_hash][(before_src_path, after_src_path)]['deletion']

                if after_src_path == '/dev/null': # No after file
                    no_after_src = True
                else: # Copy after file
                    p = subprocess.Popen(f'git show {commit_hash}:{after_src_path}', shell=True, stdout=subprocess.PIPE)
                    after_code, _ = p.communicate()

                    # Error raised while copying file
                    if p.returncode != 0:
                        # Actual addition occured but failed to copy file
                        if len(self.git_dict[commit_hash][(before_src_path, after_src_path)]['addition']) > 0:
                            with open('/root/workspace/error.txt', 'a') as file:
                                file.write(f'Failed to copy file {commit_hash}:{after_src_path}')
                            continue
                        
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
                    p = subprocess.Popen(f'git show {commit_hash}~1:{before_src_path}', shell=True, stdout=subprocess.PIPE)
                    before_code, _ = p.communicate()
                    
                    # Error raised while copying file
                    if p.returncode != 0:
                        # Actual deletion occured but failed to copy file
                        if len(self.git_dict[commit_hash][(before_src_path, after_src_path)]['deletion']) > 0:
                            with open('/root/workspace/error.txt', 'a') as file:
                                file.write(f'Failed to copy file {commit_hash}~1:{before_src_path}')
                            continue
                        
                        else:
                            no_before_src = True
                        
                    else:
                        before_code = before_code.decode(encoding='utf-8', errors='ignore')
                        with open('/root/workspace/tmp/before.java', 'w') as file:
                            file.write(before_code)
                        
                        no_before_src = False
                
                #print(no_after_src, no_before_src, addition_range.interval_data, deletion_range.interval_data)
                #print(self.git_dict[commit_hash][(before_src_path, after_src_path)])
                addition_token_interval, deletion_token_interval = gumtree_diff_token_range(no_after_src=no_after_src, no_before_src=no_before_src, addition_range=addition_range, deletion_range=deletion_range)
                
                if addition_token_interval is None or deletion_token_interval is None:
                    with open('/root/workspace/error.txt', 'a') as file:
                        file.write(f'GumTree token interval retrieval failed')

                    return


                self.gumtree_dict[self.commit_hash][self.path_tup]['addition'] |= addition_token_interval
                self.gumtree_dict[self.commit_hash][self.path_tup]['deletion'] |= deletion_token_interval

                #print(self.commit_hash)
                #print(self.path_tup)
                #print(self.gumtree_dict[self.commit_hash][self.path_tup]['addition'].format("%+g"))
                #print(self.gumtree_dict[self.commit_hash][self.path_tup]['deletion'].format("%+g"))

                return
                
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
        
        self.parse_gumtree() # Perform gumtree parsing
            
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

    line_interval_dict = get_line_interval_dict(args.project, args.version)
    COMMIT_LOG_CMD = 'git log -M -C -L {0},{1}:{2}'

    diff = Diff()

    # For each change info, run git log and parse the result
    for src_path, ranges in line_interval_dict.items():
        for begin_line, end_line in ranges:
            cmd = COMMIT_LOG_CMD.format(begin_line, end_line, src_path)
            p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
            stdout, _ = p.communicate()

            try:
                diff.parse_diff(stdout.decode(encoding='utf-8', errors='ignore'))
            except UnicodeDecodeError as e:
                print(cmd)
                raise e
    
    #print(diff.line_interval_dict)
    #print(diff.git_dict)
    #print(diff.gumtree_dict)
        
    # Save the parsed result
    savedir = f'/root/workspace/data/Defects4J/diff/{args.project}-{args.version}b'
    os.makedirs(savedir, exist_ok=True)

    with open(os.path.join(savedir, 'git_diff.pkl'), 'wb') as file:
        pickle.dump(diff.git_dict, file)

    #with open(os.path.join(savedir, 'git_vocab.pkl'), 'wb') as file:
    #    pickle.dump(diff.git_encoder.vocab, file)

    with open(os.path.join(savedir, 'gumtree_diff_interval.pkl'), 'wb') as file:
        pickle.dump(diff.gumtree_dict, file)
