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
import math

import pandas as pd
from interval import inf

sys.path.append('/root/workspace/data_collector/lib/')
from gumtree import *
from utils import *
from encoder import keywords

CORE_DATA_DIR = '/root/workspace/data/Defects4J/core/'
DIFF_DATA_DIR = '/root/workspace/data/Defects4J/diff/'

# String : " + (Escape sequence + Any string) + "
# (https://docs.oracle.com/javase/specs/jls/se8/html/jls-3.html#jls-3.10.5)

# Character : ' + (Escape sequence | One character) + '
# (https://docs.oracle.com/javase/specs/jls/se8/html/jls-3.html#jls-3.10.4)

# Line comments : // + (Ends with newline)
# Block comments/Javadoc : /* + (Any string) + */
# (https://docs.oracle.com/javase/specs/jls/se8/html/jls-3.html#jls-3.7)

# Annotation : @ + (Possible whitespace) + Identifier name
# (https://docs.oracle.com/javase/specs/jls/se8/html/jls-9.html#jls-9.7)

non_id_regex = re.compile(r'''
    ("(?:\\.|[^"\\])*")            # Double-quoted string
    | ('(?:\\.|[^'\\])')           # Single character literals
    | (//.*?$)                     # Line comment
    | (/\*[\s\S]*?\*/)             # Block comment / Javadoc
    | (@\s*[A-Za-z_$][A-Za-z0-9_$]*) # Annotation
''', re.MULTILINE | re.VERBOSE)

# Identifier : Alphabet + Decimal number + _ + $ (Number not allowed at the start)
# (https://docs.oracle.com/javase/specs/jls/se8/html/jls-3.html#jls-3.8)

id_regex = re.compile(r'(?<!\w)[A-Za-z_$][A-Za-z0-9_$]*(?!\w)')

# Extract identifiers with GumTree
def extarct_id(res_dict):
    for setting, commit_dict in res_dict.items():
        log('greedy_id', f'[INFO] ID extraction (Tracker : {dict(setting)["tracker"]})')
        start_time = time.time()

        for commit, adddel_dict in commit_dict.items():
            for adddel, path_dict in adddel_dict.items():
                target_commit = commit + ('' if adddel == 'addition' else '~1')

                for src_path in path_dict.keys():
                    code_txt = get_src_from_commit(target_commit, src_path)

                    # Failed to get the target code
                    if code_txt is None:
                        log('greedy_id', f'[ERROR] Failed to copy file {target_commit}:{src_path}')
                        return False
                    
                    # Parse the code to get identifiers
                    else:
                        # Extract interval of string, character, line/block comment, Javadoc, annotation
                        non_id_intvl = CustomInterval()

                        for match in non_id_regex.finditer(code_txt):
                            non_id_intvl |= CustomInterval(match.start(), match.end() - 1)

                        # Extract all potential identifiers
                        id_intvl = CustomInterval()

                        for match in id_regex.finditer(code_txt):
                            sub_intvl = CustomInterval(match.start(), match.end() - 1)

                            # Does not overlap with non_id interval
                            if (non_id_intvl & sub_intvl).is_empty():
                                # Ignore Java keywords
                                if match.group(0) in keywords:
                                    continue
                                
                                id_intvl |= sub_intvl
                        
                        path_dict[src_path] = {'non_id' : non_id_intvl, 'id' : id_intvl}
        
        end_time = time.time()
        log('greedy_id', f'[INFO] Elapsed time : {time_to_str(start_time, end_time)}')
    
    return True
            
def main(pid, vid):
    log('greedy_id', f'Working on {pid}_{vid}b'.format(pid, vid))
    savedir = f'/root/workspace/data/Defects4J/diff/{pid}-{vid}b'
    
    # Checkout Defects4J project
    p = subprocess.Popen(['sh', '/root/workspace/lib/checkout.sh', pid, vid], \
        stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out_txt, err_txt = p.communicate()

    if p.returncode != 0:
        log('greedy_id', '[ERROR] Checkout failed', out_txt, err_txt)
        return
    
    # Change working directory to target Defects4J project
    try:
        os.chdir(f'/tmp/{pid}-{vid}b/')
    except:
        log('greedy_id', '[ERROR] Moving directory failed')
        return

    # 
    """intvl_path = os.path.join()
    if os.path.isfile(save_path):
        with open(save_path, 'rb') as file:
            res_dict = pickle.load(file)
    
    else:
        res_dict = dict()"""
    
    with open(os.path.join(DIFF_DATA_DIR, f'{pid}-{vid}b', 'track_intvl.pkl'), 'rb') as file:
        track_intvl = pickle.load(file)

    # Initialize dictionary
    res_dict = dict()

    for setting, bug_src_dict in track_intvl.items():
        res_dict[setting] = dict()

        for bug_intvl_dict in bug_src_dict.values():
            for commit_dict in bug_intvl_dict.values():
                for commit, path_dict in commit_dict.items():
                    res_dict[setting].setdefault(commit, {'addition' : dict(), 'deletion' : dict()})
                    
                    for path_tup, adddel_dict in path_dict.items():
                        for ind, adddel in enumerate(['deletion', 'addition']):
                            if path_tup[ind] != '/dev/null': # Ignore /dev/null
                                res_dict[setting][commit][adddel].setdefault(path_tup[ind], None)
    
    # Id extraction failed
    if not extarct_id(res_dict):
        return
        
    with open(os.path.join(DIFF_DATA_DIR, f'{pid}-{vid}b', 'greedy_intvl.pkl'), 'wb') as file:
        pickle.dump(res_dict, file)
