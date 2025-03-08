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
from interval import inf

sys.path.append('/root/workspace/data_collector/lib/')
from gumtree import *
from utils import *

CORE_DATA_DIR = '/root/workspace/data/Defects4J/core/'
DIFF_DATA_DIR = '/root/workspace/data/Defects4J/diff/'

# Extract identifiers with GumTree
def extarct_id(char_id_dict):
    for setting, commit_dict in char_id_dict.items():
        start_time = time.time()

        for commit, adddel_dict in commit_dict.items():
            for adddel, path_dict in adddel_dict.items():
                target_commit = commit + ('' if adddel == 'addition' else '~1')

                for src_path in path_dict.keys():
                    code_txt = get_src_from_commit(target_commit, src_path)

                    # Failed to get the target code
                    if code_txt is None:
                        log('parse_gumtree', f'[ERROR] Failed to copy file {target_commit}:{src_path}')
                        return False
                    
                    # Parse the code to get identifiers
                    else:
                        with open('/root/workspace/tmp/tmp.java', 'w') as file:
                            file.write(code_txt)
                        
                        id_dict = gumtree_parse('tmp.java')

                        # Failed to parse the file
                        if id_dict is None:
                            log('parse_gumtree', f'[ERROR] Failed to parse file {target_commit}:{src_path}')
                            return False

                        else:    
                            path_dict[src_path] = id_dict
        
        end_time = time.time()
        log('parse_gumtree', f'[INFO] ID extraction with {dict(setting)["tracker"]} : {time_to_str(start_time, end_time)}')
    
    return True

# Perform GumTree diff
def perform_diff(char_diff_dict):
    for setting, commit_dict in char_diff_dict.items():
        start_time = time.time()

        for commit, path_dict in commit_dict.items():
            for path_tup in path_dict.keys():

                # Both files are empty
                if path_tup[0] == '/dev/null' and path_tup[1] == '/dev/null':
                    log('parse_gumtree', f'[ERROR] Both files are empty on {commit}')
                    return False
                
                # One file is empty
                elif path_tup[0] == '/dev/null' or path_tup[1] == '/dev/null':
                    path_dict[path_tup] = {'addition' : CustomInterval(-inf, inf), 'deletion' : CustomInterval(-inf, inf)}
                
                # Both file exists
                else:
                    for ind, adddel in enumerate(['deletion', 'addition']):
                        target_commit = commit + ('' if ind == 1 else '~1')
                        code_txt = get_src_from_commit(target_commit, path_tup[ind])

                        # Faile to get target file
                        if code_txt is None:
                            log('parse_gumtree', f'[ERROR] Failed to copy file {target_commit}:{path_tup[ind]}')
                            return False
                        
                        else:
                            with open(f'/root/workspace/tmp/{adddel}.java', 'w') as file:
                                file.write(code_txt)
                    
                    # Perform GumTree diff
                    add_intvl, del_intvl = gumtree_diff(before_src_path='deletion.java', after_src_path='addition.java')

                    # Failed to perform GumTree diff
                    if add_intvl is None or del_intvl is None:
                        log('parse_gumtree', f'[ERROR] Failed to perfrom GumTree diff {commit}:{path_tup[0]}, {path_tup[1]}')
                        return False
                    
                    else:
                        path_dict[path_tup] = {'addition' : add_intvl, 'deletion' : del_intvl}
            
        end_time = time.time()
        log('parse_gumtree', f'[INFO] GumTree diff with tracker({dict(setting)["tracker"]}) : {time_to_str(start_time, end_time)}')

    return True
            
def main(pid, vid):
    log('parse_gumtree', f'Working on {pid}_{vid}b'.format(pid, vid))
    savedir = f'/root/workspace/data/Defects4J/diff/{pid}-{vid}b'
    
    # Checkout Defects4J project
    p = subprocess.Popen(['sh', '/root/workspace/lib/checkout.sh', pid, vid], \
        stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out_txt, err_txt = p.communicate()

    if p.returncode != 0:
        log('parse_gumtree', '[ERROR] Checkout failed', out_txt, err_txt)
        return
    
    # Change working directory to target Defects4J project
    try:
        os.chdir(f'/tmp/{pid}-{vid}b/')
    except:
        log('parse_gumtree', '[ERROR] Moving directory failed')
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
    # 'char_id' contains intervals of identifiers
    # 'char_diff' contains intervals of GumTree diff
    res_dict = {'char_id' : dict(), 'char_diff' : dict()}

    for setting, bug_src_dict in track_intvl.items():
        res_dict['char_id'][setting], res_dict['char_diff'][setting] = dict(), dict()

        for bug_intvl_dict in bug_src_dict.values():
            for commit_dict in bug_intvl_dict.values():
                for commit, path_dict in commit_dict.items():
                    res_dict['char_id'][setting].setdefault(commit, {'addition' : dict(), 'deletion' : dict()})
                    res_dict['char_diff'][setting].setdefault(commit, dict())
                    
                    for path_tup, adddel_dict in path_dict.items():
                        for ind, adddel in enumerate(['deletion', 'addition']):
                            if path_tup[ind] != '/dev/null': # Ignore /dev/null
                                res_dict['char_id'][setting][commit][adddel].setdefault(path_tup[ind], None)
                        
                        res_dict['char_diff'][setting][commit].setdefault(path_tup, None)
    
    # Id extraction failed
    if not extarct_id(res_dict['char_id']):
        return

    # GumTree diff failed
    if not perform_diff(res_dict['char_diff']):
        return
        
    with open(os.path.join(DIFF_DATA_DIR, f'{pid}-{vid}b', 'gumtree_intvl.pkl'), 'wb') as file:
        pickle.dump(res_dict, file)
