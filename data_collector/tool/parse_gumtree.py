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
DIFF_DATA_DIR = '/root/workspace/data/Defects4J/diff/'

def log(txt, out_txt=None, err_txt=None):
    with open('/root/workspace/data_collector/log/parse_gumtree.log', 'a') as file:
        file.write(txt + '\n')

        if out_txt is not None:
            file.write('[ERROR] OUT\n' + out_txt.decode(encoding='utf-8', errors='ignore') + '\n')
        
        if err_txt is not None:
            file.write('[ERROR] ERR\n' + err_txt.decode(encoding='utf-8', errors='ignore') + '\n')

def time_to_str(start_time, end_time):
    hour, remainder = divmod(int(end_time - start_time), 3600)
    minute, second = divmod(remainder, 60)
    ms = int((end_time - start_time) * 1000) % 1000

    return f'{hour}h {minute}m {second}s {ms}ms'
            
def main(pid, vid):
    log(f'Working on {pid}_{vid}b'.format(pid, vid))
    savedir = f'/root/workspace/data/Defects4J/diff/{pid}-{vid}b'
    
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
    """intvl_path = os.path.join()
    if os.path.isfile(save_path):
        with open(save_path, 'rb') as file:
            res_dict = pickle.load(file)
    
    else:
        res_dict = dict()"""
    
    with open(os.path.join(DIFF_DATA_DIR, f'{pid}-{vid}b', 'track_intvl.pkl'), 'rb') as file:
        track_intvl = pickle.load(file)

    # Initialize
    # On track intvl, addition/deletion always exists
    # addition/deletion is added only when actual line diff is not empty
    res_dict = dict()

    for setting, tracker_dict in track_intvl.items():
        res_dict[setting] = dict()

        for src_dict in tracker_dict.values():
            for range_dict in src_dict.values():
                for commit, commit_dict in range_dict.items():
                    if commit not in res_dict[setting]:
                        res_dict[setting][commit] = dict()
                    
                    for path_tup, path_dict in commit_dict.items():
                        if path_tup not in res_dict[setting][commit]:
                            res_dict[setting][commit][path_tup] = dict()

                        for adddel, intvl_dict in path_dict.items():
                            intvl = intvl_dict['line_diff']

                            if not (adddel in res_dict[setting][commit][path_tup] or intvl.is_empty()):
                                res_dict[setting][commit][path_tup][adddel] = dict()
    
    # Parse with GumTree
    # Add data for empty ca
    for setting, tracker_dict in res_dict.items():
        tracker = dict(setting)['tracker']
        start_time = time.time()

        for commit_hash, commit_dict in tracker_dict.items():
            for (before_src_path, after_src_path), path_dict in commit_dict.items():

                if 'addition' in path_dict: # Copy after file
                    p = subprocess.Popen(['git', 'show', f'{commit_hash}:{after_src_path}'], \
                        stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                    after_code, err_txt = p.communicate()

                    # Error raised while copying file
                    if p.returncode != 0:
                        # Actual addition occured but failed to copy file
                        log(f'[ERROR] Failed to copy file {commit_hash}:{after_src_path}', after_code, err_txt)
                        continue
                    
                    else:
                        no_after_src = False

                        after_code = after_code.decode(encoding='utf-8', errors='ignore')
                        with open('/root/workspace/tmp/after.java', 'w') as file:
                            file.write(after_code)
                        
                        path_dict['addition']['char_id'] = gumtree_parse('after.java')
                else: # No after file
                    no_after_src = True

                if 'deletion' in path_dict: # Copy after file
                    p = subprocess.Popen(['git', 'show', f'{commit_hash}~1:{before_src_path}'], \
                        stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                    before_code, err_txt = p.communicate()
                    
                    # Error raised while copying file
                    if p.returncode != 0:
                        # Actual deletion occured but failed to copy file
                        log(f'[ERROR] Failed to copy file {commit_hash}~1:{before_src_path}', before_code, err_txt)
                        continue
                        
                    else:
                        no_before_src = False

                        before_code = before_code.decode(encoding='utf-8', errors='ignore')
                        with open('/root/workspace/tmp/before.java', 'w') as file:
                            file.write(before_code)
                        
                        path_dict['deletion']['char_id'] = gumtree_parse('before.java')
                else: # No before file
                    no_before_src = True
                
                # GumTree diff parsing
                add_intvl, del_intvl = gumtree_diff(no_after_src=no_after_src, no_before_src=no_before_src)
                
                if add_intvl is None or del_intvl is None:
                    log(f'[ERROR] GumTree token interval retrieval failed')
                else:
                    if not after_src_path:
                        path_dict['addition']['char_diff'] = add_intvl
                    if not before_src_path:
                        path_dict['deletion']['char_diff'] = del_intvl
            
        end_time = time.time()
        log(f'({tracker}) : {time_to_str(start_time, end_time)}')
        
    with open(os.path.join(DIFF_DATA_DIR, f'{pid}-{vid}b', 'gumtree_intvl.pkl'), 'wb') as file:
        pickle.dump(res_dict, file)

    """diff.parse_gumtree()

    # Save the parsed result
    savedir = f'/root/workspace/data/Defects4J/diff/{pid}-{vid}b'
    os.makedirs(savedir, exist_ok=True)

    with open(os.path.join(savedir, 'git_diff.pkl'), 'wb') as file:
        pickle.dump(diff.git_diff, file)

    with open(os.path.join(savedir, 'gumtree_diff_interval.pkl'), 'wb') as file:
        pickle.dump(diff.gumtree_interval, file)"""
