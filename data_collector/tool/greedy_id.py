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
from utils import *

CORE_DATA_DIR = '/root/workspace/data/Defects4J/core/'
DIFF_DATA_DIR = '/root/workspace/data/Defects4J/diff/'
            
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

    # Initialize
    # On track intvl, addition/deletion always exists
    # addition/deletion is added only when actual line diff is not empty
    res_dict = dict()

    for setting, bug_src_dict in track_intvl.items():
        res_dict[setting] = dict()

        for bug_intvl_dict in bug_src_dict.values():
            for commit_dict in bug_intvl_dict.values():
                for commit, path_dict in commit_dict.items():
                    res_dict[setting].setdefault(commit, {'addition' : dict(), 'deletion' : dict()})
                    
                    for path_tup, adddel_dict in path_dict.items():
                        for ind, src_path in enumerate(path_tup):
                            if src_path == '/dev/null':
                                continue

                            adddel = 'addition' if ind == 1 else 'deletion'
                            if adddel_dict[adddel]['method_track'].is_empty():
                                res_dict[setting][commit][adddel].setdefault(src_path, CustomInterval())
                                res_dict[setting][commit][adddel][src_path] |= adddel_dict[adddel]['line_diff']

    # Parse with GumTree
    # Add data for empty ca
    for setting, commit_dict in res_dict.items():
        tracker = dict(setting)['tracker']
        start_time = time.time()

        for commit, adddel_dict in commit_dict.items():
            for adddel, path_dict in adddel_dict.items():
                for src_path in path_dict.keys():
                    target_commit = commit + ('' if adddel == 'addtion' else '~1')
                    code_txt = get_src_from_commit(target_commit, src_path)
                    if code_txt is None:
                        log('greedy_id', f'[ERROR] Failed to get file {target_commit}:{src_path}')