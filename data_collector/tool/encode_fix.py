import os, json, argparse, pickle, sys, itertools, math
import pandas as pd
from tqdm import tqdm

sys.path.append('/root/workspace/data_collector/lib/')
from encoder import *
from utils import *

DIFF_DATA_DIR = '/root/workspace/data/Defects4J/diff'
CORE_DATA_DIR = '/root/workspace/data/Defects4J/core'
BASE_DATA_DIR = '/root/workspace/data/Defects4J/baseline'

adddel_list = ['add', 'del', 'all_uni', 'all_sep']
    
def main(pid, vid):
    log('encode_fix', f'Working on {pid}_{vid}b')

    diff_data_dir = os.path.join(DIFF_DATA_DIR, f'{pid}-{vid}b')

    with open(os.path.join(diff_data_dir, 'feature.pkl'), 'rb') as file:
        feature_dict = pickle.load(file)
    with open(os.path.join(diff_data_dir, 'encoder.pkl'), 'rb') as file:
        encoder_dict = pickle.load(file)

    commit_msg_dict = dict() 
    commit_set = set()
    base_data_dir = os.path.join(BASE_DATA_DIR, f'{pid}-{vid}b', 'commits')

    for setting_dict in feature_dict.values():
        for commit_dict in setting_dict.values():
            for commit in commit_dict.keys():
                commit_set.add(commit)


    for commit in commit_set:
        for filename in os.listdir(base_data_dir):
            if filename.startswith(f'c_{commit}'):
                with open(os.path.join(base_data_dir, filename), "r") as file:
                    data = json.load(file)
                    
                commit_msg_dict[commit] = data['log']
                break

        if commit not in commit_msg_dict:    
            log('encode_fix', f'[WARNING] No commit message for {commit}')
    
    for stage2, setting_dict in feature_dict.items():
        for setting, commit_dict in setting_dict.items():
            encoder_setting = dict(setting)
            del encoder_setting['adddel']
            
            if encoder_setting['diff_tool'] is None: # dfa
                encoder = encoder_dict[stage2][frozenset({'tracker' : encoder_setting['tracker'], 'diff_tool' : 'base', 'diff_type' : 'base'}.items())]
            else:
                encoder = encoder_dict[stage2][frozenset(encoder_setting.items())]
            
            for commit, aaa in commit_dict.items():
                id_vec, non_id_vec = encoder.encode([commit_msg_dict[commit]], update_vocab=False, mode='text')
                feature_dict[stage2][setting][commit]['msg'] = {'id' : id_vec, 'non_id' : non_id_vec}
    
    with open(os.path.join(diff_data_dir, 'feature.pkl'), 'wb') as file:
        pickle.dump(feature_dict, file)