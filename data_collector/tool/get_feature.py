import os, sys, json, argparse, pickle, itertools
import numpy as np
import pandas as pd
from spiral import ronin
from collections import Counter
from tqdm import tqdm

sys.path.append('/root/workspace/data_collector/lib/')
from encoder import sum_encode

CORE_DATA_DIR = "/root/workspace/data/Defects4J/core"
BIC_GT_DIR = "/root/workspace/data/Defects4J/BIC_dataset"
BASELINE_DATA_DIR = "/root/workspace/data/Defects4J/baseline"
DIFF_DATA_DIR = '/root/workspace/data/Defects4J/diff'

encode_type_dict = {'file' : ['file'], 'git' : ['diff'], 'gumtree_base' : ['diff'], 'gumtree_class' : ['class', 'method', 'variable', 'comment']}
adddel_list = ['add', 'del', 'all-uni', 'all-sep']

def log(txt, out_txt=None, err_txt=None):
    with open('/root/workspace/data_collector/log/get_feature.log', 'a') as file:
        file.write(txt + '\n')

        if out_txt is not None:
            file.write('[ERROR] OUT\n' + out_txt.decode(encoding='utf-8', errors='ignore') + '\n')
        
        if err_txt is not None:
            file.write('[ERROR] ERR\n' + err_txt.decode(encoding='utf-8', errors='ignore') + '\n')

# data : message, src_path, diff
def get_feature(encode_data, setting_dict):
    diff_type = setting_dict['diff_type']

    res_dict = {adddel : dict() for adddel in adddel_list}
    encode_type_list = encode_type_dict[diff_type]

    # Iterate through commits
    for commit_hash, commit_data in encode_data.items():
        src_path_set = set()
        src_path_addition = list()
        src_path_deletion = list()
        src_path_all_uni = list()

        content_addition = dict()
        content_deletion = dict()
    
        # Addition data
        for src_path, encode_dict in commit_data["addition"].items():
            if src_path not in src_path_set: # Only unique path for all-uni
                src_path_set.add(src_path)
                src_path_all_uni = sum_encode(src_path_all_uni, encode_dict['src_path'])
            
            src_path_addition = sum_encode(src_path_addition, encode_dict['src_path'])
            
            for encode_type in encode_type_list:
                content_addition[encode_type] = \
                    sum_encode(content_addition.get(encode_type, []), encode_dict.get(encode_type, []))

        # Deletion data
        for src_path, encode_dict in commit_data["deletion"].items():
            if src_path not in src_path_set: # Only unique path for all-uni
                src_path_set.add(src_path)
                src_path_all_uni = sum_encode(src_path_all_uni, encode_dict['src_path'])
            
            src_path_deletion = sum_encode(src_path_deletion, encode_dict['src_path'])
            
            for encode_type in encode_type_list:
                content_deletion[encode_type] = sum_encode(content_deletion.get(encode_type, []), encode_dict.get(encode_type, []))
        
        for adddel in adddel_list:
            res_dict[adddel][commit_hash] = [commit_data.get("message", [])]

            if adddel == 'all-uni':
                src_path_encode = [src_path_all_uni]
                content_encode = {encode_type : \
                    [sum_encode(content_addition.get(encode_type, []), content_deletion.get(encode_type, []))] \
                    for encode_type in encode_type_list}
            
            elif adddel == 'all-sep':
                src_path_encode = [src_path_addition, src_path_deletion]
                content_encode = {encode_type : \
                    [content_addition.get(encode_type, []), content_deletion.get(encode_type, [])] \
                    for encode_type in encode_type_list}
            
            elif adddel == 'add':
                src_path_encode = [src_path_addition]
                content_encode = {encode_type : [content_addition.get(encode_type, [])] for encode_type in encode_type_list}
            
            else:
                src_path_encode = [src_path_deletion]
                content_encode = {encode_type : [content_deletion.get(encode_type, [])] for encode_type in encode_type_list}

            res_dict[adddel][commit_hash] += src_path_encode

            for encode_type in encode_type_list:
                res_dict[adddel][commit_hash] += content_encode[encode_type]
        
    return res_dict

def main(pid, vid):
    log(f'Working on project {pid}-{vid}b')
    
    diff_data_dir = os.path.join(DIFF_DATA_DIR, f'{pid}-{vid}b')
    os.makedirs(diff_data_dir, exist_ok=True)

    with open(os.path.join(diff_data_dir, 'encode.pkl'), 'rb') as file:
        encode_dict = pickle.load(file)
    
    # Load the previous result if possible
    feature_save_path = os.path.join(diff_data_dir, f'feature.pkl')

    """if os.path.isfile(feature_save_path):
        with open(feature_save_path, 'rb') as file:
            res_dict = pickle.load(file)
    
    else:
        res_dict = dict()"""
    
    res_dict = dict()

    for stage2, sub_dict in encode_dict.items():
        res_dict[stage2] = dict()

        for setting, encode_data in sub_dict.items():
            setting_dict = dict(setting)
            feature_dict = get_feature(encode_data=encode_data, setting_dict=setting_dict)

            for adddel, feature_data in feature_dict.items():
                new_setting = frozenset((setting_dict | {'adddel' : adddel}).items())
                res_dict[stage2][new_setting] = feature_data
    
    with open(os.path.join(diff_data_dir, 'feature.pkl'), 'wb') as file:
        pickle.dump(res_dict, file)
