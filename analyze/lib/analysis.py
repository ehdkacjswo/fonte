import argparse, os, copy, itertools, pickle, sys, json, subprocess
import pandas as pd
from scipy.stats import wilcoxon
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import math

sys.path.append('/root/workspace/lib/')
from experiment_utils import load_BIC_GT

sys.path.append('/root/workspace/analyze/lib')
from result_gen import get_metric_dict

sys.path.append('/root/workspace/data_collector/lib/')
from encoder import *
from utils import CustomInterval

sys.path.append('/root/workspace/data_collector/tool/')
#from stage2 import get_style_change_data

RESULT_DATA_DIR = '/root/workspace/data/Defects4J/result'
CORE_DATA_DIR = '/root/workspace/data/Defects4J/core'
BASE_DATA_DIR = '/root/workspace/data/Defects4J/baseline'
DIFF_DATA_DIR = '/root/workspace/data/Defects4J/diff'

# Check how id filtering affects the document
def check_id_filter(stage2='precise', setting=frozenset({'tracker': 'git', 'diff_tool' : 'base', 'doc_level' : 'commit', 'adddel' : 'all_sep'}.items())):
    GT = load_BIC_GT("/root/workspace/data/Defects4J/BIC_dataset")
    res_dict = dict()

    org_setting = frozenset((dict(setting) | {'diff_type' : 'base'}).items())
    new_setting = frozenset((dict(setting) | {'diff_type' : 'id_all'}).items())
    adddel = dict(setting)['adddel']
    target_type_list = [('add_diff', 'add_id'), ('del_diff', 'del_id')] if adddel == 'all_sep' else [('diff', 'id')]

    # Iterate through projects
    for _, row in GT.iterrows():
        #pid, vid, BIC = row.pid, row.vid, row.commit
        pid = 'Closure'
        vid = '2'
        BIC = GT.set_index(["pid", "vid"]).loc[(pid, vid), "commit"]

        # Load data
        with open(os.path.join(f'/root/workspace/data/Defects4J/diff/{pid}-{vid}b', 'feature.pkl'), 'rb') as file:
            feature_dict = pickle.load(file)
        
        with open(os.path.join(f'/root/workspace/data/Defects4J/diff/{pid}-{vid}b', 'vocab.pkl'), 'rb') as file:
            vocab = pickle.load(file)
        
        vocab = {ind : word for word, ind in vocab.items()}
        org_dict = feature_dict[stage2][org_setting]
        new_dict = feature_dict[stage2][new_setting]

        for commit in org_dict.keys():
            print(f'Commit) {commit}')

            org_feature, new_feature = org_dict[commit], new_dict[commit]

            for (org_type, new_type) in target_type_list:
                aaa = {ind : freq for (ind, freq) in org_feature[org_type]}
                bbb = {ind : freq for (ind, freq) in new_feature[new_type]}

                for ind, freq in aaa.items():
                    if bbb.get(ind, 0) < freq:
                        print(f'Deleted token) {vocab[ind]}')
                    
                for ind, freq in bbb.items():
                    if ind not in aaa:
                        print(f'New token) {vocab[ind]}')
        
        break

def check_id_filter(stage2='precise', setting=frozenset({'tracker': 'git', 'diff_tool' : 'base', 'doc_level' : 'commit', 'adddel' : 'all_sep'}.items())):
    GT = load_BIC_GT("/root/workspace/data/Defects4J/BIC_dataset")
    res_dict = dict()

    org_setting = frozenset((dict(setting) | {'diff_type' : 'base'}).items())
    new_setting = frozenset((dict(setting) | {'diff_type' : 'id_all'}).items())
    adddel = dict(setting)['adddel']
    target_type_list = [('add_diff', 'add_id'), ('del_diff', 'del_id')] if adddel == 'all_sep' else [('diff', 'id')]

    # Iterate through projects
    for _, row in GT.iterrows():
        #pid, vid, BIC = row.pid, row.vid, row.commit
        pid = 'Closure'
        vid = '2'
        BIC = GT.set_index(["pid", "vid"]).loc[(pid, vid), "commit"]

        # Load data
        with open(os.path.join(f'/root/workspace/data/Defects4J/diff/{pid}-{vid}b', 'feature.pkl'), 'rb') as file:
            feature_dict = pickle.load(file)
        
        with open(os.path.join(f'/root/workspace/data/Defects4J/diff/{pid}-{vid}b', 'vocab.pkl'), 'rb') as file:
            vocab = pickle.load(file)
        
        vocab = {ind : word for word, ind in vocab.items()}
        org_dict = feature_dict[stage2][org_setting]
        new_dict = feature_dict[stage2][new_setting]

        for commit in org_dict.keys():
            print(f'Commit) {commit}')

            org_feature, new_feature = org_dict[commit], new_dict[commit]

            for (org_type, new_type) in target_type_list:
                aaa = {ind : freq for (ind, freq) in org_feature[org_type]}
                bbb = {ind : freq for (ind, freq) in new_feature[new_type]}

                for ind, freq in aaa.items():
                    if bbb.get(ind, 0) < freq:
                        print(f'Deleted token) {vocab[ind]}')
                    
                for ind, freq in bbb.items():
                    if ind not in aaa:
                        print(f'New token) {vocab[ind]}')
        
        break

# res_dict[stage2][(new_diff_type, use_stopword, adddel, use_br)]
if __name__ == "__main__":
    check_id_filter()