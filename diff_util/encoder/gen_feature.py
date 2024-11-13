import os, sys, json, argparse, pickle, itertools
import numpy as np
import pandas as pd
from spiral import ronin
from collections import Counter
from tqdm import tqdm

sys.path.append('/root/workspace/diff_util/lib/')
from diff import *
from encoder import *

CORE_DATA_DIR = "/root/workspace/data/Defects4J/core"
BIC_GT_DIR = "/root/workspace/data/Defects4J/BIC_dataset"
BASELINE_DATA_DIR = "/root/workspace/data/Defects4J/baseline"
DIFF_DATA_DIR = '/root/workspace/data/Defects4J/diff'

# Check if two encoded data are equal
def equal_encode(vec1, vec2):
    if len(vec1) != len(vec2):
        return False
    
    dict1 = dict(vec1)

    for (word, cnt) in vec2:
        if vec1.get(word, 0) != cnt:
            return False
    
    return True

# data : 
def gen_feature(project, stage2, use_stopword, adddel):
    project_dir = os.path.join(DIFF_DATA_DIR, project)
    with open(os.path.join(project_dir, 'encode', f'diff_encode.pkl'), 'rb') as file:
        encode_dict_total = pickle.load(file)

    encode_dict = encode_dict_total[(stage2, use_stopword)]
    feature_dict = dict()

    # Iterate through commits
    for commit_hash, [msg_encode, addition_list, deletion_list] in encode_dict.items():
        if adddel == 'all-uni':
            path_set = set()
            path_encode_sum = []
            content_encode_sum = []

            # Ecode whole data
            for (src_path_info, encode_sum) in addition_list:
                if src_path_info[0] not in path_set: # Only unique path for add & del
                    path_set.add(src_path_info[0])
                    path_encode_sum = sum_encode(path_encode_sum, src_path_info[1])
                content_encode_sum = sum_encode(content_encode_sum, encode_sum)

            for (src_path_info, encode_sum) in deletion_list:
                if src_path_info[0] not in path_set: # Only unique path for add & del
                    path_set.add(src_path_info[0])
                    path_encode_sum = sum_encode(path_encode_sum, src_path_info[1])
                content_encode_sum = sum_encode(content_encode_sum, encode_sum)
            
            feature_dict[commit_hash] = [msg_encode, path_encode_sum, content_encode_sum]

        else:
            # Encode addtion data
            add_path_encode_sum = []
            add_content_encode_sum = []

            if adddel != 'del':
                for (src_path_info, encode_sum) in addition_list:
                    add_path_encode_sum = sum_encode(add_path_encode_sum, src_path_info[1])
                    add_content_encode_sum = sum_encode(add_content_encode_sum, encode_sum)
            
            # Encode deletion data
            del_path_encode_sum = []
            del_content_encode_sum = []

            if adddel != 'add':
                for (src_path_info, encode_sum) in deletion_list:
                    del_path_encode_sum = sum_encode(del_path_encode_sum, src_path_info[1])
                    del_content_encode_sum = sum_encode(del_content_encode_sum, encode_sum)
            
            # Save the features
            if adddel == 'all-sep':
                feature_dict[commit_hash] = [msg_encode, add_path_encode_sum, del_path_encode_sum, \
                add_content_encode_sum, del_content_encode_sum]
            elif adddel == 'add':
                feature_dict[commit_hash] = [msg_encode, add_path_encode_sum, add_content_encode_sum]
            else:
                feature_dict[commit_hash] = [msg_encode, del_path_encode_sum, del_content_encode_sum]
        
    return feature_dict

if __name__ == "__main__":
    diff_dir = '/root/workspace/data/Defects4J/diff/'

    stage2_list = ['skip', True, False] # Skip stage or use OpenRewrite or not
    use_stopword_list = [True, False] # Use stopword or not
    adddel_list = ['add', 'del', 'all-uni', 'all-sep'] # Which diff data to uses
    param_list = list(itertools.product(stage2_list, use_stopword_list, adddel_list))
    
    # Iterate through projects
    for project in tqdm(os.listdir(diff_dir)):
        print(f'Working on project {project}')
        feature_res = dict()

        for (stage2, use_stopword, adddel) in param_list:
            feature_res[(stage2, use_stopword, adddel)] = gen_feature(project=project, stage2=stage2,\
                use_stopword=use_stopword, adddel=adddel)
        
        with open(os.path.join(diff_dir, project, 'feature/feature.pkl'), 'wb') as file:
            pickle.dump(feature_res, file)
        
