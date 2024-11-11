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
def gen_feature(project, tool='git', skip_stage_2=False, with_Rewrite=True, \
    use_stopword=True, adddel='all', encode_type='simple'):
    project_dir = os.path.join(DIFF_DATA_DIR, project)
    feature_dict = dict()

    file_postfix = savepath_postfix(tool, skip_stage_2, with_Rewrite, use_stopword)
    with open(os.path.join(project_dir, 'encode', f'diff_encode{file_postfix}.pkl'), 'rb') as file:
        encode_dict = pickle.load(file)

    # Iterate through commits
    for commit_hash, [addition_list, deletion_list, msg_encode] in encode_dict.items():
        feature_list = []

        if encode_type == 'simple':
            path_encode_list = list()
            path_encode_sum = []
            content_encode_sum = []

            if adddel != 'del':
                for (src_path_encode, encode_sum) in addition_list:
                    """# Check if given path is already added
                    if all(not equal_encode(src_path_encode, x) for x in path_encode_list):"""

                    path_encode_sum = sum_encode(path_encode_sum, src_path_encode)
                    content_encode_sum = sum_encode(content_encode_sum, encode_sum)
            
            if adddel != 'add':
                for (src_path_encode, encode_sum) in deletion_list:
                    """if src_path_encode in path_encode_set:
                        continue"""
                    path_encode_sum = sum_encode(path_encode_sum, src_path_encode)
                    content_encode_sum = sum_encode(content_encode_sum, encode_sum)
            
            feature_dict[commit_hash] = [msg_encode, path_encode_sum, content_encode_sum]
        
        else:
            continue
        
    feature_path = os.path.join(project_dir, 'feature')
    os.makedirs(feature_path, exist_ok=True)

    with open(os.path.join(feature_path, f'{adddel}{file_postfix}.pkl'), 'wb') as file:
        pickle.dump(feature_dict, file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ensemble encoded data")
    #parser.add_argument('coredir', help="data/Defects4J/core/<pid>-<vid>b")
    parser.add_argument('--adddel', '-a', type=str, default='all', choices=['all', 'add', 'del'],
        help='Diff data to use, all, add or del (default: all)')
    # complex : [log, file_path + content...], simple : [log, file_path, content]
    parser.add_argument('--encode_type', '-e', type=str, default='simple', choices=['complex', 'simple'],
        help='Type of encoding (default: simple)')
    args = parser.parse_args()
    
    diff_dir = '/root/workspace/data/Defects4J/diff/'

    skip_stage_2_list = [True, False]
    with_Rewrite_list = [True, False]
    use_stopword_list = [True, False]
    adddel_list = ['all', 'add', 'del']
    param_list = list(itertools.product(skip_stage_2_list, with_Rewrite_list, use_stopword_list, adddel_list))

    # Iterate through projects
    for project in os.listdir('/root/workspace/data/Defects4J/diff/'):
        print(f'Working on project {project}')
        for (skip_stage_2, with_Rewrite, use_stopword, adddel) in param_list:
            gen_feature(project, tool='git', skip_stage_2=skip_stage_2, with_Rewrite=with_Rewrite, use_stopword=use_stopword, adddel=adddel, encode_type='simple')
        
