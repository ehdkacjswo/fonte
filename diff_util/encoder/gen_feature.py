import os, json, argparse, pickle
import numpy as np
import pandas as pd
from spiral import ronin
from collections import Counter
from tqdm import tqdm
from scipy.sparse import csc_array, save_npz

CORE_DATA_DIR = "/root/workspace/data/Defects4J/core"
BIC_GT_DIR = "/root/workspace/data/Defects4J/BIC_dataset"
BASELINE_DATA_DIR = "/root/workspace/data/Defects4J/baseline"
DIFF_DATA_DIR = '/root/workspace/data/Defects4J/diff'

class Encoder():
    def __init__(self, vocab={}):
        self.vocab = vocab # {word : id}
    
    # Encode the input and returns set of used ids
    def encode(self, text, update_vocab=True):
        encode_res = [0] * len(self.vocab)
        text = ronin.split(text.strip())

        for word, cnt in Counter(text).items():
            if word in self.vocab:
                encode_res[self.vocab[word]] = cnt
                
            elif update_vocab: # New word
                encode_res.append(cnt)
                self.vocab[word] = len(self.vocab)
        
        return encode_res

# Return the sum of two encoded vectors
def sum_encode(vec1, vec2):
    res_dict = dict()

    for id, cnt in vec1:
        res_dict[id] = cnt
    
    for id, cnt in vec2:
        res_dict[id] = res_dict.get(id, 0) + cnt
    
    return list(res_dict.items())

# data : 
def gen_feature(project, adddel, encode_type):
    project_dir = os.path.join(DIFF_DATA_DIR, project)
    feature_dict = dict()

    with open(os.path.join(project_dir, 'diff_encode.pkl'), 'rb') as file:
        encode_dict = pickle.load(file)

    # Iterate through commits
    for commit_hash, [addition_list, deletion_list, msg_encode] in encode_dict.items():
        feature_list = []

        if encode_type == 'simple':
            path_encode_sum = []
            content_encode_sum = []

            if adddel != 'del':
                for (src_path_encode, encode_sum) in addition_list:
                    path_encode_sum = sum_encode(path_encode_sum, src_path_encode)
                    content_encode_sum = sum_encode(content_encode_sum, encode_sum)
            
            if adddel != 'add':
                for (src_path_encode, encode_sum) in deletion_list:
                    path_encode_sum = sum_encode(path_encode_sum, src_path_encode)
                    content_encode_sum = sum_encode(content_encode_sum, encode_sum)
            
            feature_dict[commit_hash] = [msg_encode, path_encode_sum, content_encode_sum]
        
        else:
            continue
        
    feature_path = os.path.join(project_dir, 'feature')
    os.makedirs(feature_path, exist_ok=True)

    with open(os.path.join(feature_path, f'{adddel}_{encode_type}.pkl'), 'wb') as file:
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

    # Iterate through projects
    for project in os.listdir('/root/workspace/data/Defects4J/diff/'):
        print(f'Working on project {project}')
        gen_feature(project, args.adddel, args.encode_type)
        
