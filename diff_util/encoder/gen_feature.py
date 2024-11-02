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
def encode_pid(pid, adddel, encode_type):
    project_dir = os.path.join(DIFF_DATA_DIR, pid)

    # Iterate through commits
    for commit in tqdm(os.listdir(project_dir)):
        commit_dir = os.path.join(project_dir, commit)
        
        # Consider only commit directories (Possible vocab.pkl)
        if not os.path.isdir(commit_dir):
            continue

        feature_dict = dict()

        # Added diff data
        if adddel != 'del':
            with open(os.path.join(commit_dir, 'encode/addition_encode.pkl'), 'rb') as file:
                feature_dict = pickle.load(file)
        
        # Deleted diff data
        if adddel != 'add':
            with open(os.path.join(commit_dir, 'encode/deletion_encode.pkl'), 'rb') as file:
                deletion_dict = pickle.load(file)
            
            if bool(feature_dict): # Dictionary is empty
                feature_dict = deletion_dict
            
            else: # Dictionary is already filled with data
                for file_encode, diff_encode in deletion_dict.items():
                    feature_dict[file_encode] = sum_encode(feature_dict.get(file_encode, []), diff_encode)
        
        # Message data
        with open(os.path.join(commit_dir, 'encode/message_encode.pkl'), 'rb') as file:
            feature_list = [pickle.load(file)]
        
        if encode_type == 'complex':
            for file_encode, diff_encode in feature_dict.items():
                feature_list.append(sum_encode(file_encode, diff_encode))
        
        elif encode_type == 'simple':
            file_encode_sum = []
            for file_encode in feature_dict.keys():
                file_encode_sum = sum_encode(file_encode_sum, file_encode)
            feature_list.append(file_encode_sum)

            diff_encode_sum = []
            for diff_encode in feature_dict.values():
                diff_encode_sum = sum_encode(diff_encode_sum, diff_encode)
            feature_list.append(diff_encode_sum)
        
        with open(os.path.join(commit_dir, f'encode/feature_{adddel}_{encode_type}.pkl'), 'wb') as file:
            pickle.dump(feature_list, file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ensemble encoded data")
    #parser.add_argument('coredir', help="data/Defects4J/core/<pid>-<vid>b")
    parser.add_argument('--adddel', '-a', type=str, default='all', choices=['all', 'add', 'del'],
        help='Diff data to use, all, add or del (default: all)')
    # complex : [log, file_path + content...], simple : [log, file_path, content]
    parser.add_argument('--encode_type', '-e', type=str, default='simple', choices=['complex', 'simple'],
        help='Type of encoding (default: complex)')
    
    parser.add_argument('--relevant', '-r', type=bool, default=False,
        help="To use only relevant diff or not")
    parser.add_argument('--stopwords', '-s', type=bool, default=False,
        help="history retrieval tool, git or shovel (default: git)")
    # Stopwords and special characters?
    args = parser.parse_args()
    
    diff_dir = '/root/workspace/data/Defects4J/diff/'

    # Iterate through projects
    for pid in os.listdir('/root/workspace/data/Defects4J/diff/'):
        print(f'Working on project {pid}')
        encode_pid(pid, args.adddel, args.encode_type)
        
