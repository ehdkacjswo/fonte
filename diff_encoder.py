import os
import json
import numpy as np
import pandas as pd
from spiral import ronin
from rank_bm25 import BM25Okapi
from collections import Counter
from scipy.spatial.distance import cosine
from lib.experiment_utils import *

from tqdm import tqdm

import pickle

CORE_DATA_DIR = "./data/Defects4J/core"
BIC_GT_DIR = "./data/Defects4J/BIC_dataset"
BASELINE_DATA_DIR = "./data/Defects4J/baseline"
DIFF_DATA_DIR = './data/Defects4J/diff'

def tokenize(text):
    return ronin.split(text)

class ProjectEncoder():
    class Encoder():
        def __init__(self, vocab={}):
            self.vocab = vocab
        
        # Encode the input and 
        # (Input must be list of tokens)
        def encode(self, input, update_vocab=True):
            res = [0] * len(self.vocab)

            for word, cnt in Counter(input).items():
                try:
                    res[self.vocab[word]] = cnt
                except: # New word detected
                    if update_vocab:
                        self.vocab[word] = len(self.vocab)
                        res.append(cnt)
            
            return res
    
    def __init__(self):
        self.encoder_vocab = {}
    
    # vocab = {word : index} for each projects
    # encode_dict = {filename : encoded diff, log : encoded log}
    def encode_diff(self, pid, vid):
        print(f"Working on {pid}-{vid}b..........................")

        core_data_dir = os.path.join(CORE_DATA_DIR, f"{pid}-{vid}b")
        diff_data_dir = os.path.join(DIFF_DATA_DIR, f"{pid}-{vid}b")
        baseline_data_dir = os.path.join(BASELINE_DATA_DIR, f"{pid}-{vid}b")
        commit_dir = os.path.join(baseline_data_dir, "commits")

        if pid not in self.encoder_vocab:
            self.encoder_vocab[pid] = self.Encoder()

        commits_done = set(os.listdir(os.path.join(DIFF_DATA_DIR, f'{pid}')))

        log_encode = dict()
        
        # filename : c_{commit_hash}.json
        for filename in os.listdir(commit_dir):
            commit_hash = filename[2:9]

            if not commit_hash + '.pkl' in commits_done:
                with open(os.path.join(commit_dir, filename), "r") as f:
                    data = json.load(f)
                    log_encode[commit_hash] = self.encoder_vocab[pid].encode(tokenize(data["log"].strip()))
            
        # Add diff data
        for commit_sha in tqdm(os.listdir(diff_data_dir)):
            # It seems some commit info is missing... (Ex Cli-29b/38d83e1)
            if not commit_sha[:7] in log_encode:
                #print('No commit data for {}_{}/{}'.format(pid, vid, commit_sha))
                continue

            with open(os.path.join(diff_data_dir, commit_sha, 'addition.pkl'), 'rb') as f:
                diff_dict = pickle.load(f)
            
            diff_dict_file = {}

            for filename, content in diff_dict.items():
                if filename[0].endswith('.java'):
                    try:
                        diff_dict_file[filename[0]].append(content)
                    except:
                        diff_dict_file[filename[0]] = [content]
            
            encode_list = []
            # It seems some commit info is missing... (Ex Cli-29b/38d83e1)
            for filename in diff_dict_file.keys():
                diff_dict_file[filename] = self.encoder_vocab[pid].encode(tokenize(filename + '\n' + '\n'.join(diff_dict_file[filename])))
            
            diff_dict_file['log'] = log_encode[commit_sha[:7]]
            with open(os.path.join(DIFF_DATA_DIR, f'{pid}', f'{commit_sha[:7]}.pkl'), 'wb') as f:
                pickle.dump(diff_dict_file, f)
        
    def end(self):
        for key, item in self.encoder_vocab.items():
            with open(os.path.join(DIFF_DATA_DIR, f'{key}', 'vocab.pkl'), 'wb') as f:
                pickle.dump(item, f)

if __name__ == "__main__":
    GT = load_BIC_GT(BIC_GT_DIR)
    projectencoder = ProjectEncoder()
    for _, row in GT.iterrows():
        pid, vid = row.pid, row.vid
        projectencoder.encode_diff(pid, vid)
    projectencoder.end()