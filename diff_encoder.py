import os
import json
import numpy as np
import pandas as pd
from spiral import ronin
from collections import Counter
from lib.experiment_utils import *

from tqdm import tqdm

import pickle

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
        used_set = set()
        text = ronin.split(text.strip())

        for word, cnt in Counter(text).items():
            try:
                used_set.add(self.vocab[word])
                encode_res[self.vocab[word]] = cnt
                
            except: # New word detected
                if update_vocab:
                    used_set.add(len(self.vocab))
                    encode_res.append(cnt)

                    self.vocab[word] = len(self.vocab)
        
        return encode_res, used_set

if __name__ == "__main__":
    GT = load_BIC_GT(BIC_GT_DIR)
    diff_dir = '/root/workspace/data/Defects4J/diff/'

    # Iterate through projects
    for pid in os.listdir('/root/workspace/data/Defects4J/diff/'):
        print(f'Working on project {pid}')
        project_dir = os.path.join(diff_dir, pid)
        encoder = Encoder()

        # Iterate through commits
        for commit in tqdm(os.listdir(project_dir)):
            commit_dir = os.path.join(project_dir, commit)

            diff_addition_dict = dict() # filepath : contents
            diff_deletion_dict = dict() # new filepath if not /dev/null

            # Iterate through files
            for filename in os.listdir(commit_dir):
                cur_path = os.path.join(commit_dir, filename)
                if not os.path.isfile(cur_path):
                    continue

                if cur_path.endswith('.csv'): # Diff data
                    diff_df = pd.read_csv(cur_path)
                    for _, row in diff_df.iterrows(): # [is_addition, old_file_path, new_file_path, line_num, content]
                        file_path = row['new_file_path']
                        content = row['content']

                        if row['is_addition']:
                            try:
                                diff_addition_dict[file_path] += ('\n' + content)
                            except:
                                diff_addition_dict[file_path] = content
                        
                        else:
                            # If file is deleted, use the old filename instead
                            if not file_path.endswith('.java'):
                                file_path = row['old_file_path']
                            
                            try:
                                diff_deletion_dict[file_path] += ('\n' + content)
                            except:
                                diff_deletion_dict[file_path] = content
                
                # Message data
                elif cur_path.endswith('.txt'):
                    with open(cur_path, 'r') as file:
                        msg = file.read()

            # Encode the features
            encode_res = []
            used_set_res = set()

            for file_path, content in diff_addition_dict.items():
                res, used_set = encoder.encode(file_path + content)

                if bool(used_set):
                    encode_res.append(res)
                    used_set_res = used_set_res.union(used_set)
            
            for file_path, content in diff_deletion_dict.items():
                res, used_set = encoder.encode(file_path + content)

                if bool(used_set):
                    encode_res.append(res)
                    used_set_res = used_set_res.union(used_set)
            
            res, used_set = encoder.encode(msg)
            if bool(used_set):
                encode_res.append(res)
                used_set_res = used_set_res.union(used_set)
            
            # Save the encoding result for each commits
            encode_dir = os.path.join(commit_dir, 'encode')
            os.makedirs(encode_dir, exist_ok=True)

            with open(os.path.join(encode_dir, 'encode_res.pkl'), 'wb') as file:
                pickle.dump(encode_res, file)
            with open(os.path.join(encode_dir, 'used_set.pkl'), 'wb') as file:
                pickle.dump(used_set_res, file)
        
        # Save the vocabulary for each projects
        with open(os.path.join(project_dir, 'vocab.pkl'), 'wb') as file:
            pickle.dump(encoder.vocab, file)