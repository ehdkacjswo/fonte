import os, json, argparse, pickle
import numpy as np
import pandas as pd
from spiral import ronin
from collections import Counter
from tqdm import tqdm

DIFF_DATA_DIR = '/root/workspace/data/Defects4J/diff'

# Class that encodes string while expanding the vocabulary
class Encoder():
    def __init__(self, vocab={}):
        self.vocab = vocab # {word : id}
    
    # Encode the input and list of used word index and count
    def encode(self, text):
        encode_res = []
        text = ronin.split(text.strip())

        for word, cnt in Counter(text).items():
            if word in self.vocab: # Word in vocab
                encode_res.append((self.vocab[word], cnt))
                
            else: # New word
                encode_res.append((len(self.vocab), cnt))
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

# Encode the raw diff data
# Original data : [is_addition, old_file_path, new_file_path, line_num, content]
# Encoded data : addition / deletion {file_path : sum(content_encode)}
# Encoded data is list of (non_zero_index, non_zero_data)
def encode_pid(pid):
    project_dir = os.path.join(DIFF_DATA_DIR, pid)
    encoder = Encoder()

    # Iterate through commits
    for commit in tqdm(os.listdir(project_dir)):
        commit_dir = os.path.join(project_dir, commit)
        
        # Consider only commit directories (Possible vocab.pkl)
        if not os.path.isdir(commit_dir):
            continue

        # Iterate through files
        for filename in os.listdir(commit_dir):
            cur_path = os.path.join(commit_dir, filename)
            
            # Consider only diff info (Possible ./encode/)
            if not os.path.isfile(cur_path):
                continue

            if cur_path.endswith('.csv'): # Diff data
                diff_df = pd.read_csv(cur_path)
                addition_dict = dict()
                deletion_dict = dict()

                for _, row in diff_df.iterrows(): # [is_addition, old_file_path, new_file_path, line_num, content]
                    file_path = str(row['new_file_path'])
                    file_path_encode = tuple(encoder.encode(file_path)) # Tuple to use it as key
                    content_encode = encoder.encode(str(row['content']))

                    if row['is_addition']:
                        addition_dict[file_path_encode] = \
                            sum_encode(addition_dict.get(file_path_encode, []), content_encode)
                    
                    else:
                        if not file_path.endswith('.java'): # File deleted
                            file_path_encode = tuple(encoder.encode(str(row['old_file_path']))) # Use the old path instead
                        deletion_dict[file_path_encode] = \
                            sum_encode(deletion_dict.get(file_path_encode, []), content_encode)
                
                # Save the encoded data             
                with open(os.path.join(commit_dir, 'encode/addition_encode.pkl'), 'wb') as file:
                    pickle.dump(addition_dict, file)    
                with open(os.path.join(commit_dir, 'encode/deletion_encode.pkl'), 'wb') as file:
                    pickle.dump(deletion_dict, file)
            
            if filename == 'message.txt': # Msg data
                with open(cur_path, 'r') as file:
                    msg = file.read()
                    
                msg_encode = encoder.encode(msg)
                with open(os.path.join(commit_dir, 'encode/message_encode.pkl'), 'wb') as file:
                    pickle.dump(msg_encode, file)
        
    # Save the vocabulary for each projects
    with open(os.path.join(project_dir, 'vocab.pkl'), 'wb') as file:
        pickle.dump(encoder.vocab, file)

if __name__ == "__main__":
    # Iterate through projects
    for pid in os.listdir(DIFF_DATA_DIR):
        print(f'Working on project {pid}')
        encode_pid(pid)
        
