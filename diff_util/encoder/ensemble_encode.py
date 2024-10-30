import os, json, argparse, pickle
import numpy as np
import pandas as pd
from spiral import ronin
from collections import Counter
from tqdm import tqdm
from scipy.sparse import csc_array, save_npz
from lib.experiment_utils import *

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

# Add new encoded vector to original one
# According to encoding theme, newer one is always longer
def add_zero_pad(org, new):
    for idx, cnt in enumerate(org):
        new[idx] += cnt
    
    return new

def encode_pid(pid, data, encode_type):
    project_dir = os.path.join(DIFF_DATA_DIR, pid)
    encoder = Encoder()

    # Iterate through commits
    for commit in tqdm(os.listdir(project_dir)):
        commit_dir = os.path.join(project_dir, commit)
        
        # Consider only commit directories (Possible vocab.pkl)
        if not os.path.isdir(commit_dir):
            continue

        if encode_type == 'complex':
            diff_addition_dict = dict() # filepath : contents
            diff_deletion_dict = dict() # new filepath if not /dev/null"""
        else:
            path_set = set() # To avoid duplicated path set
            path_res = [] # Encoding result of path info
            diff_res = [] # Encoding result of diff

        used_set_res = set()

        # Iterate through files
        for filename in os.listdir(commit_dir):
            cur_path = os.path.join(commit_dir, filename)
            
            # Consider only diff info (Possible ./encode/)
            if not os.path.isfile(cur_path):
                continue

            if cur_path.endswith('.csv'): # Diff data
                diff_df = pd.read_csv(cur_path)
                
                # Filter data in interest
                if data == 'all':
                    res_diff_df = diff_df
                elif data == 'add':
                    res_diff_df = diff_df[diff_df['is_addition'] == True]
                else:
                    res_diff_df = diff_df[diff_df['is_addition'] == False]

                for _, row in res_diff_df.iterrows(): # [is_addition, old_file_path, new_file_path, line_num, content]
                    file_path = str(row['new_file_path'])
                    encoded_content = str(row['content'])


                    # Target file doesn't exist (Deletion)
                    if not file_path.endswith('.java'):
                        file_path = str(row['old_file_path'])
                        path_set.add(str(row['old_file_path']))
                    
                    # Complexed encoding
                    if encode_type == 'complex':
                        if row['is_addition']:
                            if file_path in diff_addition_dict:
                                diff_addition_dict[file_path] = add_zero_pad(diff_addition_dict[file_path])
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

                    # Simple encoding
                    else:
                        path_set.add(file_path)
                        
                        diff_encode, used_set = encoder.encode(content)
                        
                        # At least one token exists
                        if bool(used_set):
                            for idx, cnt in enumerate(diff_res):
                                diff_encode[idx] += cnt
                            diff_res = diff_encode
                            used_set_res = used_set_res.union(used_set)

                    
            
            # Message data
            elif cur_path.endswith('.txt'):
                with open(cur_path, 'r') as file:
                    msg = file.read()

        # Encode the features
        encode_res = [diff_res]
        #used_set_res = set()

        """for file_path, content in diff_addition_dict.items():
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
            used_set_res = used_set_res.union(used_set)"""
        
        file_path_all = ""
        for file_path in path_set:
            file_path_all = file_path_all + '\n' + file_path
        
        res, used_set = encoder.encode(file_path_all)
        if bool(used_set):
            encode_res.append(res)
            used_set_res = used_set_res.union(used_set)
        
        # Save the encoding result for each commits
        encode_dir = os.path.join(commit_dir, 'encode')
        os.makedirs(encode_dir, exist_ok=True)

        with open(os.path.join(encode_dir, 'simple_encode_res.pkl'), 'wb') as file:
            pickle.dump(encode_res, file)
        with open(os.path.join(encode_dir, 'simple_used_set.pkl'), 'wb') as file:
            pickle.dump(used_set_res, file)
        
        for i in range(len(encode_res)):
            encode_res[i] = encode_res[i] + [0] * (len(encoder.vocab) - len(encode_res[i]))

        save_npz(os.path.join(encode_dir, 'simple_encode_res.npz'), csc_array(np.array(encode_res, dtype=int)))
    
    # Save the vocabulary for each projects
    with open(os.path.join(project_dir, 'vocab.pkl'), 'wb') as file:
        pickle.dump(encoder.vocab, file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ensemble encoded data")
    #parser.add_argument('coredir', help="data/Defects4J/core/<pid>-<vid>b")
    parser.add_argument('--data', '-d', type=str, default='all', choices=['all', 'add', 'del']
        help='Diff data to use, all, add or del (default: all)')
    # complex : [log, file_path + content...], simple : [log, file_path, content]
    parser.add_argument('--encode_type', '-e', type=str, default='complex', choices=['complex', 'simple']
        help='Type of encoding (default: complex)')
    
    parser.add_argument('--relevant', '-r', type=bool, default=False,
        help="To use only relevant diff or not")
    parser.add_argument('--stopwords', '-s', type=bool, default=False,
        help="history retrieval tool, git or shovel (default: git)")
    # Stopwords and special characters?
    args = parser.parse_args()
    
    GT = load_BIC_GT(BIC_GT_DIR)
    diff_dir = '/root/workspace/data/Defects4J/diff/'

    # Iterate through projects
    for pid in os.listdir('/root/workspace/data/Defects4J/diff/'):
        print(f'Working on project {pid}')
        encode_pid(pid)
        
