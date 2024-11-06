import os, json, argparse, pickle, sys
import numpy as np
import pandas as pd
import regex as re
from spiral import ronin
from collections import Counter
from tqdm import tqdm
from nltk.corpus import stopwords

sys.path.append('/root/workspace/diff_util/')
from diff import Diff_commit

DIFF_DATA_DIR = '/root/workspace/data/Defects4J/diff'
CORE_DATA_DIR = '/root/workspace/data/Defects4J/core'
BASE_DATA_DIR = '/root/workspace/data/Defects4J/baseline'

# Class that encodes string while expanding the vocabulary
class Encoder():
    def __init__(self, vocab={}):
        self.vocab = vocab # {word : id}
        self.stopword_list = stopwords.words('english')
    
    def tokenize(self, text):
        text = re.sub(r'[^A-Za-z0-9]', ' ', text) # Remove characters except alphabets and numbers
        token_list = ronin.split(text) # Split the text
        token_list = [token.lower() for token in token_list] # Apply lowercase
        token_list = [token for token in token_list if \
            (len(token) > 1 and not token.isdigit() and token not in self.stopword_list)]
            # Remove single character, numbers and stopwords

        return token_list

    # Encode the input and list of used word index and count
    def encode(self, text):
        encode_res = []
        text = self.tokenize(text)

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

# Get the unchanged file data
# [(commit, src_path)]
def get_style_change_data(coredir, tool='git', with_Rewrite=True):
    postfix = "" if with_Rewrite else "_noOpenRewrite"
    val_df = pd.read_csv(
        os.path.join(coredir, tool, f"validation{postfix}.csv"), 
        header=None,
        names=["commit", "src_path", "AST_diff"])
    
    unchanged_df = val_df[val_df["AST_diff"] == "U"]
    return list(zip(unchanged_df["commit"], unchanged_df["src_path"]))

# Encode the raw diff data
# Encoded data : {commit_hash : [addition_list, deletion_list, msg_encode]}
# addition/deletion dict : [(before/after_src_path_encode, content_encode_sum)]
def encode_pid(pid, vid, skip_stage_2=False, with_Rewrite=True):
    # Load related diff data
    diff_path = os.path.join(DIFF_DATA_DIR, f'{pid}-{vid}b/diff.pkl')
    with open(diff_path, 'rb') as file:
        diff_data = pickle.load(file)
    
    # Get list of style change commits
    if skip_stage_2:
        excluded = []
    else:
        excluded = get_style_change_data(os.path.join(CORE_DATA_DIR, f'{pid}-{vid}b'), 'git', with_Rewrite)

    # Merge the diff data
    diff_dict = dict()

    for commit_hash, commit_diff in diff_data.diff_dict.items(): # Iterate through commits
        addition_dict = dict()
        deletion_dict = dict()

        for src_path, src_diff in commit_diff.items(): # Iterate through source files editted by commit
            if (commit_hash, src_path) in excluded: # Exclude style change
                continue
            
            for (before_src_path, after_src_path), [addition, deletion] in src_diff.diff_dict.items():
                addition_content_dict = addition_dict.get(after_src_path, dict())
                deletion_content_dict = deletion_dict.get(before_src_path, dict())

                for line, content in addition.items():
                    if line not in addition_content_dict:
                        addition_content_dict[line] = content
                    
                    """elif content != addition_content_dict[line]: # Different diff data for same source file
                        print(f'Different addition content!!! {commit_hash} {after_src_path} {line}')"""
                
                for line, content in deletion.items():
                    if line not in deletion_content_dict:
                        deletion_content_dict[line] = content
                    
                    """elif content != deletion_content_dict[line]: # Different diff data for same source file
                        print(f'Different deletion content!!! {commit_hash} {before_src_path} {line}')"""
                
                addition_dict[after_src_path] = addition_content_dict
                deletion_dict[before_src_path] = deletion_content_dict
        
        diff_dict[commit_hash] = [addition_dict, deletion_dict]
    
    # Encode the merged data
    encode_dict = dict()
    encoder = Encoder()
    base_data_dir = os.path.join(BASE_DATA_DIR, f'{pid}-{vid}b/commits/')

    for commit_hash, [addition_dict, deletion_dict] in diff_dict.items():
        addition_list = []
        deletion_list = []

        # Encode addition data
        for after_src_path, addition_content_dict in addition_dict.items():
            after_src_path_encode = encoder.encode(after_src_path)
            addition_encode_sum = list()

            for content in addition_content_dict.values():
                addition_encode_sum = sum_encode(addition_encode_sum, encoder.encode(content))
            
            addition_list.append((after_src_path_encode, addition_encode_sum))
        
        # Encode deletion data
        for before_src_path, deletion_content_dict in deletion_dict.items():
            before_src_path_encode = encoder.encode(before_src_path)
            deletion_encode_sum = list()

            for content in deletion_content_dict.values():
                addition_encode_sum = sum_encode(deletion_encode_sum, encoder.encode(content))
            
            deletion_list.append((before_src_path_encode, deletion_encode_sum))
        
        # Encode message
        for filename in os.listdir(base_data_dir):
            if filename.startswith(f'c_{commit_hash}'):
                with open(os.path.join(base_data_dir, filename), "r") as file:
                    data = json.load(file)
                msg_encode = encoder.encode(data['log'])
                break
        
        encode_dict[commit_hash] = [addition_list, deletion_list, msg_encode]

    # Save encoded data and vocab
    with open(os.path.join(DIFF_DATA_DIR, f'{pid}-{vid}b/diff_encode.pkl'), 'wb') as file:
        pickle.dump(encode_dict, file)
    with open(os.path.join(DIFF_DATA_DIR, f'{pid}-{vid}b/vocab.pkl'), 'wb') as file:
        pickle.dump(encoder.vocab, file)

if __name__ == "__main__":
    # Iterate through projects
    for project_dir in os.listdir(DIFF_DATA_DIR):
        print(f'Working on project {project_dir}')
        [pid, vid] = project_dir[:-1].split("-")
        encode_pid(pid, vid)
        
