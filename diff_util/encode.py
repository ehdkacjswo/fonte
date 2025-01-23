import os, json, argparse, pickle, sys, itertools
import pandas as pd
from tqdm import tqdm

sys.path.append('/root/workspace/diff_util/lib/')
from diff import *
from encoder import *

DIFF_DATA_DIR = '/root/workspace/data/Defects4J/diff'
CORE_DATA_DIR = '/root/workspace/data/Defects4J/core'
BASE_DATA_DIR = '/root/workspace/data/Defects4J/baseline'

# Encode the raw diff data
# skip_stage_2 = Excluding style change diff, with_Rewrite = , use_stopword
# Encoded data : {commit_hash : [addition_list, deletion_list, msg_encode]}
# addition/deletion dict : [(before/after_src_path_encode, content_encode_sum)]
def stage3(stage2_data, pid, vid, diff_type, use_stopword):
    encoder = Encoder()
    res_dict = dict()

    for commit_hash, commit_diff in stage2_data.items(): # Iterate through commits
        res_dict[commit_hash] = dict()

        for modify, modify_diff in commit_diff.items():
            res_dict[commit_hash][modify] = dict()
        
            for src_path, src_diff in modify_diff.items():
                res_dict[commit_hash][modify][src_path] = dict()

                for token_type, token_list in src_diff.items():
                    res_dict[commit_hash][modify][src_path][token_type] = []

                    for token in token_list:
                        res_dict[commit_hash][modify][src_path][token_type] = \
                            sum_encode(res_dict[commit_hash][modify][src_path][token_type], \
                            encoder.encode(token, use_stopword=use_stopword, update_vocab=True))
                
                # Encode path
                res_dict[commit_hash][modify][src_path]['src_path'] = \
                    encoder.encode(src_path, use_stopword=use_stopword, update_vocab=True)
            
        # Encode message
        base_data_dir = os.path.join(BASE_DATA_DIR, f'{pid}-{vid}b', 'commits')

        for filename in os.listdir(base_data_dir):
            if filename.startswith(f'c_{commit_hash}'):
                with open(os.path.join(base_data_dir, filename), "r") as file:
                    data = json.load(file)
                res_dict[commit_hash]['message'] = encoder.encode(data['log'], use_stopword=use_stopword, update_vocab=True)
                break
    
    return res_dict, encoder.vocab

if __name__ == "__main__":
    #diff_type_list = ['git', 'gumtree_base', 'gumtree_class']
    #stage2_list = ['skip'] # ['skip', True, False] Skip stage or use OpenRewrite or not
    use_stopword_list = [True, False] # [True, False] Use stopword or not    
    #param_list = list(itertools.product(diff_type_list, stage2_list, use_stopword_list))
    
    for project_dir in tqdm(os.listdir(DIFF_DATA_DIR)):
        print(f'Working on project {project_dir}')
        [pid, vid] = project_dir[:-1].split("-")

        pid = "Cli"
        vid = "29"
        project_dir = "Cli-29b"
        
        with open(os.path.join(DIFF_DATA_DIR, project_dir, 'stage2.pkl'), 'rb') as file:
            stage2_dict = pickle.load(file)

        # Encode diff for every settings
        encode_dict = dict()
        vocab_dict = dict()

        for stage2, sub_dict in stage2_dict.items():
            encode_dict[stage2] = dict()
            vocab_dict[stage2] = dict()

            for diff_type, stage2_data in sub_dict.items():
                for use_stopword in use_stopword_list:
                    encode_res, vocab = stage3(stage2_data=stage2_data, pid=pid, vid=vid, \
                        diff_type=diff_type, use_stopword=use_stopword)

                    encode_dict[stage2][(diff_type, use_stopword)] = encode_res
                    vocab_dict[stage2][(diff_type, use_stopword)] = vocab

        diff_data_dir = os.path.join(DIFF_DATA_DIR, f'{pid}-{vid}b')
        os.makedirs(diff_data_dir, exist_ok=True)

        with open(os.path.join(diff_data_dir, f'encode.pkl'), 'wb') as file:
            pickle.dump(encode_dict, file)
        with open(os.path.join(diff_data_dir, f'vocab.pkl'), 'wb') as file:
            pickle.dump(vocab_dict, file)
        
        print(encode_dict)
        print(vocab_dict)
        break