import os, json, argparse, pickle, sys, itertools
import pandas as pd
from tqdm import tqdm

sys.path.append('/root/workspace/data_collector/lib/')
from encoder import *

DIFF_DATA_DIR = '/root/workspace/data/Defects4J/diff'
CORE_DATA_DIR = '/root/workspace/data/Defects4J/core'
BASE_DATA_DIR = '/root/workspace/data/Defects4J/baseline'

def log(txt, out_txt=None, err_txt=None):
    with open('/root/workspace/data_collector/log/encode.log', 'a') as file:
        file.write(txt + '\n')

        if out_txt is not None:
            file.write('[ERROR] OUT\n' + out_txt.decode(encoding='utf-8', errors='ignore') + '\n')
        
        if err_txt is not None:
            file.write('[ERROR] ERR\n' + err_txt.decode(encoding='utf-8', errors='ignore') + '\n')

# Encode the raw diff data
# skip_stage_2 = Excluding style change diff, with_Rewrite = , use_stopword
# Encoded data : {commit_hash : [addition_list, deletion_list, msg_encode]}
# addition/deletion dict : [(before/after_src_path_encode, content_encode_sum)]
def encode(stage2_data, pid, vid, use_stopword):
    encoder = Encoder()
    res_dict = dict()

    for commit_hash, commit_diff in stage2_data.items(): # Iterate through commits
        res_dict[commit_hash] = dict()

        for modify, modify_diff in commit_diff.items(): # addition / deletion
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

def main(pid, vid):
    log(f'Working on {pid}_{vid}b')

    with open(os.path.join(DIFF_DATA_DIR, f'{pid}-{vid}b', 'stage2.pkl'), 'rb') as file:
        stage2_dict = pickle.load(file)

    # Encode diff for every settings
    encode_dict = dict()
    vocab_dict = dict()
    use_stopword_list = [True]

    for stage2, sub_dict in stage2_dict.items():
        encode_dict[stage2] = dict()
        vocab_dict[stage2] = dict()

        for setting, stage2_data in sub_dict.items():
            setting_dict = dict(setting)

            for use_stopword in use_stopword_list:
                new_setting = frozenset((setting_dict | {'use_stopword' : True}).items())
                encode_res, vocab = encode(stage2_data=stage2_data, pid=pid, vid=vid, use_stopword=use_stopword)

                encode_dict[stage2][new_setting] = encode_res
                vocab_dict[stage2][new_setting] = vocab

    diff_data_dir = os.path.join(DIFF_DATA_DIR, f'{pid}-{vid}b')
    os.makedirs(diff_data_dir, exist_ok=True)

    with open(os.path.join(diff_data_dir, f'encode.pkl'), 'wb') as file:
        pickle.dump(encode_dict, file)
    with open(os.path.join(diff_data_dir, f'vocab.pkl'), 'wb') as file:
        pickle.dump(vocab_dict, file)