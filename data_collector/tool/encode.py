import os, json, argparse, pickle, sys, itertools
import pandas as pd
from tqdm import tqdm

sys.path.append('/root/workspace/data_collector/lib/')
from encoder import *
from gumtree import CustomInterval
from utils import *

DIFF_DATA_DIR = '/root/workspace/data/Defects4J/diff'
CORE_DATA_DIR = '/root/workspace/data/Defects4J/core'
BASE_DATA_DIR = '/root/workspace/data/Defects4J/baseline'

# Encode the raw diff data
# skip_stage_2 = Excluding style change diff, with_Rewrite = , use_stopword
# Encoded data : {commit_hash : [addition_list, deletion_list, msg_encode]}
# addition/deletion dict : [(before/after_src_path_encode, content_encode_sum)]
def encode(stage2_data, pid, vid, use_stopword):
    encoder = Encoder()
    res_dict = dict()

    for commit_hash, commit_diff in tqdm(stage2_data.items()): # Iterate through commits
        #print(commit_hash)
        res_dict[commit_hash] = dict()

        for modify, modify_diff in commit_diff.items(): # addition / deletion
            #print(modify)
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
        print('Encode message')
        base_data_dir = os.path.join(BASE_DATA_DIR, f'{pid}-{vid}b', 'commits')

        for filename in os.listdir(base_data_dir):
            if filename.startswith(f'c_{commit_hash}'):
                with open(os.path.join(base_data_dir, filename), "r") as file:
                    data = json.load(file)
                res_dict[commit_hash]['message'] = encoder.encode(data['log'], use_stopword=use_stopword, update_vocab=True)
                break
    
    return res_dict, encoder.vocab

# Encode the data 
# Add file
def pre_encode(intvl_dict, encoder):
    for setting, setting_dict in intvl_dict.items():
        for commit, commit_dict in setting_dict.items():
            for path_tup, diff_dict in commit_dict.items():
                diff_dict['encoded_path'] = (encoder.encode(path_tup[0]), encoder.encode(path_tup[1]))
                file_dict[tracker][commit][path_tup].setdefault('encoded_path', diff_dict['encoded_path'])

                # Get the corresponding code text (Possibly None)
                after_code = get_src_from_commit(commit, path_tup[1])
                before_code = get_src_from_commit(commit + '~1', path_tup[0])

                # Get the tokens in the list
                for diff_type, char_intvl in diff_dict['addition'].items():
                    tokens = get_tokens_intvl(after_code, char_intvl)
                    if tokens is None:
                        log('encode', f'[ERROR] Failed to get file {commit}:{path_tup[1]}')
                        # How to handle such situations?
                    else:
                        diff_dict['addition'][diff_type] = [encoder.encode(token) for token in tokens]
                
                for diff_type, char_intvl in diff_dict['deletion'].items():
                    tokens = get_tokens_intvl(before_code, char_intvl)
                    if tokens is None:
                        log('encode', f'[ERROR] Failed to get file {commit}~1:{path_tup[0]}')
                    else:
                        diff_dict['deletion'][diff_type] = [encoder.encode(token) for token in tokens]
    

    
def main(pid, vid):
    log('encode', f'Working on {pid}_{vid}b')

    # Checkout Defects4J project
    p = subprocess.Popen(['sh', '/root/workspace/lib/checkout.sh', pid, vid], \
        stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out_txt, err_txt = p.communicate()

    if p.returncode != 0:
        log('encode', '[ERROR] Checkout failed', out_txt, err_txt)
        return
    
    # Change working directory to target Defects4J project
    try:
        os.chdir(f'/tmp/{pid}-{vid}b/')
    except:
        log('encode', '[ERROR] Moving directory failed')
        return

    diff_data_dir = os.path.join(DIFF_DATA_DIR, f'{pid}-{vid}b')
    os.makedirs(diff_data_dir, exist_ok=True)
    
    with open(os.path.join(diff_data_dir, 'total_intvl.pkl'), 'rb') as file:
        intvl_dict = pickle.load(file)
    
    # Load the previous result if possible
    """encode_save_path = os.path.join(diff_data_dir, f'encode.pkl')
    vocab_save_path = os.path.join(diff_data_dir, f'vocab.pkl')

    if os.path.isfile(encode_save_path) and os.path.isfile(vocab_save_path):
        with open(encode_save_path, 'rb') as file:
            encode_dict = pickle.load(file)
        with open(vocab_save_path, 'rb') as file:
            vocab_dict = pickle.load(file)
    
    else:
        encode_dict = dict()
        vocab_dict = dict()"""

    # Encode diff for every settings
    encoder = Encoder()
    pre_encode(intvl_dict, encoder)

    """with open(encode_save_path, 'wb') as file:
        pickle.dump(encode_dict, file)
    with open(vocab_save_path, 'wb') as file:
        pickle.dump(vocab_dict, file)"""