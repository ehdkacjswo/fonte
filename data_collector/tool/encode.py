import os, json, argparse, pickle, sys, itertools, math
import pandas as pd
from tqdm import tqdm

sys.path.append('/root/workspace/data_collector/lib/')
from encoder import *
from utils import *

DIFF_DATA_DIR = '/root/workspace/data/Defects4J/diff'
CORE_DATA_DIR = '/root/workspace/data/Defects4J/core'
BASE_DATA_DIR = '/root/workspace/data/Defects4J/baseline'

adddel_list = ['add', 'del', 'all_uni', 'all_sep']

# Returns total_intvl, {commit : {adddel : src_path} }, {commit : commit message}, encoder
def load_data(pid, vid):
    
    # Load total interval data
    with open(os.path.join(os.path.join(DIFF_DATA_DIR, f'{pid}-{vid}b', 'total_intvl.pkl')), 'rb') as file:
        total_intvl = pickle.load(file)

    # Get src paths related to commits (For faster encoding)
    commit_path_dict = dict()

    for setting_dict in total_intvl.values():
        for commit_dict in setting_dict.values():
            for commit, adddel_dict in commit_dict.items():
                commit_path_dict.setdefault(commit, {'addition' : set(), 'deletion' : set()})

                for adddel, path_dict in commit_dict.items():
                    for path in path_dict.keys():
                        commit_path_dict[commit][adddel].add(path)
    
    # Get commit messages
    encoder, commit_msg_dict = Encoder(), dict() 
    base_data_dir = os.path.join(BASE_DATA_DIR, f'{pid}-{vid}b', 'commits')

    for commit in commit_path_dict.keys():
        for filename in os.listdir(base_data_dir):
            if filename.startswith(f'c_{commit}'):
                with open(os.path.join(base_data_dir, filename), "r") as file:
                    data = json.load(file)
                    
                commit_msg_dict[commit] = encoder.encode(data['log'], use_stopword=True, update_vocab=True)
                break
    
    return total_intvl, commit_path_dict, commit_msg_dict


# Encode the data 
# Add file
def pre_encode(total_intvl, commit_path_dict, commit_msg_dict):
    
    # For identifier handling, each setting needs independent vocabulary
    encoder_dict = dict()

    # Get the tokens from interval & encode
    for commit, adddel_dict in commit_path_dict.items():
        for adddel, path_set in adddel_dict.items():
            target_commit = commit + ('' if adddel == 'addition' else '~1')

            for src_path in path_set:
                
                # Get the code text
                code_txt = get_src_from_commit(target_commit, src_path)

                if code_txt is None:
                    log('encode', f'[ERROR] Failed to get file {target_commit}:{src_path}')
                    return False
                
                for stage2, setting_dict in total_intvl.items():
                    encoder_dict.setdefault(stage2, dict())

                    for setting, commit_dict in setting_dict.items():
                        encoder_dict[stage2].setdefault(setting, Encoder())

                        # Setting contains interval data for given commit, adddel, src_path
                        if (commit in commit_dict) and (src_path in commit_dict[commit][adddel]):
                            
                            # Get the tokens in the given interval
                            token_intvl_dict = commit_dict[commit][adddel][adddel][src_path]
                            token_intvl_dict = get_tokens_intvl(code_txt, token_intvl_dict)

                            for diff_type, tokens in token_intvl_dict.items():

                                # Encode source code (Not classified)
                                if diff_type == 'diff':
                                    mode = 'code'
                                
                                # Encode comment, string and annotation (Treated as pure text)
                                elif diff_type == 'comment' or diff_type == 'non_id':
                                    mode = 'text'
                                
                                # Encode identifier
                                else:
                                    mode = 'id'
                                    
                                tokens = encoder_dict[stage2][setting].encode()


    for stage2, setting_dict in total_intvl.items():
        encoder_dict.setdefault(stage2, dict())

        for setting, commit_dict in setting_dict.items():
            encoder_dict[stage2].setdefault(setting, Encoder())

            # Setting contains interval data for given commit, adddel, src_path
            if (commit in commit_dict) and (src_path in commit_dict[commit][adddel]):
                
                # Get the tokens in the given interval
                token_intvl_dict = commit_dict[commit][adddel][adddel][src_path]
                token_intvl_dict = get_tokens_intvl(code_txt, token_intvl_dict)

                for diff_type, tokens in token_intvl_dict.items():
                    tokens = encoder_dict[stage2][setting].encode()
    
    return intvl_dict, commit_msg_dict, encoder.vocab

# Encode the raw diff data
# skip_stage_2 = Excluding style change diff, with_Rewrite = , use_stopword
# Encoded data : {commit_hash : [addition_list, deletion_list, msg_encode]}
# addition/deletion dict : [(before/after_src_path_encode, content_encode_sum)]
# data : message, src_path, diff
# No diff 추가(는 나중에?)
def gen_feature(enc_data, commit_msg_dict):
    res_dict = {adddel : dict() for adddel in adddel_list}
    no_diff_dict = {adddel : dict() for adddel in adddel_list}

    # Iterate through commits
    for commit, commit_dict in enc_data.items():
        path_set = {'addition' : set(), 'deletion' : set(), 'all' : set()}
        path_enc = {'addition' : list(), 'deletion' : list(), 'all' : list()}
        diff_enc_dict = {'addition' : dict(), 'deletion' : dict()}

        # Aggregate encoded data of commit
        for adddel, adddel_dict in commit_dict.items(): # adddel : addition, deletion
            for src_path, src_enc_dict in adddel_dict.items():
                    
                # Consider only unique paths
                if src_path not in path_set[adddel]:
                    path_set[adddel].add(src_path)
                    path_enc[adddel] = sum_encode(path_enc[adddel], src_enc_dict['path'])

                if src_path not in path_set['all']:
                    path_set['all'].add(src_path)
                    path_enc['all'] = sum_encode(path_enc['all'], src_enc_dict['path'])
                
                # Aggregate encoded data (except path)
                for diff_type, diff_enc in src_enc_dict.items():
                    if diff_type != 'path':
                        diff_enc_dict[adddel][diff_type] = sum_encode(diff_enc_dict[adddel].get(diff_type, []), diff_enc)
        
        # Generate feature
        for adddel in adddel_list: # adddel : add, del, all_uni, all_sep
            res_dict[adddel][commit] = {'msg' : commit_msg_dict.get(commit, [])}
            no_diff_dict[adddel][commit] = {'msg' : commit_msg_dict.get(commit, [])}

            if adddel == 'all_uni':
                res_dict[adddel][commit]['path'] = path_enc['all']
                no_diff_dict[adddel][commit]['path'] = path_enc['all']

                for diff_type in diff_enc_dict['addition'].keys() | diff_enc_dict['deletion'].keys():
                    res_dict[adddel][commit][diff_type] = sum_encode(diff_enc_dict['addition'].get(diff_type, []), \
                        diff_enc_dict['deletion'].get(diff_type, []))
            
            elif adddel == 'all_sep':
                res_dict[adddel][commit]['add_path'] = path_enc['addition']
                res_dict[adddel][commit]['del_path'] = path_enc['deletion']

                no_diff_dict[adddel][commit]['add_path'] = path_enc['addition']
                no_diff_dict[adddel][commit]['del_path'] = path_enc['deletion']

                for diff_type in diff_enc_dict['addition'].keys() | diff_enc_dict['deletion'].keys():
                    res_dict[adddel][commit]['add_' + diff_type] = diff_enc_dict['addition'].get(diff_type, [])
                    res_dict[adddel][commit]['del_' + diff_type] = diff_enc_dict['deletion'].get(diff_type, [])
            
            elif adddel == 'add':
                res_dict[adddel][commit]['path'] = path_enc['addition']
                no_diff_dict[adddel][commit]['path'] = path_enc['addition']

                for diff_type, diff_enc in diff_enc_dict['addition'].items():
                    res_dict[adddel][commit][diff_type] = diff_enc
            
            elif adddel == 'del':
                res_dict[adddel][commit]['path'] = path_enc['deletion']
                no_diff_dict[adddel][commit]['path'] = path_enc['deletion']

                for diff_type, diff_enc in diff_enc_dict['deletion'].items():
                    res_dict[adddel][commit][diff_type] = diff_enc
        
    return res_dict, no_diff_dict
    
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
    
    # Encode the data
    start_time = time.time()
    total_intvl, commit_path_dict, commit_msg_dict = load_data(pid, vid)

    encode_data, commit_msg_dict, vocab = pre_encode(pid, vid)
    if encode_data is None or commit_msg_dict is None or vocab is None:
        return

    # Generate features
    res_dict = dict()

    for stage2, stage2_dict in encode_data.items():
        res_dict.setdefault(stage2, dict())

        for setting, setting_dict in stage2_dict.items():
            tracker = dict(setting)['tracker']
            feature_dict, no_diff_dict = gen_feature(setting_dict, commit_msg_dict)

            for adddel, adddel_dict in feature_dict.items():
                res_dict[stage2][frozenset((dict(setting) | {'adddel' : adddel}).items())] = adddel_dict
            
            for adddel, adddel_dict in no_diff_dict.items():
                res_dict[stage2][frozenset({'tracker' : tracker, 'diff_tool' : None, 'adddel' : adddel}.items())] = adddel_dict

    end_time = time.time()
    log('encode', f'{time_to_str(start_time, end_time)}')

    with open(os.path.join(diff_data_dir, 'feature.pkl'), 'wb') as file:
        pickle.dump(res_dict, file)
    with open(os.path.join(diff_data_dir, 'vocab.pkl'), 'wb') as file:
        pickle.dump(vocab, file)