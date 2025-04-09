import os, json, argparse, pickle, sys, itertools, math, copy
import pandas as pd
from tqdm import tqdm

sys.path.append('/root/workspace/data_collector/lib/')
from encoder import *
from utils import *

DIFF_DATA_DIR = '/root/workspace/data/Defects4J/diff'
CORE_DATA_DIR = '/root/workspace/data/Defects4J/core'
BASE_DATA_DIR = '/root/workspace/data/Defects4J/baseline'

# Decided to use all_uni as default
#adddel_list = ['add', 'del', 'all_uni', 'all_sep']
classfy_id_list = [True, False]

# Returns total_intvl, {commit : {adddel : src_path} }, {commit : commit message}, encoder
def load_data(pid, vid):
    # Load total interval data
    with open(os.path.join(os.path.join(DIFF_DATA_DIR, f'{pid}-{vid}b', 'total_intvl.pkl')), 'rb') as file:
        total_intvl_dict = pickle.load(file)

    # Get src paths related to commits (For faster encoding)
    # {commit : {adddel : set(src_path)} }
    commit_path_dict = dict()

    for setting_dict in total_intvl_dict.values():
        for commit_dict in setting_dict.values():
            for commit, adddel_dict in commit_dict.items():
                commit_path_dict.setdefault(commit, {'addition' : set(), 'deletion' : set()})

                for adddel, path_dict in adddel_dict.items():
                    for src_path in path_dict.keys():
                        commit_path_dict[commit][adddel].add(src_path)
    
    # Get commit messages
    # {commit : message}
    commit_msg_dict = dict() 
    base_data_dir = os.path.join(BASE_DATA_DIR, f'{pid}-{vid}b', 'commits')

    for commit in commit_path_dict.keys():
        for filename in os.listdir(base_data_dir):
            if filename.startswith(f'c_{commit}'):
                with open(os.path.join(base_data_dir, filename), "r") as file:
                    data = json.load(file)
                    
                commit_msg_dict[commit] = data['log']
                break

        if commit not in commit_msg_dict:    
            log('encode', f'[WARNING] No commit message for {commit}')
    
    return total_intvl_dict, commit_path_dict, commit_msg_dict


# Encode the data 
# Add file
# {commit : {feature_type : feature}}
def encode(total_intvl_dict, commit_path_dict, commit_msg_dict):
    # For identifier handling, each setting needs independent vocabulary
    log('encode', '[INFO] Encoding data')
    start_time = time.time()

    enc_dict, encoder_dict = dict(), dict()

    # Encode identifiers first
    # (To build automaton for ID extraction)
    for commit, adddel_dict in commit_path_dict.items():
        for adddel, path_set in adddel_dict.items(): # adddel : addition / deletion
            target_commit = commit + ('' if adddel == 'addition' else '~1')

            for src_path in path_set:
                code_txt = get_src_from_commit(target_commit, src_path) # Get the code text

                if code_txt is None:
                    log('encode', f'[ERROR] Failed to get file {target_commit}:{src_path}')
                    return None, None
                
                for stage2, setting_dict in total_intvl_dict.items():
                    enc_dict.setdefault(stage2, dict())
                    encoder_dict.setdefault(stage2, dict())

                    for setting, commit_dict in setting_dict.items():
                        
                        # Ignore settings that don't consider identifiers
                        if not dict(setting)['diff_type'].endswith('id'): 
                            continue

                        enc_dict[stage2].setdefault(setting, dict())
                        encoder_dict[stage2].setdefault(setting, Encoder())
                        encoder = encoder_dict[stage2][setting]

                        # Setting contains interval data for given commit, adddel, src_path
                        if (commit in commit_dict) and (src_path in commit_dict[commit][adddel]):
                            enc_dict[stage2][setting].setdefault(commit, dict())
                            
                            # Get the tokens in the given interval
                            token_intvl_dict = commit_dict[commit][adddel][src_path]
                            token_dict = get_tokens_intvl(code_txt, token_intvl_dict)

                            for feature_type, tokens in token_dict.items():
                                if feature_type == 'comment' or feature_type == 'non_id': # Encode only identifiers
                                    continue

                                id_vec, non_id_vec = encoder.encode(tokens, update_vocab=True, mode='id')

                                if feature_type in enc_dict[stage2][setting][commit]:
                                    enc_dict[stage2][setting][commit][feature_type]['id'] += id_vec
                                    enc_dict[stage2][setting][commit][feature_type]['non_id'] += non_id_vec
                                else:
                                    enc_dict[stage2][setting][commit][feature_type] = {'id' : id_vec, 'non_id' : non_id_vec}
    
    # Build automaton
    for stage2, setting_dict in encoder_dict.items():
        for setting, encoder in setting_dict.items():
            encoder.make_automaton()
    
    # Get the tokens from interval & encode
    for commit, adddel_dict in commit_path_dict.items():
        for adddel, path_set in adddel_dict.items():
            target_commit = commit + ('' if adddel == 'addition' else '~1')

            for src_path in path_set:
                code_txt = get_src_from_commit(target_commit, src_path) # Get the code text

                if code_txt is None:
                    log('encode', f'[ERROR] Failed to get file {target_commit}:{src_path}')
                    return None, None
                
                for stage2, setting_dict in total_intvl_dict.items():
                    for setting, commit_dict in setting_dict.items():
                        enc_dict[stage2].setdefault(setting, dict())
                        encoder_dict[stage2].setdefault(setting, Encoder())
                        encoder = encoder_dict[stage2][setting]

                        # Setting contains interval data for given commit, adddel, src_path
                        if (commit in commit_dict) and (src_path in commit_dict[commit][adddel]):
                            enc_dict[stage2][setting].setdefault(commit, dict())
                            
                            # Get the tokens in the given interval
                            token_intvl_dict = commit_dict[commit][adddel][src_path]
                            token_dict = get_tokens_intvl(code_txt, token_intvl_dict)

                            for feature_type, tokens in token_dict.items():

                                # Encode source code (Not classified)
                                if feature_type == 'diff':
                                    mode = 'code'
                                
                                # Encode comment, string and annotation (Treated as pure text)
                                elif feature_type == 'comment' or feature_type == 'non_id':
                                    mode = 'text'
                                
                                # Identifiers are already encoded
                                else:
                                    continue
                                    
                                id_vec, non_id_vec = encoder.encode(tokens, update_vocab=True, mode=mode)

                                if feature_type in enc_dict[stage2][setting][commit]:
                                    enc_dict[stage2][setting][commit][feature_type]['id'] += id_vec
                                    enc_dict[stage2][setting][commit][feature_type]['non_id'] += non_id_vec
                                else:
                                    enc_dict[stage2][setting][commit][feature_type] = {'id' : id_vec, 'non_id' : non_id_vec}
                        
                            # Encode path
                            id_vec, non_id_vec = encoder.encode([src_path], update_vocab=True, mode=mode)
                                
                            if 'path' in enc_dict[stage2][setting][commit]:
                                enc_dict[stage2][setting][commit]['path']['id'] += id_vec
                                enc_dict[stage2][setting][commit]['path']['non_id'] += non_id_vec
                            else:
                                enc_dict[stage2][setting][commit]['path'] = {'id' : id_vec, 'non_id' : non_id_vec}

    # Encode commit message
    for stage2, setting_dict in total_intvl_dict.items():
        for setting, commit_dict in setting_dict.items():
            encoder = encoder_dict[stage2][setting]

            for commit in commit_dict.keys():
                id_vec, non_id_vec = encoder.encode([commit_msg_dict[commit]], update_vocab=True, mode='text')
                enc_dict[stage2][setting][commit]['msg'] = {'id' : id_vec, 'non_id' : non_id_vec}
    
    # Classify ID or not
    for stage2, setting_dict in enc_dict.items():
        del_setting_list, new_setting_dict = list(), dict()

        for setting, commit_dict in setting_dict.items():
            
            # Only GumTree Identifiers can classify IDs
            if dict(setting)['diff_type'] != 'gumtree_id':
                continue
            
            del_setting_list.append(setting)
            new_setting_dict[frozenset((dict(setting) | {'classify_id' : True}).items())] = \
                {'encoder' : copy.deepcopy(encoder_dict[stage2][setting]), 'commit_dict' : copy.deepcopy(commit_dict)}
            
            # Aggregate ID types
            new_setting = frozenset((dict(setting) | {'classify_id' : False}).items())
            new_commit_dict = copy.deepcopy(commit_dict)
            
            for commit, feature_dict in new_commit_dict.items():
                id_vec = feature_dict['class']['id'] + feature_dict['method']['id'] + feature_dict['variable']['id']
                non_id_vec = feature_dict['class']['non_id'] + feature_dict['method']['non_id'] + feature_dict['variable']['non_id']

                # Delete classes
                del feature_dict['class']
                del feature_dict['method']
                del feature_dict['variable']

                feature_dict['id'] = {'id' : id_vec, 'non_id' : non_id_vec}

            new_setting_dict[new_setting] = {'encoder' : copy.deepcopy(encoder_dict[stage2][setting]), 'commit_dict' : new_commit_dict}

        # Delete old settings & Add new settings
        for del_setting in del_setting_list:
            del setting_dict[del_setting]
            del (encoder_dict[stage2])[del_setting]
        
        for new_setting, new_data_dict in new_setting_dict.items():
            setting_dict[new_setting] = new_data_dict['commit_dict']
            encoder_dict[stage2][new_setting] = new_data_dict['encoder']
    
    # Settings not using diff
    for stage2, setting_dict in enc_dict.items():
        no_diff_dict = dict()

        for setting, commit_dict in setting_dict.items():
            
            # diff_tool & diff_type are not required
            no_diff_setting = dict(setting) | {'diff_tool' : None}
            del no_diff_setting['diff_type']
            no_diff_setting.pop('classify_id', None)
            no_diff_setting = frozenset(no_diff_setting.items())

            # Setting already visited
            if no_diff_setting in no_diff_dict:
                continue
            
            no_diff_dict[no_diff_setting] = dict()
            for commit, feature_dict in commit_dict.items():
                no_diff_dict[no_diff_setting][commit] = dict()

                # Consider only path & commit message
                for feature_type in ['path', 'msg']:
                    non_id_vec = feature_dict[feature_type]['non_id']
                    no_diff_dict[no_diff_setting][commit][feature_type] = {'id' : Counter(), 'non_id' : non_id_vec}

            encoder_dict[stage2][no_diff_setting] = encoder_dict[stage2][setting]
        
        enc_dict[stage2] |= no_diff_dict
    
    end_time = time.time()
    log('encode', f'[INFO] Elapsed time : {time_to_str(start_time, end_time)}')
    
    return enc_dict, encoder_dict
    
def main(pid, vid):
    log('encode', f'Working on {pid}_{vid}b')

    diff_data_dir = os.path.join(DIFF_DATA_DIR, f'{pid}-{vid}b')
    os.makedirs(diff_data_dir, exist_ok=True)
    
    # Encode the data
    start_time = time.time()
    total_intvl_dict, commit_path_dict, commit_msg_dict = load_data(pid, vid)

    enc_dict, encoder_dict = encode(total_intvl_dict, commit_path_dict, commit_msg_dict)
    if enc_dict is None or commit_msg_dict is None or encoder_dict is None:
        return
    
    """print('Feature')
    for stage2, setting_dict in enc_dict.items():
        print(f'Stage2) {stage2}')

        for setting, commit_dict in setting_dict.items():
            print(f'Setting) {setting}')
            
            encoder = encoder_dict[stage2][setting]
            vocab = encoder.id_vocab | encoder.non_id_vocab

            vocab = {value : key for key, value in vocab.items()}

            for commit, diff_dict in commit_dict.items():
                print(f'Commit) {commit}')

                for diff_type, sub_dict in diff_dict.items():
                    print(f'Diff type) {diff_type}')

                    print('ID)')
                    for a in sub_dict['id']:
                        print(vocab[a], sub_dict['id'][a])
                    
                    print('Non ID)')
                    for a in sub_dict['non_id']:
                        print(vocab[a], sub_dict['non_id'][a])"""

    #print(enc_dict)
    with open(os.path.join(diff_data_dir, 'feature.pkl'), 'wb') as file:
        pickle.dump(enc_dict, file)
    with open(os.path.join(diff_data_dir, 'encoder.pkl'), 'wb') as file:
        pickle.dump(encoder_dict, file)