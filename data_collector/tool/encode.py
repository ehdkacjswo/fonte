import os, json, argparse, pickle, sys, itertools, math, copy
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
# 
def pre_encode(total_intvl_dict, commit_path_dict, commit_msg_dict):
    # For identifier handling, each setting needs independent vocabulary
    enc_dict, encoder_dict = dict(), dict()
    log('encode', '[INFO] Encoding data')
    start_time = time.time()

    # Encode identifiers first
    # (To build automaton for ID extraction)
    for commit, adddel_dict in commit_path_dict.items():
        for adddel, path_set in adddel_dict.items():
            target_commit = commit + ('' if adddel == 'addition' else '~1')

            for src_path in path_set:
                code_txt = get_src_from_commit(target_commit, src_path) # Get the code text

                if code_txt is None:
                    log('encode', f'[ERROR] Failed to get file {target_commit}:{src_path}')
                    return None, None, None
                
                for stage2, setting_dict in total_intvl_dict.items():
                    enc_dict.setdefault(stage2, dict())
                    encoder_dict.setdefault(stage2, dict())

                    for setting, commit_dict in setting_dict.items():
                        
                        # Ignore settings that don't consider identifiers
                        if not dict(setting)['diff_type'].endswith('id'): 
                            continue

                        #print(setting)
                        enc_dict[stage2].setdefault(setting, dict())
                        encoder_dict[stage2].setdefault(setting, Encoder())
                        encoder = encoder_dict[stage2][setting]

                        # Setting contains interval data for given commit, adddel, src_path
                        if (commit in commit_dict) and (src_path in commit_dict[commit][adddel]):
                            enc_dict[stage2][setting].setdefault(commit, {'addition' : dict(), 'deletion' : dict()})
                            enc_dict[stage2][setting][commit][adddel][src_path] = dict()
                            
                            # Get the tokens in the given interval
                            token_intvl_dict = commit_dict[commit][adddel][src_path]
                            token_intvl_dict = get_tokens_intvl(code_txt, token_intvl_dict)

                            for diff_type, tokens in token_intvl_dict.items():
                                if diff_type == 'comment' or diff_type == 'non_id': # Encode only identifiers
                                    continue

                                id_vec, non_id_vec = encoder.encode(tokens, update_vocab=True, mode='id')
                                enc_dict[stage2][setting][commit][adddel][src_path][diff_type] = {'id' : id_vec, 'non_id' : non_id_vec}
    
    # Build automaton
    for stage2, setting_dict in encoder_dict.items():
        for setting, encoder in setting_dict.items():
            #print('Building automaton', stage2, setting)
            encoder.init_automaton()
    
    # Get the tokens from interval & encode
    for commit, adddel_dict in commit_path_dict.items():
        for adddel, path_set in adddel_dict.items():
            target_commit = commit + ('' if adddel == 'addition' else '~1')

            for src_path in path_set:
                code_txt = get_src_from_commit(target_commit, src_path) # Get the code text

                if code_txt is None:
                    log('encode', f'[ERROR] Failed to get file {target_commit}:{src_path}')
                    return None, None, None
                
                for stage2, setting_dict in total_intvl_dict.items():
                    for setting, commit_dict in setting_dict.items():
                        enc_dict[stage2].setdefault(setting, dict())
                        encoder_dict[stage2].setdefault(setting, Encoder())
                        encoder = encoder_dict[stage2][setting]

                        # Setting contains interval data for given commit, adddel, src_path
                        if (commit in commit_dict) and (src_path in commit_dict[commit][adddel]):
                            enc_dict[stage2][setting].setdefault(commit, {'addition' : dict(), 'deletion' : dict()})
                            enc_dict[stage2][setting][commit][adddel].setdefault(src_path, dict())
                            
                            # Get the tokens in the given interval
                            token_intvl_dict = commit_dict[commit][adddel][src_path]
                            token_intvl_dict = get_tokens_intvl(code_txt, token_intvl_dict)

                            for diff_type, tokens in token_intvl_dict.items():

                                # Encode source code (Not classified)
                                if diff_type == 'diff':
                                    mode = 'code'
                                
                                # Encode comment, string and annotation (Treated as pure text)
                                elif diff_type == 'comment' or diff_type == 'non_id':
                                    mode = 'text'
                                
                                # Identifiers are already encoded
                                else:
                                    continue
                                    
                                id_vec, non_id_vec = encoder.encode(tokens, update_vocab=True, mode=mode)
                                enc_dict[stage2][setting][commit][adddel][src_path][diff_type] = {'id' : id_vec, 'non_id' : non_id_vec}
                        
                            # Encode path
                            id_vec, non_id_vec = encoder.encode([src_path], update_vocab=True, mode='text')
                            enc_dict[stage2][setting][commit][adddel][src_path]['path'] = {'id' : id_vec, 'non_id' : non_id_vec}

    # Encode commit message 
    enc_commit_msg_dict = dict()

    for stage2, setting_dict in total_intvl_dict.items():
        enc_commit_msg_dict[stage2] = dict()

        for setting, commit_dict in setting_dict.items():
            enc_commit_msg_dict[stage2][setting] = dict()
            encoder = encoder_dict[stage2][setting]

            for commit in commit_dict.keys():
                id_vec, non_id_vec = encoder.encode([commit_msg_dict[commit]], update_vocab=True, mode='text')
                enc_commit_msg_dict[stage2][setting][commit] = {'id' : id_vec, 'non_id' : non_id_vec}
    
    end_time = time.time()
    log('encode', f'[INFO] Elapsed time : {time_to_str(start_time, end_time)}')
    
    return enc_dict, enc_commit_msg_dict, encoder_dict

# Encode the raw diff data
# skip_stage_2 = Excluding style change diff, with_Rewrite = , use_stopword
# Encoded data : {commit_hash : [addition_list, deletion_list, msg_encode]}
# addition/deletion dict : [(before/after_src_path_encode, content_encode_sum)]
# data : message, src_path, diff
# No diff 추가(는 나중에?)
# No diff, no id
def gen_feature(enc_dict, commit_msg_dict, encoder_dict):
    res_dict = dict()
    #log('encode', '[INFO] Generating features')
    #start_time = time.time()

    for stage2, setting_dict in enc_dict.items():
        res_dict.setdefault(stage2, dict())

        for setting, commit_dict in setting_dict.items():
            for commit, adddel_dict in commit_dict.items():
                
                # Aggregate encoded data of commit
                path_set_dict = {'addition' : set(), 'deletion' : set(), 'all' : set()}
                diff_vec_dict = {'addition' : dict(), 'deletion' : dict()}
                path_vec_dict = {'addition' : {'id' : Counter(), 'non_id' : Counter()}, \
                    'deletion' : {'id' : Counter(), 'non_id' : Counter()}, \
                    'all' : {'id' : Counter(), 'non_id' : Counter()}}

                for adddel, path_dict in adddel_dict.items(): # adddel : addition, deletion
                    for src_path, vec_dict in path_dict.items():
                            
                        # Consider only unique paths
                        if src_path not in path_set_dict[adddel]:
                            path_set_dict[adddel].add(src_path)
                            
                            for is_id, vec in vec_dict['path'].items():
                                path_vec_dict[adddel][is_id] += vec

                        if src_path not in path_set_dict['all']:
                            path_set_dict['all'].add(src_path)

                            for is_id, vec in vec_dict['path'].items():
                                path_vec_dict['all'][is_id] += vec
                        
                        # Aggregate encoded data (except path)
                        for diff_type, diff_vec in vec_dict.items():
                            if diff_type != 'path':
                                diff_vec_dict[adddel].setdefault(diff_type, {'id' : Counter(), 'non_id' : Counter()})

                                for is_id, vec in diff_vec.items():
                                    diff_vec_dict[adddel][diff_type][is_id] += vec
                
                # Generate feature
                for adddel in adddel_list: # adddel : add, del, all_uni, all_sep
                    diff_setting = frozenset((dict(setting) | {'adddel' : adddel}).items())
                    no_diff_setting = frozenset({'tracker' : dict(setting)['tracker'], 'diff_tool' : None, 'adddel' : adddel}.items())

                    res_dict[stage2].setdefault(diff_setting, dict())
                    res_dict[stage2].setdefault(no_diff_setting, dict())

                    if res_dict[stage2][no_diff_setting].get(commit, None) is not None:
                        skip_no_diff = True
                    else:
                        encoder_dict[stage2][frozenset({'tracker' : dict(setting)['tracker'], 'diff_tool' : None}.items())] = copy.deepcopy(encoder_dict[stage2][setting])
                        skip_no_diff = False

                    diff_dict, no_diff_dict = res_dict[stage2][diff_setting], res_dict[stage2][no_diff_setting]

                    diff_dict[commit] = {'msg' : commit_msg_dict[stage2][setting].get(commit, {'id' : Counter(), 'non_id' : Counter()})}
                    if not skip_no_diff:
                        no_diff_dict[commit] = {'msg' : commit_msg_dict[stage2][setting].get(commit, {'id' : Counter(), 'non_id' : Counter()})}

                    # Aggregate addition/deletion together
                    if adddel == 'all_uni':
                        diff_dict[commit]['path'] = path_vec_dict['all']
                        if not skip_no_diff:
                            no_diff_dict[commit]['path'] = path_vec_dict['all']

                        for diff_type in diff_vec_dict['addition'].keys() | diff_vec_dict['deletion'].keys():
                            diff_dict[commit].setdefault(diff_type, {'id' : Counter(), 'non_id' : Counter()})

                            for is_id in ['id', 'non_id']:
                                diff_dict[commit][diff_type][is_id] += \
                                    diff_vec_dict['addition'].get(diff_type, dict()).get(is_id, Counter()) + \
                                    diff_vec_dict['deletion'].get(diff_type, dict()).get(is_id, Counter())
                    
                    # sdf
                    elif adddel == 'all_sep':
                        diff_dict[commit]['add_path'] = path_vec_dict['addition']
                        diff_dict[commit]['del_path'] = path_vec_dict['deletion']

                        if not skip_no_diff:
                            no_diff_dict[commit]['add_path'] = path_vec_dict['addition']
                            no_diff_dict[commit]['del_path'] = path_vec_dict['deletion']

                        for diff_type in diff_vec_dict['addition'].keys() | diff_vec_dict['deletion'].keys():
                            diff_dict[commit].setdefault('add_' + diff_type, {'id' : Counter(), 'non_id' : Counter()})
                            diff_dict[commit].setdefault('del_' + diff_type, {'id' : Counter(), 'non_id' : Counter()})

                            for is_id in ['id', 'non_id']:
                                diff_dict[commit]['add_' + diff_type][is_id] += diff_vec_dict['addition'].get(diff_type, dict()).get(is_id, Counter())
                                diff_dict[commit]['del_' + diff_type][is_id] += diff_vec_dict['deletion'].get(diff_type, dict()).get(is_id, Counter())
                    
                    # Use only addition data
                    elif adddel == 'add':
                        diff_dict[commit]['path'] = path_vec_dict['addition']
                        if not skip_no_diff:
                            no_diff_dict[commit]['path'] = path_vec_dict['addition']

                        for diff_type, add_diff_vec in diff_vec_dict['addition'].items():
                            diff_dict[commit].setdefault(diff_type, {'id' : Counter(), 'non_id' : Counter()})

                            for is_id, add_vec in add_diff_vec.items():
                                diff_dict[commit][diff_type][is_id] += add_vec
                    
                    # Use only deletion data
                    elif adddel == 'del':
                        diff_dict[commit]['path'] = path_vec_dict['deletion']
                        if not skip_no_diff:
                            no_diff_dict[commit]['path'] = path_vec_dict['deletion']

                        for diff_type, del_diff_vec in diff_vec_dict['deletion'].items():
                            diff_dict[commit].setdefault(diff_type, {'id' : Counter(), 'non_id' : Counter()})

                            for is_id, del_vec in del_diff_vec.items():
                                diff_dict[commit][diff_type][is_id] += del_vec
        
    #end_time = time.time()
    #log('encode', f'[INFO] Elapsed time : {time_to_str(start_time, end_time)}')
    return res_dict
    
def main(pid, vid):
    log('encode', f'Working on {pid}_{vid}b')

    diff_data_dir = os.path.join(DIFF_DATA_DIR, f'{pid}-{vid}b')
    os.makedirs(diff_data_dir, exist_ok=True)
    
    # Encode the data
    start_time = time.time()
    total_intvl_dict, commit_path_dict, commit_msg_dict = load_data(pid, vid)

    enc_dict, commit_msg_dict, encoder_dict = pre_encode(total_intvl_dict, commit_path_dict, commit_msg_dict)
    if enc_dict is None or commit_msg_dict is None or encoder_dict is None:
        return
    
    #print(commit_msg_dict)
    
    """print('encoded_data')
    for stage2, setting_dict in enc_dict.items():
        print(f'Stage2) {stage2}')

        for setting, commit_dict in setting_dict.items():
            print(f'Setting) {setting}')
            encoder = encoder_dict[stage2][setting]
            id_vocab = encoder.id_vocab.copy()
            non_id_vocab = encoder.non_id_vocab.copy()

            id_vocab = {value : key for key, value in id_vocab.items()}
            non_id_vocab = {value : key for key, value in non_id_vocab.items()}

            for commit, adddel_dict in commit_dict.items():
                print(f'Commit) {commit}')

                for adddel, path_dict in adddel_dict.items():
                    print(f'Adddel) {adddel}')

                    for src_path, diff_dict in path_dict.items():
                        print(f'src_path) {src_path}')

                        for diff_type, sub_dict in diff_dict.items():
                            print(f'Diff type) {diff_type}')

                            print('ID)')
                            for a in sub_dict['id']:
                                print(id_vocab[a], sub_dict['id'][a])
                            
                            print('Non ID)')
                            for a in sub_dict['non_id']:
                                print(non_id_vocab[a], sub_dict['non_id'][a])"""

    # Generate features
    res_dict = gen_feature(enc_dict, commit_msg_dict, encoder_dict)
    
    """print('Feature')
    for stage2, setting_dict in res_dict.items():
        print(f'Stage2) {stage2}')

        for setting, commit_dict in setting_dict.items():
            print(f'Setting) {setting}')
            new_setting = dict(setting)
            del new_setting['adddel']
            
            encoder = encoder_dict[stage2][frozenset(new_setting.items())]
            id_vocab = encoder.id_vocab.copy()
            non_id_vocab = encoder.non_id_vocab.copy()

            id_vocab = {value : key for key, value in id_vocab.items()}
            non_id_vocab = {value : key for key, value in non_id_vocab.items()}

            for commit, diff_dict in commit_dict.items():
                print(f'Commit) {commit}')

                for diff_type, sub_dict in diff_dict.items():
                    print(f'Diff type) {diff_type}')

                    print('ID)')
                    for a in sub_dict['id']:
                        print(id_vocab[a], sub_dict['id'][a])
                    
                    print('Non ID)')
                    for a in sub_dict['non_id']:
                        print(non_id_vocab[a], sub_dict['non_id'][a])"""

    with open(os.path.join(diff_data_dir, 'feature.pkl'), 'wb') as file:
        pickle.dump(res_dict, file)
    with open(os.path.join(diff_data_dir, 'encoder.pkl'), 'wb') as file:
        pickle.dump(encoder_dict, file)