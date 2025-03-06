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

# Get style change data list [(commit, before_src_path, after_src_path)]
def get_excluded(coredir, tool='git', stage2='skip'):
    if stage2 == 'skip':
        return []

    elif stage2 == 'precise':
        val_df = pd.read_csv(
            os.path.join(coredir, tool, f"precise_validation_noOpenRewrite.csv"), 
            header=None,
            names=["commit", "before_src_path", "after_src_path", "AST_diff"])

        unchanged_df = val_df[val_df["AST_diff"] == "U"]
        return list(zip(unchanged_df["commit"], unchanged_df["before_src_path"], unchanged_df["after_src_path"]))

# Code txt could be possibly "None" (Failed to get code data)
def get_tokens_intvl(commit, src_path, intvl_dict):
    
    # Convert index from interval to integer
    def convert_ind(ind):
        if ind == -inf:
            return 0
        if ind == inf:
            return len(code_txt)
        
        return math.floor(ind) + 1
    
    # Get code text
    code_txt = get_src_from_commit(commit, src_path)

    # Get tokens from code
    res_dict = dict()
    for diff_type, intvl in intvl_dict.items():
        if intvl.is_empty(): # Empty interval
            res_dict[diff_type] = []
        if code_txt is None:
            log('encode', f'[ERROR] Failed to get file {commit}:{src_path}')
            return None
        
        res_dict[diff_type] = [''.join(code_txt[convert_ind(sub_intvl[0]) : convert_ind(sub_intvl[1])]) for sub_intvl in intvl]

    return res_dict

# Encode the data 
# Add file
def pre_encode(pid, vid):
    with open(os.path.join(os.path.join(DIFF_DATA_DIR, f'{pid}-{vid}b', 'total_intvl.pkl')), 'rb') as file:
        intvl_dict = pickle.load(file)

    base_data_dir = os.path.join(BASE_DATA_DIR, f'{pid}-{vid}b', 'commits')
    commit_msg_dict = dict()
    encoder = Encoder()

    for stage2, stage2_dict in intvl_dict.items():
        for setting, setting_dict in stage2_dict.items():
            for commit, commit_dict in setting_dict.items():
                
                # Get/Encode commit message
                for filename in os.listdir(base_data_dir):
                    if filename.startswith(f'c_{commit}'):
                        with open(os.path.join(base_data_dir, filename), "r") as file:
                            data = json.load(file)
                        commit_msg_dict[commit] = encoder.encode(data['log'], use_stopword=True, update_vocab=True)
                        break

                for adddel, adddel_dict in commit_dict.items(): # Get/Encode commit featues
                    for src_path, src_intvl_dict in adddel_dict.items():
                        # Get the tokens in the list
                        token_dict = get_tokens_intvl(commit + ('' if adddel == 'addition' else '~1'), src_path, src_intvl_dict)
                        
                        if token_dict is None:
                            return None, None, None

                        for diff_type, tokens in token_dict.items():
                            diff_enc = []

                            for token in tokens:
                                diff_enc = sum_encode(diff_enc, encoder.encode(token))
                            
                            src_intvl_dict[diff_type] = diff_enc
                        
                        src_intvl_dict['enc_path'] = (encoder.encode(src_path))
    
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
                    path_enc[adddel] = sum_encode(path_enc[adddel], src_enc_dict['enc_path'])

                if src_path not in path_set['all']:
                    path_set['all'].add(src_path)
                    path_enc['all'] = sum_encode(path_enc['all'], src_enc_dict['enc_path'])
                
                # Aggregate encoded data (except path)
                for diff_type, diff_enc in src_enc_dict.items():
                    if diff_type != 'enc_path':
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