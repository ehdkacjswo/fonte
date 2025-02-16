import os, json, argparse, pickle, sys, subprocess, logging, itertools
import pandas as pd
import numpy as np
from scipy.spatial.distance import cosine

sys.path.append('/root/workspace/data_collector/lib/')
from BM25_Custom import BM25_Encode
from gumtree import *
from encoder import *
from vote_util import *

DIFF_DATA_DIR = '/root/workspace/data/Defects4J/diff'
CORE_DATA_DIR = '/root/workspace/data/Defects4J/core'
BASE_DATA_DIR = '/root/workspace/data/Defects4J/baseline'
RESULT_DATA_DIR = '/root/workspace/data/Defects4J/result'
BASELINE_DATA_DIR = "/root/workspace/data/Defects4J/baseline"

def log(txt, out_txt=None, err_txt=None):
    with open('/root/workspace/data_collector/log/vote.log', 'a') as file:
        file.write(txt + '\n')

        if out_txt is not None:
            file.write('[ERROR] OUT\n' + out_txt.decode(encoding='utf-8', errors='ignore') + '\n')
        
        if err_txt is not None:
            file.write('[ERROR] ERR\n' + err_txt.decode(encoding='utf-8', errors='ignore') + '\n')

# Bug2Commit with diff info
# 특정 버전에 대하여 vote for commits와 동일한 방법으로 evolve relationship 구축하고
# 해당 
def vote_bug2commit(pid, vid, feature_data, setting_dict, stage2):
    core_data_dir = os.path.join(CORE_DATA_DIR, f"{pid}-{vid}b")
    diff_data_dir = os.path.join(DIFF_DATA_DIR, f"{pid}-{vid}b")
    baseline_data_dir = os.path.join(BASELINE_DATA_DIR, f"{pid}-{vid}b")
    commit_dir = os.path.join(baseline_data_dir, "commits")

    # Retreive settings
    diff_type = setting_dict['diff_type']
    use_stopword = setting_dict['use_stopword']
    adddel = setting_dict['adddel']
    use_br = setting_dict['use_br']
    
    # Load vocab and build encoder
    with open(os.path.join(diff_data_dir, f'vocab.pkl'), 'rb') as file:
        vocab_dict = pickle.load(file)

    encoder = Encoder(vocab_dict[stage2][frozenset({'diff_type' : 'git' if diff_type == 'no_diff' else diff_type, \
        'use_stopword' : use_stopword}.items())])
    
    # Encode query features
    query_feature = []

    with open(os.path.join(core_data_dir, "failing_tests"), "r") as f:
        query_feature.append(encoder.encode(f.read().strip(), use_stopword, update_vocab=False))

    if use_br:
        with open(os.path.join(baseline_data_dir, "br_long.txt"), "r") as f:
            query_feature.append(encoder.encode(f.read().strip(), use_stopword, update_vocab=False)) # 1st bug report feature
        with open(os.path.join(baseline_data_dir, "br_short.txt"), "r") as f:
            query_feature.append(encoder.encode(f.read().strip(), use_stopword, update_vocab=False)) # 2nd bug report feature

    # For target list of commits get used 
    if diff_type == 'no_diff': # Commit message + Source path
        num_commit_feature = 3 if adddel == 'all-sep' else 2
    else:
        num_commit_feature = len(next(iter(feature_data.values()), []))
    
    score_rows = []
    
    for i in range(num_commit_feature):
        # Build BM25 vocabulary
        bm25 = BM25_Encode()
        for commit_hash, commit_feature in feature_data.items():
            bm25.add_document(commit_feature[i])
        bm25.init_end()

        # Vectorize query, commit features & evaluate similarity
        query_vector = bm25.vectorize_complex(query_feature)
        if np.all(query_vector == 0):
            score_rows.append([commit_hash, 0])
            continue

        for commit_hash, commit_feature in feature_data.items():
            commit_vector = bm25.vectorize_complex([commit_feature[i]])

            if np.all(commit_vector == 0):
                score_rows.append([commit_hash, 0])
                continue
            
            else:
                score_rows.append([commit_hash, 1 - cosine(query_vector, commit_vector)])
    
    # Format the result as DataFrame
    score_df = pd.DataFrame(data=score_rows, columns=["commit", "vote"])
    score_df = score_df.groupby("commit").sum()
    score_df["rank"] = score_df["vote"].rank(ascending=False, method="max")
    score_df["rank"] = score_df["rank"].astype(int)
    score_df.sort_values(by="rank", inplace=True)

    return score_df

# For a given project, generate dataframe with result scores of fonte for every settings
# Settings : ['HSFL', 'score_mode', 'ensemble', 'use_br', 'use_diff', 'stage2', 'use_stopword', 'adddel']
def vote_fonte(pid, vid):
    fault_dir = os.path.join(CORE_DATA_DIR, f'{pid}-{vid}b')
    res_dict = dict()

    for stage2 in ['skip', 'precise']:
        excluded = get_style_change_commits(fault_dir, tool='git', stage2=stage2)
            
        fonte_df = vote_for_commits(fault_dir, tool='git', formula='Ochiai', decay=0.1, \
            voting_func=(lambda r: 1/r.max_rank), use_method_level_score=False, excluded=excluded, adjust_depth=True)
        
        # Add ranking
        fonte_df["rank"] = fonte_df["vote"].rank(ascending=False, method="max")
        fonte_df["rank"] = fonte_df["rank"].astype(int)

        res_dict[stage2] = fonte_df
    
    return res_dict

def vote_ensemble(pid, vid):
    # Load voting results
    with open(os.path.join(RESULT_DATA_DIR, f'{pid}-{vid}b', 'vote', 'bug2commit.pkl'), 'rb') as file:
        bug2commit_dict = pickle.load(file)
    
    with open(os.path.join(RESULT_DATA_DIR, f'{pid}-{vid}b', 'vote', 'fonte.pkl'), 'rb') as file:
        fonte_dict = pickle.load(file)
    
    res_dict = dict()
    
    for stage2, sub_dict in bug2commit_dict.items():
        fonte_df = fonte_dict[stage2]
        res_dict[stage2] = dict()

        for setting, bug2commit_df in sub_dict.items():
            merged_df = fonte_df.merge(bug2commit_df, on='commit', how='left', suffixes=('_fonte', '_bug2commit'))
            merged_df['vote_bug2commit'].fillna(0, inplace=True)

            for beta in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]:
                merged_df['vote'] = merged_df['vote_fonte'] * (1 + beta * merged_df['vote_bug2commit'])
                result_df = merged_df[['vote']].copy()

                result_df["rank"] = result_df["vote"].rank(ascending=False, method="max")
                result_df["rank"] = result_df["rank"].astype(int)
                result_df = result_df.sort_values(by="rank")

                # Update setting with beta
                new_setting = frozenset((dict(setting) | {'beta' : beta}).items())
                res_dict[stage2][new_setting] = result_df
    
    return res_dict

def main(pid, vid):
    log(f'Working on {pid}_{vid}b')

    savedir = os.path.join(RESULT_DATA_DIR, f'{pid}-{vid}b', 'vote')
    os.makedirs(savedir, exist_ok=True)

    # Load the previous result if possible
    bug2commit_save_path = os.path.join(savedir, f'bug2commit.pkl')

    if os.path.isfile(bug2commit_save_path):
        with open(bug2commit_save_path, 'rb') as file:
            res_dict = pickle.load(file)
    
    else:
        res_dict = dict()

    # Bug2Commit
    res_dict = dict()
    use_br_list = [True, False]
    use_diff_list = [True, False]
    param_list = list(itertools.product(use_br_list, use_diff_list))

    with open(os.path.join(DIFF_DATA_DIR, f'{pid}-{vid}b', 'feature.pkl'), 'rb') as file:
        feature_dict = pickle.load(file)

    for stage2, sub_dict in feature_dict.items():
        res_dict[stage2] = res_dict.get(stage2, dict())

        for setting, feature_data in sub_dict.items():
            setting_dict = dict(setting)

            for (use_br, use_diff) in param_list:
                diff_type = setting_dict['diff_type'] if use_diff else 'no_diff'

                new_setting_dict = setting_dict | {'diff_type' : diff_type, 'use_br' : use_br}
                new_setting = frozenset(new_setting_dict.items())
                
                if new_setting not in res_dict[stage2]:
                    res_dict[stage2][new_setting] = vote_bug2commit(pid, vid, feature_data=feature_data, setting_dict=new_setting_dict, stage2=stage2)

    with open(bug2commit_save_path, 'wb') as file:
        pickle.dump(res_dict, file)

    #with open(os.path.join(savedir, 'fonte.pkl'), 'wb') as file:
    #    pickle.dump(vote_fonte(pid, vid), file)

    with open(os.path.join(savedir, 'ensemble.pkl'), 'wb') as file:
        pickle.dump(vote_ensemble(pid, vid), file)