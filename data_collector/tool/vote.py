import os, json, argparse, pickle, sys, subprocess, logging, itertools, time
import pandas as pd
import numpy as np
from scipy.spatial.distance import cosine

sys.path.append('/root/workspace/data_collector/lib/')
from encoder import Encoder
from vote_util import *
from utils import log, get_excluded, time_to_str
from BM25_Custom import *

DIFF_DATA_DIR = '/root/workspace/data/Defects4J/diff'
CORE_DATA_DIR = '/root/workspace/data/Defects4J/core'
BASE_DATA_DIR = '/root/workspace/data/Defects4J/baseline'
RESULT_DATA_DIR = '/root/workspace/data/Defects4J/result'
BASELINE_DATA_DIR = "/root/workspace/data/Defects4J/baseline"

# Bug2Commit with diff info
# 특정 버전에 대하여 vote for commits와 동일한 방법으로 evolve relationship 구축하고
# 해당 
def vote_bug2commit(pid, vid, feature_dict, bug_dict):
    # Get list of commits/feature types
    commit_type_set = set()

    for commit_dict in feature_dict.values():
        commit_type_set |= set(commit_dict.keys())
        
    # Bug2Commit scoring
    res_dict = {'all' : list()}

    for commit_type in commit_type_set:
        
        # Build BM25 vocabulary
        bm25 = BM25_Encode()
        for commit, commit_dict in feature_dict.items():
            bm25.add_document(commit_dict.get(commit_type, [])) #???
        bm25.init_end()

        # Vectorize query, commit features & evaluate similarity
        for bug_type, bug_feature in bug_dict.items():
            type_setting = frozenset({'commit' : commit_type, 'bug' : bug_type}.items())
            res_dict[type_setting] = list()

            bug_vector = bm25.vectorize_complex(bug_feature)
            
            for commit, commit_dict in feature_dict.items():
                if np.all(bug_vector == 0):
                    res_dict['all'].append([commit, 0])
                    res_dict[type_setting].append([commit, 0])
                    continue

                commit_vector = bm25.vectorize_complex(commit_dict.get(commit_type, []))

                if np.all(commit_vector == 0):
                    res_dict['all'].append([commit, 0])
                    res_dict[type_setting].append([commit, 0])
                
                else:
                    res_dict['all'].append([commit, 1 - cosine(bug_vector, commit_vector)])
                    res_dict[type_setting].append([commit, 1 - cosine(bug_vector, commit_vector)])
    
    # Format the result as DataFrame
    for type_setting, score_rows in res_dict.items():
        vote_df = pd.DataFrame(data=score_rows, columns=["commit", "vote"])
        vote_df = vote_df.groupby("commit").sum()
        vote_df["rank"] = vote_df["vote"].rank(ascending=False, method="max")
        vote_df["rank"] = vote_df["rank"].astype(int)
        vote_df.sort_values(by="rank", inplace=True)
        res_dict[type_setting] = vote_df

    return res_dict

# For a given project, generate dataframe with result scores of fonte for every settings
# Settings : ['HSFL', 'score_mode', 'ensemble', 'use_br', 'use_diff', 'stage2', 'use_stopword', 'adddel']
def vote_fonte(pid, vid):
    fault_dir = os.path.join(CORE_DATA_DIR, f'{pid}-{vid}b')
    res_dict = dict()

    for stage2 in ['skip', 'precise']:
        excluded = get_style_change_commits(fault_dir, tool='git', stage2=stage2)
            
        fonte_df = vote_for_commits(fault_dir, tool='git', formula='Ochiai', decay=0.1, \
            voting_func=(lambda r: 1/r.max_rank), excluded=excluded, adjust_depth=True)
        
        # Add ranking
        fonte_df["rank"] = fonte_df["vote"].rank(ascending=False, method="max")
        fonte_df["rank"] = fonte_df["rank"].astype(int)

        res_dict[stage2] = fonte_df
    
    return res_dict

# Ensemble results from Bug2Commit & Fonte
def vote_ensemble(bug2commit_dict, fonte_dict):
    res_dict = dict()
    
    for stage2, sub_dict in bug2commit_dict.items():
        fonte_df = fonte_dict[stage2]
        res_dict[stage2] = dict()

        for setting, bug2commit_df in sub_dict.items():
            merged_df = fonte_df.merge(bug2commit_df['all'], on='commit', how='left', suffixes=('_fonte', '_bug2commit'))
            merged_df['vote_bug2commit'].fillna(0, inplace=True) # 

            for beta in [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]:
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
    log('vote', f'Working on {pid}_{vid}b')

    savedir = os.path.join(RESULT_DATA_DIR, f'{pid}-{vid}b', 'vote')
    os.makedirs(savedir, exist_ok=True)

    """# Load the previous result if possible
    bug2commit_save_path = os.path.join(savedir, f'bug2commit.pkl')

    if os.path.isfile(bug2commit_save_path):
        with open(bug2commit_save_path, 'rb') as file:
            res_dict = pickle.load(file)
    
    else:
        res_dict = dict()"""

    # Load feature & vocab for the project
    start_time = time.time()
    with open(os.path.join(DIFF_DATA_DIR, f'{pid}-{vid}b', 'feature.pkl'), 'rb') as file:
        feature_dict = pickle.load(file)
    
    with open(os.path.join(DIFF_DATA_DIR, f'{pid}-{vid}b', 'vocab.pkl'), 'rb') as file:
        vocab = pickle.load(file)
    
    # Encode bug features
    encoder = Encoder(vocab)
    bug_feature = dict()

    with open(os.path.join(CORE_DATA_DIR, f'{pid}-{vid}b', "failing_tests"), "r") as f:
        bug_feature['failing_test'] = encoder.encode(f.read().strip(), update_vocab=False)
    with open(os.path.join(BASELINE_DATA_DIR, f'{pid}-{vid}b', "br_long.txt"), "r") as f:
        bug_feature['br_long'] = encoder.encode(f.read().strip(), update_vocab=False) # 1st bug report feature
    with open(os.path.join(BASELINE_DATA_DIR, f'{pid}-{vid}b', "br_short.txt"), "r") as f:
        bug_feature['br_short'] = encoder.encode(f.read().strip(), update_vocab=False) # 2nd bug report feature

    # Bug2Commit voting
    bug2commit_vote = dict()

    for stage2, stage2_dict in feature_dict.items():
        bug2commit_vote[stage2] = bug2commit_vote.get(stage2, dict())

        for setting, setting_dict in stage2_dict.items():
            for use_br in [True, False]:
                bug2commit_vote[stage2][frozenset((dict(setting) | {'use_br' : use_br}).items())] = vote_bug2commit(pid, vid, setting_dict, bug_feature if use_br else {'failing_test' : bug_feature['failing_test']})
    
    with open(os.path.join(savedir, f'bug2commit.pkl'), 'wb') as file:
        pickle.dump(bug2commit_vote, file)

    # Fonte voting
    fonte_vote = vote_fonte(pid, vid)
    with open(os.path.join(savedir, 'fonte.pkl'), 'wb') as file:
        pickle.dump(fonte_vote, file)

    # Ensemble voting
    with open(os.path.join(savedir, 'ensemble.pkl'), 'wb') as file:
        pickle.dump(vote_ensemble(bug2commit_vote, fonte_vote), file)

    end_time = time.time()
    log('vote', f'{time_to_str(start_time, end_time)}')