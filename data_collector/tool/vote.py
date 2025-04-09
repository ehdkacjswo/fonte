import os, json, argparse, pickle, sys, subprocess, logging, itertools, time, re
import pandas as pd
import numpy as np
from scipy.spatial.distance import cosine
from collections import Counter

sys.path.append('/root/workspace/data_collector/lib/')
from encoder import Encoder
from vote_util import *
from utils import *
from BM25_Custom import *

DIR_NAME = '/home/coinse/doam/fonte/tmp'
DIFF_DATA_DIR = '/root/workspace/data/Defects4J/diff'
CORE_DATA_DIR = '/root/workspace/data/Defects4J/core'
BASE_DATA_DIR = '/root/workspace/data/Defects4J/baseline'
RESULT_DATA_DIR = '/root/workspace/data/Defects4J/result'
BASELINE_DATA_DIR = "/root/workspace/data/Defects4J/baseline"

use_br_list = [True, False]
use_id_list = [True, False]
classify_id_list = [True, False]

# Bug2Commit with diff info
# 특정 버전에 대하여 vote for commits와 동일한 방법으로 evolve relationship 구축하고
# 해당 
def vote_bug2commit(total_feature_dict, bug_feature_dict):
    log('vote', '[INFO] Bug2Commit voting')
    start_time = time.time()
    res_dict = dict()

    for stage2, setting_dict in total_feature_dict.items():
        res_dict[stage2] = dict()

        for setting, commit_dict in setting_dict.items():
            
            # Get corresponding encoded bug feature
            encoder_setting = dict(setting)
            del encoder_setting['adddel']
            enc_bug_feature_dict = bug_feature_dict[stage2][frozenset(encoder_setting.items())]

            # Get list of commit feature types (Some features may be missing on some commits)
            commit_type_set = set()

            for feature_dict in commit_dict.values():
                commit_type_set |= set(feature_dict.keys())
            
            # Add extra setting (Use bug report or not)
            new_setting_list = [frozenset((dict(setting) | {'use_br' : use_br}).items()) for use_br in use_br_list]

            # For settings using identifires,
            # add extra setting that use full identifier or not
            if 'diff_type' in dict(setting) and dict(setting)['diff_type'].endswith('id'):
                extra_setting_list = itertools.product(new_setting_list, use_id_list)
                new_setting_list = [frozenset((dict(new_setting) | {'use_id' : use_id}).items()) \
                    for (new_setting, use_id) in extra_setting_list]

                # For settings using GumTree based identifier extraction,
                # add extra setting that classifies identifier or not
                if dict(setting)['diff_type'] == 'gumtree_id':
                    extra_setting_list = itertools.product(new_setting_list, classify_id_list)
                    new_setting_list = [frozenset((dict(new_setting) | {'classify_id' : classify_id}).items()) \
                        for (new_setting, classify_id) in extra_setting_list]
                
            # Start voting
            for new_setting in new_setting_list:
                res_dict[stage2][new_setting] = {'all' : list()}
                score_dict = res_dict[stage2][new_setting]

                use_br = dict(new_setting)['use_br']
                use_id = dict(new_setting).get('use_id', False)
                classify_id = dict(new_setting).get('classify_id', False)

                # Need extra handling to ignore ID classes
                id_type_handled = False

                for commit_type in commit_type_set:
                    
                    # ID class handling completed
                    if id_type_handled and commit_type in set(['class', 'method', 'variable']):
                        continue
                    
                    handle_id_type = not classify_id and commit_type in set(['class', 'method', 'variable'])
                    
                    # Build BM25 vocabulary
                    # For the commits that don't have corresponding types, they are currently adding empty docuemnt
                    # 
                    bm25 = BM25_Encode()
                    for commit, feature_dict in commit_dict.items():
                        
                        # Ignore type of IDs
                        if handle_id_type:
                            sub_feature_dict = {'id' : Counter(), 'non_id' : Counter()}

                            # Sum up the ID features
                            # It doesn't work on all_sep...
                            for id_type in ['class', 'method', 'variable']:
                                id_feature_dict = feature_dict.get(id_type, {'id' : Counter(), 'non_id' : Counter()})
                                sub_feature_dict['id'] += id_feature_dict['id']
                                sub_feature_dict['non_id'] += id_feature_dict['non_id']
                        else:    
                            sub_feature_dict = feature_dict.get(commit_type, {'id' : Counter(), 'non_id' : Counter()}) # How should I handle empty type
                        #print(commit_type, sub_feature_dict)

                        # Use full identifier or not
                        if use_id:
                            bm25.add_document(sub_feature_dict['id'] + sub_feature_dict['non_id']) 
                        else:
                            bm25.add_document(sub_feature_dict['non_id'])
                    bm25.init_end()

                    # Vectorize query, commit features & evaluate similarity
                    for bug_type, bug_feature in enc_bug_feature_dict.items():
                        
                        # When not using bug report, ignore bug report data
                        if not use_br and bug_type.startswith('br'):
                            continue

                        # Pair of commit & bug feature type
                        type_setting = frozenset({'commit' : 'id_all' if handle_id_type else commit_type, 'bug' : bug_type}.items())
                        score_dict[type_setting] = list()

                        # Vectorize bug feature
                        bug_vector = bm25.vectorize_complex(\
                            bug_feature['id'] + bug_feature['non_id'] if use_id else bug_feature['non_id'])
                        
                        for commit, feature_dict in commit_dict.items():
                            
                            # Bug feature vector is 0
                            if np.all(bug_vector == 0):
                                score_dict['all'].append([commit, 0])
                                score_dict[type_setting].append([commit, 0])
                                continue
                            
                            # Ignore type of IDs
                            if handle_id_type:
                                sub_feature_dict = {'id' : Counter(), 'non_id' : Counter()}

                                # Sum up the ID features
                                for id_type in ['class', 'method', 'variable']:
                                    id_feature_dict = feature_dict.get(id_type, {'id' : Counter(), 'non_id' : Counter()})
                                    sub_feature_dict['id'] += id_feature_dict['id']
                                    sub_feature_dict['non_id'] += id_feature_dict['non_id']
                            else:    
                                sub_feature_dict = feature_dict.get(commit_type, {'id' : Counter(), 'non_id' : Counter()}) # How should I handle empty type
                            #print(commit_type, sub_feature_dict)

                            # Consider full identifiers or not
                            if use_id:
                                #print(commit, sub_feature_dict['id'] + sub_feature_dict['non_id'])
                                commit_vector = bm25.vectorize_complex(sub_feature_dict['id'] + sub_feature_dict['non_id'])
                            else:
                                #print(commit, sub_feature_dict['non_id'])
                                commit_vector = bm25.vectorize_complex(sub_feature_dict['non_id']) 

                            # Commit feature vector is 0
                            if np.all(commit_vector == 0):
                                score_dict['all'].append([commit, 0])
                                score_dict[type_setting].append([commit, 0])
                            
                            else:
                                score_dict['all'].append([commit, 1 - cosine(bug_vector, commit_vector)])
                                score_dict[type_setting].append([commit, 1 - cosine(bug_vector, commit_vector)])
                    
                    id_type_handled |= handle_id_type
                
                # Format the result as DataFrame
                for type_setting, score_rows in score_dict.items():
                    vote_df = pd.DataFrame(data=score_rows, columns=["commit", "vote"])
                    vote_df = vote_df.groupby("commit").sum()
                    vote_df["rank"] = vote_df["vote"].rank(ascending=False, method="max")
                    vote_df["rank"] = vote_df["rank"].astype(int)
                    vote_df.sort_values(by="rank", inplace=True)
                    score_dict[type_setting] = vote_df

                    #print(res_dict[stage2][new_setting][type_setting])

    end_time = time.time()
    log('vote', f'[INFO] Elapsed time : {time_to_str(start_time, end_time)}')
    return res_dict

# For a given project, generate dataframe with result scores of fonte for every settings
# Settings : ['HSFL', 'score_mode', 'ensemble', 'use_br', 'use_diff', 'stage2', 'use_stopword', 'adddel']
def vote_fonte(pid, vid):
    #log('vote', '[INFO] Fonte voting')
    #start_time = time.time()
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
    
    #end_time = time.time()
    #log('vote', f'[INFO] Elapsed time : {time_to_str(start_time, end_time)}')
    return res_dict

# Ensemble results from Bug2Commit & Fonte
def vote_ensemble(bug2commit_dict, fonte_dict):
    #log('vote', '[INFO] Vote ensembling')
    #start_time = time.time()
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
    
    #end_time = time.time()
    #log('vote', f'[INFO] Elapsed time : {time_to_str(start_time, end_time)}')
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
    
    with open(os.path.join(DIFF_DATA_DIR, f'{pid}-{vid}b', 'bug_feature.pkl'), 'rb') as file:
        bug_feature_dict = pickle.load(file)
    
    """print('Feature')
    for stage2, setting_dict in feature_dict.items():
        print(f'Stage2) {stage2}')
        for setting, sub_dict in setting_dict.items():
            print(f'Setting) {setting}')
            print(json.dumps(sub_dict, indent=4))"""

    # Bug2Commit voting
    bug2commit_vote = vote_bug2commit(feature_dict, bug_feature_dict)

    """print('Vote')
    for stage2, setting_dict in bug2commit_vote.items():
        print(f'Stage2) {stage2}')
        for setting, sub_dict in setting_dict.items():
            print(f'Setting) {setting}')
            print(sub_dict.keys())
            #for aaa, bbb in sub_dict.items():
            #    print(f'Pair) {aaa}')
            #    print(bbb)
            #print(json.dumps(sub_dict, indent=4))
    """

    
    with open(os.path.join(savedir, f'bug2commit.pkl'), 'wb') as file:
        pickle.dump(bug2commit_vote, file)

    # Fonte voting
    fonte_vote = vote_fonte(pid, vid)
    with open(os.path.join(savedir, 'fonte.pkl'), 'wb') as file:
        pickle.dump(fonte_vote, file)

    # Ensemble voting
    with open(os.path.join(savedir, 'ensemble.pkl'), 'wb') as file:
        pickle.dump(vote_ensemble(bug2commit_vote, fonte_vote), file)