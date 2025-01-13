import os, sys, itertools
import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine
from lib.experiment_utils import *

from BM25_Custom import BM25_Encode
from tqdm import tqdm

import pickle

CORE_DATA_DIR = "./data/Defects4J/core"
BIC_GT_DIR = "./data/Defects4J/BIC_dataset"
BASELINE_DATA_DIR = "./data/Defects4J/baseline"
DIFF_DATA_DIR = './data/Defects4J/diff'

sys.path.append('/root/workspace/diff_util/lib/')
from diff import *
from encoder import *

# Bug2Commit with diff info
# 특정 버전에 대하여 vote for commits와 동일한 방법으로 evolve relationship 구축하고
# 해당 
def run_bug2commit(pid, vid, use_br, use_diff, stage2, use_stopword, adddel):
    core_data_dir = os.path.join(CORE_DATA_DIR, f"{pid}-{vid}b")
    diff_data_dir = os.path.join(DIFF_DATA_DIR, f"{pid}-{vid}b")
    baseline_data_dir = os.path.join(BASELINE_DATA_DIR, f"{pid}-{vid}b")
    commit_dir = os.path.join(baseline_data_dir, "commits")
    
    #print(f"{pid}-{vid}b: Encode query feature and build BM25..........................")
    # Load vocab and build encoder
    with open(os.path.join(diff_data_dir, f'encode/vocab.pkl'), 'rb') as file:
        vocab_dict = pickle.load(file)

    encoder = Encoder(vocab_dict[(stage2, use_stopword)])
    bm25 = BM25_Encode()

    # Get query features (failing tests)
    query_features = []
    with open(os.path.join(core_data_dir, "failing_tests"), "r") as f:
        query_features.append(f.read().strip())
    if use_br:
        with open(os.path.join(baseline_data_dir, "br_long.txt"), "r") as f:
            query_features.append(f.read().strip()) # 1st bug report feature
        with open(os.path.join(baseline_data_dir, "br_short.txt"), "r") as f:
            query_features.append(f.read().strip()) # 2nd bug report feature
    
    # Encode the query feature
    encoded_query_features = []
    for feature in query_features:
        encoded_feature = encoder.encode(feature, use_stopword, update_vocab=False)
        encoded_query_features.append(encoded_feature)
        #bm25.add_document(encoded_feature)

    # For target list of commits get used set
    with open(os.path.join(diff_data_dir, 'feature/feature.pkl'), 'rb') as file:
        commit_feature_dict_total = pickle.load(file)
    
    commit_feature_dict = commit_feature_dict_total[(stage2, use_stopword, adddel)]
    filtered_commit_feature_dict = dict()
    
    for commit_hash, commit_feature_list in commit_feature_dict.items():
        if not use_diff: # Filter out diff data
            if adddel == 'all-sep':
                filtered_commit_feature_dict[commit_hash] = commit_feature_list[0:3]
            else:
                filtered_commit_feature_dict[commit_hash] = commit_feature_list[0:2]
        else:
            filtered_commit_feature_dict[commit_hash] = commit_feature_list

        for feature in filtered_commit_feature_dict[commit_hash]:
            bm25.add_document(feature)
    
    bm25.init_end()
    
    #print(f"{pid}-{vid}b: Vectorizing features..........................")
    # Handle query features
    query_vector = bm25.vectorize_complex(encoded_query_features)
    if np.all(query_vector == 0):
        with open('/root/workspace/eror.txt', 'a') as file:
                file.write(f'Query vector of {pid}-{vid}b is zero {use_br}:{use_diff}:{adddel}\n')

    # Handle commit features
    for commit_hash in filtered_commit_feature_dict.keys():
        filtered_commit_feature_dict[commit_hash] = bm25.vectorize_complex(filtered_commit_feature_dict[commit_hash])

        if np.all(commit_feature_dict[commit_hash] == 0):
            with open('/root/workspace/eror.txt', 'a') as file:
                file.write(f'Commit vector of {pid}-{vid}b:{commit_hash} is zero {use_br}:{use_diff}:{adddel}\n')

    #print(f"{pid}-{vid}b: Calculating the scores of commits...................")
    score_rows = []
    for commit_hash, vector in filtered_commit_feature_dict.items():
        if np.all(query_vector == 0) or np.all(vector == 0):
            similarity = 0
        else:
            similarity = 1 - cosine(vector, query_vector)
        
        score_rows.append([commit_hash, similarity])

    score_df = pd.DataFrame(data=score_rows, columns=["commit_hash", "score"])
    score_df["rank"] = score_df["score"].rank(ascending=False, method="max")
    score_df["rank"] = score_df["rank"].astype(int)
    score_df.sort_values(by="rank", inplace=True)
    score_df = score_df[["commit_hash", "rank", "score"]]

    return score_df

if __name__ == "__main__":
    GT = load_BIC_GT(BIC_GT_DIR)

    use_br_list = [True, False]
    use_diff_list = [True, False]
    stage2_list = [True, False] # ['skip', True, False] Skip stage or use OpenRewrite or not
    use_stopword_list = [True] # [True, False] Use stopword or not
    adddel_list = ['add', 'del', 'all-uni', 'all-sep'] # Which diff data to uses
    param_list = list(itertools.product(use_br_list, use_diff_list, stage2_list, use_stopword_list, adddel_list))

    for _, row in GT.iterrows():
        pid, vid = row.pid, row.vid
        print(f'Bug2Commit : Working on {pid}-{vid}b')
        results_dict = dict()

        for (use_br, use_diff, stage2, use_stopword, adddel) in param_list:
            results_dict[(str(use_br), str(use_diff), str(stage2), str(use_stopword), adddel)] = \
                run_bug2commit(pid, vid, use_br=use_br, use_diff=use_diff, stage2=stage2, \
                use_stopword=use_stopword, adddel=adddel)
        
        results_df = pd.concat(results_dict, \
            names=['use_br', 'use_diff', 'stage2', 'use_stopword', 'adddel']).unstack()
        
        results_df.to_hdf(os.path.join(DIFF_DATA_DIR, f'{pid}-{vid}b/scores.hdf'), key='data', mode='w')

