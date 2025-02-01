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
RESULT_DATA_DIR = "/root/workspace/data/Defects4J/result"

sys.path.append('/root/workspace/diff_util/lib/')
from diff import *
from encoder import *

# Bug2Commit with diff info
# 특정 버전에 대하여 vote for commits와 동일한 방법으로 evolve relationship 구축하고
# 해당 
def run_bug2commit(pid, vid, feature_data, diff_type, use_stopword, adddel, use_br):
    core_data_dir = os.path.join(CORE_DATA_DIR, f"{pid}-{vid}b")
    diff_data_dir = os.path.join(DIFF_DATA_DIR, f"{pid}-{vid}b")
    baseline_data_dir = os.path.join(BASELINE_DATA_DIR, f"{pid}-{vid}b")
    commit_dir = os.path.join(baseline_data_dir, "commits")
    
    # Load vocab and build encoder
    with open(os.path.join(diff_data_dir, f'vocab.pkl'), 'rb') as file:
        vocab_dict = pickle.load(file)

    encoder = Encoder(vocab_dict[stage2][('git' if diff_type == 'no_diff' else diff_type, use_stopword)])
    
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
    if diff_type == 'no_diff':
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
            continue

        for commit_hash, commit_feature in feature_data.items():
            commit_vector = bm25.vectorize_complex([commit_feature[i]])

            if np.all(commit_vector == 0):
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

if __name__ == "__main__":
    GT = load_BIC_GT(BIC_GT_DIR)

    use_br_list = [True, False]
    use_diff_list = [True, False]
    param_list = list(itertools.product(use_br_list, use_diff_list))

    for _, row in GT.iterrows():
        pid, vid = row.pid, row.vid
        pid = "Cli"
        vid = 29
        print(f'Bug2Commit : Working on {pid}-{vid}b')
        res_dict = dict()

        with open(os.path.join(DIFF_DATA_DIR, f'{pid}-{vid}b', 'feature.pkl'), 'rb') as file:
            feature_dict = pickle.load(file)

        for stage2, sub_dict in feature_dict.items():
            res_dict[stage2] = dict()

            for (diff_type, use_stopword, adddel), feature_data in sub_dict.items():
                for (use_br, use_diff) in param_list:
                    new_diff_type = diff_type if use_diff else 'no_diff'
                    res_dict[stage2][(new_diff_type, use_stopword, adddel, use_br)] = \
                        run_bug2commit(pid, vid, feature_data=feature_data, diff_type=new_diff_type, \
                        use_stopword=use_stopword, adddel=adddel, use_br=use_br)
        
        result_dir = os.path.join(RESULT_DATA_DIR, f'{pid}-{vid}b', 'vote')
        os.makedirs(result_dir, exist_ok=True)

        with open(os.path.join(result_dir, 'bug2commit.pkl'), 'wb') as file:
            pickle.dump(res_dict, file)
        
        #results_df = pd.concat(res_dict, \
        #    names=['use_br', 'use_diff', 'stage2', 'use_stopword', 'adddel']).unstack()
        print(res_dict)
        
        break
        
        #results_df.to_hdf(os.path.join(DIFF_DATA_DIR, f'{pid}-{vid}b/scores.hdf'), key='data', mode='w')

