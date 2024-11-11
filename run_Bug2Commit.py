import os, sys, itertools
import json
import argparse
import numpy as np
import pandas as pd
from spiral import ronin
from rank_bm25 import BM25Okapi
from collections import Counter
from scipy.spatial.distance import cosine
from lib.experiment_utils import *
from nltk.corpus import stopwords
import regex as re

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

def vectorize_complex(bm25, features):
    vectors = []
    for feature in features:
        tokenize_feature = tokenize(feature)
        doc_len = len(tokenize_feature)
        doc = Counter(tokenize_feature)
        vector = []
        for v in bm25.vocab:
            freq = doc.get(v, 0)
            vector.append(bm25.idf.get(v, 0) * (freq * (bm25.k1 + 1) /
                (freq + bm25.k1 * (1 - bm25.b + bm25.b * doc_len / bm25.avgdl))))
        vectors.append(vector)
    assert len(vectors) == len(features)
    assert len(vectors[0]) == len(bm25.vocab)
    vectors = np.array(vectors)
    return vectors.mean(axis=0)

# Bug2Commit with diff info
# 특정 버전에 대하여 vote for commits와 동일한 방법으로 evolve relationship 구축하고
# 해당 
def run_bug2commit(pid, vid, use_diff=True, tool='git', skip_stage_2=False, with_Rewrite=True, use_stopword=True, adddel='all', encode_type='simple'):
    core_data_dir = os.path.join(CORE_DATA_DIR, f"{pid}-{vid}b")
    diff_data_dir = os.path.join(DIFF_DATA_DIR, f"{pid}-{vid}b")
    baseline_data_dir = os.path.join(BASELINE_DATA_DIR, f"{pid}-{vid}b")
    commit_dir = os.path.join(baseline_data_dir, "commits")

    file_postfix = savepath_postfix(tool, skip_stage_2, with_Rewrite, use_stopword)
    diff_prefix = 'diff_' if use_diff else ''
    savepath = os.path.join(baseline_data_dir, f'{diff_prefix}ranking{file_postfix}.csv')
    """if os.path.exists(savepath):
        print(f"{pid}-{vid}b: {savepath} already exists")
        return"""
    
    print(f"{pid}-{vid}b: Encode query feature and build BM25..........................")
    # Load vocab and build encoder
    with open(os.path.join(diff_data_dir, f'encode/vocab{file_postfix}.pkl'), 'rb') as file:
        vocab = pickle.load(file)

    encoder = Encoder(vocab)
    bm25 = BM25_Encode()

    # Get query features (failing tests)
    with open(os.path.join(core_data_dir, "failing_tests"), "r") as f:
        query_feature = f.read().strip()
    
    # Encode the query feature
    query_encode = encoder.encode(query_feature)
    bm25.add_document(query_encode)

    # For target list of commits get used set
    commit_feature_dict = dict()
    with open(os.path.join(diff_data_dir, f'feature/{adddel}{file_postfix}.pkl'), 'rb') as file:
        commit_feature_dict = pickle.load(file)
    
    for commit_feature_list in commit_feature_dict.values():
        if not use_diff: # Ignore diff
            commit_feature_list = commit_feature_list[:2]
        for feature in commit_feature_list:
            bm25.add_document(feature)
    
    bm25.init_end()
    
    print(f"{pid}-{vid}b: Vectorizing features..........................")
    # Handle query features
    query_vector = bm25.vectorize_complex([query_encode])

    if not np.any(query_vector):
        with open('/root/workspace/eror.txt', 'a') as file:
                file.write(f'Query vector of {pid}-{vid}b is zero\n')

    # Handle commit features
    for commit_hash in tqdm(commit_feature_dict.keys()):
        commit_feature_dict[commit_hash] = bm25.vectorize_complex(commit_feature_dict[commit_hash])

        if not np.any(commit_feature_dict[commit_hash]):
            with open('/root/workspace/eror.txt', 'a') as file:
                file.write(f'Commit vector of {pid}-{vid}b:{commit_hash} is zero\n')

    print(f"{pid}-{vid}b: Calculating the scores of commits...................")
    score_rows = []
    for commit_hash, vector in commit_feature_dict.items():
        similarity = 1 - cosine(vector, query_vector)
        #print(vector, query_vector, similarity)
        score_rows.append([commit_hash, similarity])

    score_df = pd.DataFrame(data=score_rows,
        columns=["commit_hash", "score"])
    score_df["rank"] = score_df["score"].rank(ascending=False, method="max")
    score_df["rank"] = score_df["rank"].astype(int)
    score_df.sort_values(by="rank", inplace=True)
    score_df = score_df[["commit_hash", "rank", "score"]]
    if savepath:
        score_df.to_csv(savepath, index=False, header=None)
        print(f"{pid}-{vid}b: Saved to {savepath}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Encode diff data")
    parser.add_argument('--tool', type=str, default="git",
        help="history retrieval tool, git or shovel (default: git)")
    parser.add_argument('--skip-stage-2', action="store_true",
        help="skiping stage 2 (default: False)")
    parser.add_argument('--with-Rewrite', action="store_true",
        help="skiping stage 2 (default: False)")
    parser.add_argument('--use-stopword', type=int, default=0,
        help="alpha (default: 0)")
    parser.add_argument('--adddel', type=str, default="max",
        help="tau (default: max)")

    GT = load_BIC_GT(BIC_GT_DIR)

    use_diff_list = [True, False]
    skip_stage_2_list = [True, False]
    with_Rewrite_list = [True, False]
    use_stopword_list = [True, False]
    adddel_list = ['all', 'add', 'del']
    param_list = list(itertools.product(use_diff_list, skip_stage_2_list, with_Rewrite_list, use_stopword_list, adddel_list))

    for _, row in GT.iterrows():
        pid, vid = row.pid, row.vid

        for (use_diff, skip_stage_2, with_Rewrite, use_stopword, adddel) in param_list:
            try:
                run_bug2commit(pid, vid, use_diff=use_diff, tool='git', skip_stage_2=skip_stage_2, with_Rewrite=with_Rewrite, use_stopword=use_stopword, adddel=adddel, encode_type='simple')
            except:
                with open('/root/workspace/eror.txt', 'a') as file:
                    file.write(f'Error on {pid}-{vid}b {skip_stage_2} {with_Rewrite} {use_stopword} {adddel}\n')