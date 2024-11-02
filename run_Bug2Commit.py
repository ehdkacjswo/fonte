import os
import json
import numpy as np
import pandas as pd
from spiral import ronin
from rank_bm25 import BM25Okapi
from collections import Counter
from scipy.spatial.distance import cosine
from lib.experiment_utils import *

from BM25_Custom import BM25_Encode
from tqdm import tqdm

import pickle

CORE_DATA_DIR = "./data/Defects4J/core"
BIC_GT_DIR = "./data/Defects4J/BIC_dataset"
BASELINE_DATA_DIR = "./data/Defects4J/baseline"
DIFF_DATA_DIR = './data/Defects4J/diff'

class Encoder():
    def __init__(self, vocab={}):
        self.vocab = vocab # {word : id}
    
    # Encode the input and list of used word index and count
    def encode(self, text):
        encode_res = []
        text = ronin.split(text.strip())

        for word, cnt in Counter(text).items():
            if word in self.vocab: # Word in vocab
                encode_res.append((self.vocab[word], cnt))
                
            else: # New word
                encode_res.append((len(self.vocab), cnt))
                self.vocab[word] = len(self.vocab)
        
        return encode_res

def tokenize(text):
    return ronin.split(text)

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

def run_bug2commit(pid, vid):
    core_data_dir = os.path.join(CORE_DATA_DIR, f"{pid}-{vid}b")
    baseline_data_dir = os.path.join(BASELINE_DATA_DIR, f"{pid}-{vid}b")
    commit_dir = os.path.join(baseline_data_dir, "commits")
    
    savepath = os.path.join(baseline_data_dir, "ranking_Bug2Commit_no_br.csv")
    """if os.path.exists(savepath):
        print(f"{pid}-{vid}b: {savepath} already exists")
        return"""

    print(f"{pid}-{vid}b: Collecting commit features..........................")
    # get doc (commit) features
    commit_features = {}
    for filename in os.listdir(commit_dir):
        with open(os.path.join(commit_dir, filename), "r") as f:
            data = json.load(f)
            commit_message = data["log"].strip() # 1st commit feature
            modified_files = "\n".join(list(set([  # 2nd commit feature
                l[6:]
                for l in data["commit"].strip().split("\n")
                if l.startswith("+++ ")
            ])))
            commit_features[filename] = [commit_message, modified_files]

    print(f"{pid}-{vid}b: Collecting query features...........................")
    # get query (bug report) features
    query_features = []
    with open(os.path.join(core_data_dir, "failing_tests"), "r") as f:
        query_features.append(f.read().strip()) # 3rd bug report feature

    corpus = sum(commit_features.values(), [])
    print(corpus)
    return
    tokenized_corpus = [tokenize(doc) for doc in corpus]
    bm25 = BM25Okapi(tokenized_corpus)
    bm25.vocab = list(set( # collect all words appearning in the corpus
        sum([list(doc.keys()) for doc in bm25.doc_freqs], [])))

    # vectorize the features (using the vector_complex)
    print(f"{pid}-{vid}b: Vectorizing the features............................")

    commit_vectors = {
        filename: vectorize_complex(bm25, commit_features[filename])
        for filename in commit_features
    }
    query_vector = vectorize_complex(bm25, query_features)

    print(f"{pid}-{vid}b: Calculating the scores of commits...................")

    # filename : Name of the file in baseline/{pid}_{vid}b/commits/ (c_{commit_hash})
    score_rows = []
    for filename in commit_vectors:
        commit = filename[2:9]
        similarity = 1 - cosine(commit_vectors[filename], query_vector)
        score_rows.append([commit, filename, similarity])

    score_df = pd.DataFrame(data=score_rows,
        columns=["commit", "filename", "score"])
    score_df["rank"] = score_df["score"].rank(ascending=False, method="max")
    score_df["rank"] = score_df["rank"].astype(int)
    score_df.sort_values(by="rank", inplace=True)
    score_df = score_df[["commit", "filename", "rank", "score"]]
    if savepath:
        score_df.to_csv(savepath, index=False, header=None)
        print(f"{pid}-{vid}b: Saved to {savepath}")
    return score_df

# Bug2Commit with diff info
# 특정 버전에 대하여 vote for commits와 동일한 방법으로 evolve relationship 구축하고
# 해당 
def run_diff_bug2commit(pid, vid):
    core_data_dir = os.path.join(CORE_DATA_DIR, f"{pid}-{vid}b")
    diff_data_dir = os.path.join(DIFF_DATA_DIR, f"{pid}")
    baseline_data_dir = os.path.join(BASELINE_DATA_DIR, f"{pid}-{vid}b")
    commit_dir = os.path.join(baseline_data_dir, "commits")

    savepath = os.path.join(baseline_data_dir, "ranking_diff_all_simple_Bug2Commit.csv")
    """if os.path.exists(savepath):
        print(f"{pid}-{vid}b: {savepath} already exists")
        return"""
    
    print(f"{pid}-{vid}b: Encode query feature and build BM25..........................")
    # Load vocab and build encoder
    with open(os.path.join(diff_data_dir, 'vocab.pkl'), 'rb') as file:
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
    for commit in os.listdir(diff_data_dir):
        commit_dir = os.path.join(diff_data_dir, commit)

        # Skip vocab.pkl
        if not os.path.isdir(commit_dir):
            continue

        with open(os.path.join(commit_dir, 'encode/feature_all_simple.pkl'), 'rb') as file:
            commit_feature = pickle.load(file)
        
        for feature in commit_feature:
            bm25.add_document(feature)

        commit_feature_dict[commit[:7]] = commit_feature
    
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
    GT = load_BIC_GT(BIC_GT_DIR)
    cont = True
    for _, row in GT.iterrows():
        pid, vid = row.pid, row.vid

        try:
            run_diff_bug2commit(pid, vid)
        except:
            with open('/root/workspace/eror.txt', 'a') as file:
                file.write(f'Error on {pid}-{vid}b\n')