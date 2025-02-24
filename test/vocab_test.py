import argparse, os, copy, itertools, pickle, sys, json, subprocess
import pandas as pd
from scipy.stats import wilcoxon
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import math

sys.path.append('/root/workspace/lib/')
from experiment_utils import load_BIC_GT

DIFF_DATA_DIR = '/root/workspace/data/Defects4J/diff'

if __name__ == "__main__":
    GT = load_BIC_GT("/root/workspace/data/Defects4J/BIC_dataset")

    # Iterate through projects
    for _, row in GT.iterrows():
        pid, vid = row.pid, row.vid
        vocab_path = os.path.join(DIFF_DATA_DIR, f'{pid}-{vid}b', 'vocab.pkl')
        encode_path = os.path.join(DIFF_DATA_DIR, f'{pid}-{vid}b', 'encode.pkl')

        if not os.path.isfile(vocab_path) or (pid == 'Closure' and vid == '131'):
            continue

        print(f'{pid}-{vid}b')
        
        with open(vocab_path, 'rb') as file:
            vocab_dict = pickle.load(file)
        
        with open(encode_path, 'rb') as file:
            encode_dict = pickle.load(file)
        
        file_vocab = set(vocab_dict['precise'][frozenset({'diff_type':'file', 'use_stopword':True}.items())].keys())
        git_vocab = set(vocab_dict['precise'][frozenset({'diff_type':'git', 'use_stopword':True}.items())].keys())
        base_vocab = set(vocab_dict['precise'][frozenset({'diff_type':'gumtree_base', 'use_stopword':True}.items())].keys())
        class_vocab = set(vocab_dict['precise'][frozenset({'diff_type':'gumtree_class', 'use_stopword':True}.items())].keys())

        if len(git_vocab - file_vocab) > 0:
            print('New token on git')

            #new_ind = set(vocab_dict['precise'][frozenset({'diff_type':'git', 'use_stopword':True}.items())][word] for word in git_vocab - file_vocab)

            #for commit, adddel_dict in encode_dict['preicse'][frozenset({'diff_type':'git', 'use_stopword':True}.items())].items():
            #    for adddel, path_dict in sub_dict.items():
                    
            
        
        if len(base_vocab - git_vocab) > 0:
            print('New token on base')
        
        if len(class_vocab - base_vocab) > 0:
            print('New token on class')

