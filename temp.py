from scipy import sparse
import pandas as pd
from sbfl.base import SBFL
import os, pickle, math, re, subprocess, sys, shutil, json
from intervaltree import Interval, IntervalTree
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import shutil

sys.path.append('/root/workspace/lib/')
from experiment_utils import load_BIC_GT

DIFF_DATA_DIR = '/root/workspace/data/Defects4J/diff'
RESULT_DATA_DIR = '/root/workspace/data/Defects4J/result'
CORE_DATA_DIR = '/root/workspace/data/Defects4J/core'

#test_regex = re.compile()
# Directory mounted on "original" adfasdf
DIR_NAME = '/home/coinse/doam/fonte/tmp'

if __name__ == "__main__":
    all_GT = load_BIC_GT("/root/workspace/data/Defects4J/BIC_dataset")
    df = all_GT[all_GT['provenance'].str.contains("Manual", na=False)]
    print(df)
    
    path_ = '/root/workspace/data/Defects4J/result/'
    for aaa in os.listdir(path_):
        pid, vid = aaa[:-1].split('-')

        exist = ((df['pid'] == pid) & (df['vid'] == vid)).any()
        if not exist:
            shutil.rmtree(os.path.join(path_, aaa))
        
        print(pid, vid, exist)