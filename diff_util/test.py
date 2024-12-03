import pickle as pkl
import os, sys
import pandas as pd

sys.path.append('/root/workspace/diff_util/lib/')
from diff import *

DIFF_DATA_DIR = '/root/workspace/data/Defects4J/diff'
CORE_DATA_DIR = '/root/workspace/data/Defects4J/core/'

def test_diff(pid='Cli', vid='29'):
    with open(os.path.join(DIFF_DATA_DIR, f'{pid}-{vid}b/diff.pkl'), 'rb') as file:
        diff_data = pkl.load(file)
    
    diff_data.self_print()

def get_range_dict(pid='Closure', vid='33', tool='git'):
    commit_path = os.path.join(CORE_DATA_DIR, f'{pid}-{vid}b', tool, 'commits.pkl')
    com_df = pd.read_pickle(commit_path)
    com_df.drop_duplicates(subset=['src_path', 'begin_line', 'end_line'], inplace=True)
    range_dict = dict()

    for _, row in com_df.iterrows():
        src_path = row['src_path']
        range_val = range_dict.get(src_path, [])
        range_val.append((row['begin_line'], row['end_line']))
        range_dict[src_path] = range_val

    return range_dict

if __name__ == "__main__":
    test_diff()
    #print(get_range_dict())