from scipy import sparse
import pandas as pd
from sbfl.base import SBFL
import os, pickle
from intervaltree import Interval, IntervalTree

DIFF_DATA_DIR = '/root/workspace/data/Defects4J/diff'
RESULT_DATA_DIR = '/root/workspace/data/Defects4J/result'

if __name__ == "__main__":
    for project in os.listdir(DIFF_DATA_DIR):
        project_dir = os.path.join(RESULT_DATA_DIR_DATA_DIR, project)

        """with open(os.path.join(project_dir, 'stage2.pkl'), 'rb') as file:
            a = pickle.load(file)
        
        new_dict = dict()
        
        for stage2, sub_dict in a.items():
            new_dict[stage2] = dict()

            for key, value in sub_dict.items():
                new_key = frozenset({'diff_type' : key}.items())
                new_dict[stage2][new_key] = value
        
        with open(os.path.join(project_dir, 'stage2.pkl'), 'wb') as file:
            pickle.dump(new_dict, file)"""
        
        """with open(os.path.join(project_dir, 'encode.pkl'), 'rb') as file:
            a = pickle.load(file)
        
        new_dict = dict()
        
        for stage2, sub_dict in a.items():
            new_dict[stage2] = dict()

            for key, value in sub_dict.items():
                new_key = frozenset({'diff_type' : key[0], 'use_stopword' : key[1]}.items())
                new_dict[stage2][new_key] = value
        
        with open(os.path.join(project_dir, 'encode.pkl'), 'wb') as file:
            pickle.dump(new_dict, file)"""
        
        """with open(os.path.join(project_dir, 'vocab.pkl'), 'rb') as file:
            a = pickle.load(file)
        
        new_dict = dict()
        
        for stage2, sub_dict in a.items():
            new_dict[stage2] = dict()

            for key, value in sub_dict.items():
                new_key = frozenset({'diff_type' : key[0], 'use_stopword' : key[1]}.items())
                new_dict[stage2][new_key] = value
        
        with open(os.path.join(project_dir, 'vocab.pkl'), 'wb') as file:
            pickle.dump(new_dict, file)"""
        
        """with open(os.path.join(project_dir, 'feature.pkl'), 'rb') as file:
            a = pickle.load(file)
        
        new_dict = dict()
        
        for stage2, sub_dict in a.items():
            new_dict[stage2] = dict()

            for key, value in sub_dict.items():
                new_key = frozenset({'diff_type' : key[0], 'use_stopword' : key[1], 'adddel' : key[2]}.items())
                new_dict[stage2][new_key] = value
        
        with open(os.path.join(project_dir, 'feature.pkl'), 'wb') as file:
            pickle.dump(new_dict, file)"""
        
        with open(os.path.join(project_dir, 'feature.pkl'), 'rb') as file:
            a = pickle.load(file)
        
        new_dict = dict()
        
        for stage2, sub_dict in a.items():
            new_dict[stage2] = dict()

            for key, value in sub_dict.items():
                new_key = frozenset({'diff_type' : key[0], 'use_stopword' : key[1], 'adddel' : key[2]}.items())
                new_dict[stage2][new_key] = value
        
        with open(os.path.join(project_dir, 'feature.pkl'), 'wb') as file:
            pickle.dump(new_dict, file)
        
        