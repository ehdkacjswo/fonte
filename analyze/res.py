import argparse, os, copy, itertools, pickle, sys
import pandas as pd

sys.path.append('/root/workspace/lib/')
from experiment_utils import load_BIC_GT

RESULT_DATA_DIR = '/root/workspace/data/Defects4J/result/'

# res_dict[stage2][(new_diff_type, use_stopword, adddel, use_br)]
if __name__ == "__main__":
    fonte_iter = 0
    buggy_dict = dict()
    ensemble_dict = dict()

    bug_no = 0
    bug_git = 0
    bug_base = 0
    bug_class = 0

    GT = load_BIC_GT("/root/workspace/data/Defects4J/BIC_dataset")

    for project in os.listdir(RESULT_DATA_DIR):
        iter_dir = os.path.join(RESULT_DATA_DIR, project, 'iteration')

        with open(os.path.join(iter_dir, 'fonte.pkl'), 'rb') as file:
            fonte_dict = pickle.load(file)
            fonte_iter += fonte_dict['skip']
        
        with open(os.path.join(iter_dir, 'bug2commit.pkl'), 'rb') as file:
            bug_dict = pickle.load(file)
            for key, item in bug_dict['skip'].items():
                if not key[3] and key[1]:
                    buggy_dict[key] = buggy_dict.get(key, 0) + item
        
        with open(os.path.join(iter_dir, 'ensemble.pkl'), 'rb') as file:
            bug_dict = pickle.load(file)
            for key, item in bug_dict['skip'].items():
                if not key[3] and key[1]:
                    ensemble_dict[key] = ensemble_dict.get(key, 0) + item
        
        # Vote
        """[pid, vid] = project[:-1].split('-')
        BIC = GT.set_index(["pid", "vid"]).loc[(pid, vid), "commit"]

        vote_dir = os.path.join(RESULT_DATA_DIR, project, 'vote')

        with open(os.path.join(vote_dir, 'fonte.pkl'), 'rb') as file:
            fonte_dict = pickle.load(file)
            fonte_iter += fonte_dict['skip']
        
        with open(os.path.join(vote_dir, 'bug2commit.pkl'), 'rb') as file:
            bug_dict = pickle.load(file)
            for key, item in bug_dict['skip'].items():
                if not key[3] and key[1]:
                    buggy_dict[key] = buggy_dict.get(key, 0) + item
        
        with open(os.path.join(vote_dir, 'ensemble.pkl'), 'rb') as file:
            bug_dict = pickle.load(file)
            for key, item in bug_dict['skip'].items():
                if not key[3] and key[1]:
                    ensemble_dict[key] = ensemble_dict.get(key, 0) + item"""
    
    print(fonte_iter)
    print(buggy_dict)
    print(ensemble_dict)