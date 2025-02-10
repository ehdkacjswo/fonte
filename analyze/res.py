import argparse, os, copy, itertools, pickle, sys
import pandas as pd

sys.path.append('/root/workspace/lib/')
from experiment_utils import load_BIC_GT

RESULT_DATA_DIR = '/root/workspace/data/Defects4J/result/'

# res_dict[stage2][(new_diff_type, use_stopword, adddel, use_br)]
if __name__ == "__main__":
    fonte_iter = dict()
    ensemble_iter = dict()

    fonte_rank = dict()
    bug_rank = dict()
    ensemble_rank = dict()

    bug_no = 0
    bug_git = 0
    bug_base = 0
    bug_class = 0

    GT = load_BIC_GT("/root/workspace/data/Defects4J/BIC_dataset")

    for project in os.listdir(RESULT_DATA_DIR):
        if project == 'Closure-131b':
            continue
        iter_dir = os.path.join(RESULT_DATA_DIR, project, 'iteration')

        # Fonte iteration
        with open(os.path.join(iter_dir, 'fonte.pkl'), 'rb') as file:
            fonte_dict = pickle.load(file)

        for stage2, num_iter in fonte_dict.items():
            fonte_iter[stage2] = fonte_iter.get(stage2, 0) + num_iter
        
        # Ensemble iteration
        with open(os.path.join(iter_dir, 'ensemble.pkl'), 'rb') as file:
            ensemble_dict = pickle.load(file)
        
        for stage2, sub_dict in ensemble_dict.items():
            if stage2 not in ensemble_iter:
                ensemble_iter[stage2] = dict()

            for setting, num_iter in sub_dict.items():
                if not key[3] and key[1]:
                    ensemble_dict[key] = ensemble_dict.get(key, 0) + item
        
        # Vote
        [pid, vid] = project[:-1].split('-')
        BIC = GT.set_index(["pid", "vid"]).loc[(pid, vid), "commit"]

        vote_dir = os.path.join(RESULT_DATA_DIR, project, 'vote')

        # Fonte voting
        with open(os.path.join(vote_dir, 'fonte.pkl'), 'rb') as file:
            fonte_dict = pickle.load(file)
        
        for stage2, fonte_df in fonte_dict.items():
            rank = fonte_df['rank'].get(BIC, None)
            if rank is not None:
                print(pid, vid, stage2, fonte_df['rank'].get(BIC, None), fonte_df['vote'].get(BIC, None))

        # Bug2Commit voting
        with open(os.path.join(vote_dir, 'bug2commit.pkl'), 'rb') as file:
            bug_dict = pickle.load(file)
        
        for stage2, sub_dict in bug_dict.items():
            fonte_rank[stage2] = dict()

            for setting, bug_df in sub_dict.items():
                setting_dict = dict(setting)
                rank = bug_df['rank'].get(BIC, None)
                # if not setting_dict['use_br']

                fonte_rank[stage2][setting] = fonte_rank[stage2].get(setting, 0) + 1 / rank
        
        # Ensemble voting
        with open(os.path.join(vote_dir, 'ensemble.pkl'), 'rb') as file:
            ensemble_dict = pickle.load(file)

        for stage2, sub_dict in ensemble_dict.items():
            ensemble_rank[stage2] = dict()

            for setting, ensemble_df in sub_dict.items():
                setting_dict = dict(setting)
                rank = ensemble_df['rank'].get(BIC, None)
                # if not setting_dict['use_br']

                ensemble_rank[stage2][setting] = ensemble_rank[stage2].get(setting, 0) + 1 / rank
    
    print(fonte_iter)
    print(buggy_dict)
    print(ensemble_dict)