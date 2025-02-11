import os, json, argparse, pickle, sys, itertools, subprocess, logging
import pandas as pd

BIC_GT_DIR = "/root/workspace/data/Defects4J/BIC_dataset"
DIFF_DATA_DIR = '/root/workspace/data/Defects4J/diff'
CORE_DATA_DIR = '/root/workspace/data/Defects4J/core'
BASE_DATA_DIR = '/root/workspace/data/Defects4J/baseline'
RESULT_DATA_DIR = '/root/workspace/data/Defects4J/result'

sys.path.append('/root/workspace/lib/')
from experiment_utils import load_BIC_GT, weighted_bisection, get_all_commits

def log(txt, out_txt=None, err_txt=None):
    with open('/root/workspace/data_collector/log/bisection.log', 'a') as file:
        file.write(txt + '\n')

        if out_txt is not None:
            file.write('[ERROR] OUT\n' + out_txt.decode(encoding='utf-8', errors='ignore') + '\n')
        
        if err_txt is not None:
            file.write('[ERROR] ERR\n' + err_txt.decode(encoding='utf-8', errors='ignore') + '\n')

def get_style_change_commits(fault_dir, tool='git', stage2='precise'):
    if stage2 == 'skip':
        return []
    
    if stage2 == 'precise':
        val_df = pd.read_csv(
            os.path.join(fault_dir, tool, f"precise_validation.csv"), 
            header=None,
            names=["commit", "before_src_path", "after_src_path", "AST_diff"])

    val_df["unchanged"] = val_df["AST_diff"] == "U"
    agg_df = val_df.groupby("commit").all()[["unchanged"]]
    style_change_commits = agg_df.index[agg_df["unchanged"]].tolist()

    # Current precise style change doesn't include '/dev/null' path
    # Have to exclude commits that have '/dev/null' path
    # Can be deleted if it handles the cases
    com_df = pd.read_pickle(os.path.join(fault_dir, tool, "commits.pkl"))
    com_df["commit_hash"] = com_df["commit_hash"].apply(lambda s: str(s)[:7])

    valid_commits = com_df[~com_df[['before_src_path', 'after_src_path']].\
        isin(['/dev/null']).any(axis=1)]['commit_hash'].unique()
    
    return [commit for commit in style_change_commits if commit in valid_commits]

# Perform bisecion
def main(pid, vid):
    log(f'Working on {pid}_{vid}b')

    # Basic commit info
    GT = load_BIC_GT("/root/workspace/data/Defects4J/BIC_dataset")
    BIC = GT.set_index(["pid", "vid"]).loc[(pid, vid), "commit"]
    #print(BIC)

    fault_dir = os.path.join(CORE_DATA_DIR, f'{pid}-{vid}b')
    all_commits = get_all_commits(fault_dir)

    # Load voting results
    with open(os.path.join(RESULT_DATA_DIR, f'{pid}-{vid}b', 'vote', 'bug2commit.pkl'), 'rb') as file:
        bug2commit_dict = pickle.load(file)
    
    with open(os.path.join(RESULT_DATA_DIR, f'{pid}-{vid}b', 'vote', 'fonte.pkl'), 'rb') as file:
        fonte_dict = pickle.load(file)
    
    with open(os.path.join(RESULT_DATA_DIR, f'{pid}-{vid}b', 'vote', 'ensemble.pkl'), 'rb') as file:
        ensemble_dict = pickle.load(file)

    fonte_iter = dict()
    bug2commit_iter = dict()
    ensemble_iter = dict()
    
    for stage2, sub_dict in bug2commit_dict.items():
        fonte_df = fonte_dict[stage2]
        bug2commit_iter[stage2] = dict()
        ensemble_iter[stage2] = dict()

        # Get list of target commits
        style_change_commits = get_style_change_commits(fault_dir)

        # May check score contains 0 or not later

        C_BIC = [c for c in all_commits if c in fonte_df.index and c not in style_change_commits]

        # Bisection with scores
        votes = [float(fonte_df.loc[c, "vote"]) for c in C_BIC]
        fonte_iter[stage2] = weighted_bisection(C_BIC, votes, BIC)

        """for key, bug2commit_df in sub_dict.items():
            votes = [float(bug2commit_df.loc[c, "vote"]) for c in C_BIC]
            bug2commit_iter[stage2][key] = weighted_bisection(C_BIC, votes, BIC)"""
        
        for key, ensemble_df in ensemble_dict[stage2].items():
            votes = [float(ensemble_df.loc[c, "vote"]) for c in C_BIC]
            ensemble_iter[stage2][key] = weighted_bisection(C_BIC, votes, BIC)
    
    savedir = os.path.join(RESULT_DATA_DIR, f'{pid}-{vid}b', 'iteration')
    os.makedirs(savedir, exist_ok=True)

    with open(os.path.join(savedir, 'fonte.pkl'), 'wb') as file:
        pickle.dump(fonte_iter, file)
    
    """with open(os.path.join(savedir, 'bug2commit.pkl'), 'wb') as file:
        pickle.dump(bug2commit_iter, file)"""
    
    with open(os.path.join(savedir, 'ensemble.pkl'), 'wb') as file:
        pickle.dump(ensemble_iter, file)