import os, json, argparse, pickle, sys, itertools, time
import pandas as pd

BIC_GT_DIR = "/root/workspace/data/Defects4J/BIC_dataset"
DIFF_DATA_DIR = '/root/workspace/data/Defects4J/diff'
CORE_DATA_DIR = '/root/workspace/data/Defects4J/core'
BASE_DATA_DIR = '/root/workspace/data/Defects4J/baseline'
RESULT_DATA_DIR = '/root/workspace/data/Defects4J/result'

sys.path.append('/root/workspace/lib/')
from experiment_utils import load_BIC_GT, weighted_bisection, get_all_commits

sys.path.append('/root/workspace/data_collector/lib/')
from utils import log, time_to_str

# Get list of style change commits
def get_style_change_commits(fault_dir, tool='git', stage2='precise'):
    if stage2 == 'skip':
        return []
    
    if stage2 == 'precise':
        val_df = pd.read_csv(
            os.path.join(fault_dir, tool, f"precise_validation_noOpenRewrite.csv"), 
            header=None,
            names=["commit", "before_src_path", "after_src_path", "AST_diff"])

    # Get list of commits with every change is style change
    val_df["unchanged"] = (val_df["AST_diff"] == "U")
    agg_df = val_df.groupby("commit").all()[["unchanged"]]
    style_change_commits = agg_df.index[agg_df["unchanged"]].tolist()

    # Precise style change doesn't consider path '/dev/null'
    # Exclude commits that have '/dev/null' path (File creation / deletion)
    # Can be deleted if it handles the cases
    com_df = pd.read_pickle(os.path.join(fault_dir, tool, "commits.pkl"))
    com_df["commit_hash"] = com_df["commit_hash"].apply(lambda s: str(s)[:7])

    valid_commits = com_df[~com_df[['before_src_path', 'after_src_path']].\
        isin(['/dev/null']).any(axis=1)]['commit_hash'].unique()
    
    return [commit for commit in style_change_commits if commit in valid_commits]

# Perform bisecion
def main(pid, vid):
    log('bisection', f'Working on {pid}_{vid}b')
    start_time = time.time()

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
    
    with open(os.path.join(RESULT_DATA_DIR, f'{pid}-{vid}b', 'vote', 'fbl_bert.pkl'), 'rb') as file:
        fbl_bert_dict = pickle.load(file)

    fonte_iter = dict()
    bug2commit_iter = dict()
    ensemble_iter = dict()
    fbl_bert_iter = dict()
    
    for stage2, sub_dict in bug2commit_dict.items():
        fonte_df = fonte_dict[stage2]
        fbl_bert_df = fbl_bert_dict[stage2]
        bug2commit_iter[stage2] = dict()
        ensemble_iter[stage2] = dict()

        # Get list of target commits
        # Currently using only git, but have to change it later
        style_change_commits = get_style_change_commits(fault_dir, stage2=stage2)

        # May check score contains 0 or not later

        C_BIC = [c for c in all_commits if c in fonte_df.index and c not in style_change_commits]

        # Automatically driven BIC could be style change commit
        if BIC not in C_BIC:
            log('bisection', f'[ERROR] BIC({BIC}) is style change commit for stage2({stage2})')
            continue
        
        # Bisection with scores
        # Fonte & Ensemble
        if float(fonte_df.loc[BIC, "vote"]) == 0: # When BIC has 0 score with Fonte, unable to perform bisection
            log('bisection', f'[ERROR] BIC({BIC}) has 0 Fonte score for stage2({stage2})')

        else:
            votes = [float(fonte_df.loc[c, "vote"]) for c in C_BIC]
            fonte_iter[stage2] = weighted_bisection(C_BIC, votes, BIC)

            for setting, ensemble_df in ensemble_dict[stage2].items():
                votes = [float(ensemble_df.loc[c, "vote"]) for c in C_BIC]
                ensemble_iter[stage2][setting] = weighted_bisection(C_BIC, votes, BIC)
        
        # Bug2Commit 
        for setting, bug2commit_df in sub_dict.items():
            #print(bug2commit_df['all'])
            if float(bug2commit_df['all'].loc[BIC, "vote"]) == 0:
                log('bisection', f'[INFO] BIC({BIC}) has 0 Bug2Commit score for stage2({stage2}), {str(setting)}')

            votes = [float(bug2commit_df['all'].loc[c, "vote"]) for c in C_BIC]
            
            # Apply positive score for all 0 score commits
            min_vote = min((vote for vote in votes if vote > 0), default=1)
            zero_ind = [ind for ind, vote in enumerate(votes) if vote == 0]

            for beta in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
                new_votes = [vote if vote > 0 else beta * min_vote for vote in votes]
                bug2commit_iter[stage2][frozenset((dict(setting) | {'beta' : beta}).items())] = weighted_bisection(C_BIC, new_votes, BIC)
        
        """
        # FBL-BERT
        if float(fbl_bert_df.loc[BIC, "vote"]) == 0:
            log('bisection', f'[INFO] BIC({BIC}) has 0 FBL-BERT score for stage2({stage2})')

        votes = [float(fbl_bert_df.loc[c, "vote"]) for c in C_BIC]
            
        # Apply positive score for all 0 score commits
        min_vote = min((vote for vote in votes if vote > 0), default=1)
        zero_ind = [ind for ind, vote in enumerate(votes) if vote == 0]

        fbl_bert_iter[stage2] = dict()

        for beta in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
            new_votes = [vote if vote > 0 else beta * min_vote for vote in votes]
            fbl_bert_iter[stage2][frozenset({'beta' : beta}.items())] = weighted_bisection(C_BIC, new_votes, BIC)
        """
        
    savedir = os.path.join(RESULT_DATA_DIR, f'{pid}-{vid}b', 'iteration')
    os.makedirs(savedir, exist_ok=True)

    with open(os.path.join(savedir, 'fonte.pkl'), 'wb') as file:
        pickle.dump(fonte_iter, file)
    
    with open(os.path.join(savedir, 'bug2commit.pkl'), 'wb') as file:
        pickle.dump(bug2commit_iter, file)
    
    with open(os.path.join(savedir, 'ensemble.pkl'), 'wb') as file:
        pickle.dump(ensemble_iter, file)
    
    #with open(os.path.join(savedir, 'fbl_bert.pkl'), 'wb') as file:
    #    pickle.dump(fbl_bert_iter, file)
    
    end_time = time.time()
    log('bisection', f'[INFO] Elapsed time : {time_to_str(start_time, end_time)}')