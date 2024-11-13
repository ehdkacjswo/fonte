import argparse, os, copy, itertools, pickle
import matplotlib.pyplot as plt
from lib.experiment_utils import *
from sklearn.linear_model import LinearRegression
from scipy.stats import wilcoxon, ttest_rel
from statistics import mean
import numpy

sys.path.append('/root/workspace/diff_util/lib/')
from encoder import savepath_postfix
from tqdm import tqdm

DIFF_DATA_DIR = '/root/workspace/data/Defects4J/diff'

voting_functions = {
    # key: (alpha, tau)
    (1, 'max'): (lambda r: r.score/r.max_rank),
    (0, 'max'): (lambda r: 1/r.max_rank),
    (1, 'dense'): (lambda r: r.score/r.dense_rank),
    (0, 'dense'): (lambda r: 1/r.dense_rank),
}    

def analyze_ranks(ranks):
    # Analyze acc@n
    """for i in [1, 2, 3, 5, 10]:
        print('acc@{} : {}'.format(i, sum(x <= i for x in ranks)))"""
    
    # Analyze MRR
    print('MRR : {}'.format(mean([1/x for x in ranks])))

def compare_iters(base, new):
    better = 0
    same = 0
    worse = 0

    for i in range(len(base)):
        if new[i] < base[i]:
            better = better + 1
        elif new[i] == base[i]:
            same = same + 1
        else:
            worse = worse + 1

    print('Better:{}, Same:{}, Worse:{}, Total:{}', better, same, worse, better + same + worse)
    print('Base total iterations : {}, New total iterations : {}'.format(sum(base), sum(new)))
    
    try:
        print(ttest_rel(base, new, alternative='greater'))
    except:
        print('Error on Ttest')
    
    try:
        print(wilcoxon(base, new, alternative='greater'))
    except:
        print('Error on Wilcoxon')

# Perform weighted bisection while ignoring lower scores
def ignore_bisection(C_BIC_list, scores_list, BIC_list, BIC_rank_list):
    num_iter_list = []
    for (C_BIC, scores, BIC, BIC_rank) in zip(C_BIC_list, scores_list, BIC_list, BIC_rank_list):
        num_iter_list.append(weighted_bisection(C_BIC, scores, BIC))
    return num_iter_list

def BIC_rank(use_diff=True, skip_stage_2=False, with_Rewrite=True, \
    use_stopword=True, adddel='all'):

    # Use Bug2Commit
    file_postfix = savepath_postfix('git', skip_stage_2, with_Rewrite, use_stopword)
    diff_prefix = ('diff_' if use_diff else '') + adddel + '_'

    # Return values
    BIC_rank_list = []

    # Load BIC data
    GT = load_BIC_GT("./data/Defects4J/BIC_dataset")

    # Iterate through every projects
    for folder in os.listdir('data/Defects4J/baseline/'):

        # Get BIC data
        [pid, vid] = folder[:-1].split("-")
        fault = (pid, vid)
        BIC = GT.set_index(["pid", "vid"]).loc[fault, "commit"]

        df = df = pd.read_csv(os.path.join('/root/workspace/data/Defects4J/baseline/', folder,\
            f'{diff_prefix}ranking{file_postfix}.csv'), names=["commit_hash", "rank", "score"])

        # Get the rank of BIC
        BIC_rank_list.append(int(df.loc[df["commit_hash"] == BIC, "rank"]))
    
    return BIC_rank_list
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute commit scores")
    #parser.add_argument('coredir', help="data/Defects4J/core/<pid>-<vid>b")
    parser.add_argument('--tool', type=str, default="git",
        help="history retrieval tool, git or shovel (default: git)")
    parser.add_argument('--formula', type=str, default="Ochiai",
        help="SBFL formula (default: Ochiai)")
    parser.add_argument('--alpha', type=int, default=0,
        help="alpha (default: 0)")
    parser.add_argument('--tau', type=str, default="max",
        help="tau (default: max)")
    parser.add_argument('--lamb', type=float, default=0.1,
        help="lambda (default: 0.1)")
    parser.add_argument('--skip-stage-2', action="store_true",
        help="skiping stage 2 (default: False)")
    parser.add_argument('--with-Rewrite', action="store_true",
        help="not using openrewrite in Stage 2(default: True)")
    parser.add_argument('--output', '-o',
        help="path to output file (example: output.csv)")
    # Additional argument
    parser.add_argument('--beta', type=float, default=1.0,
        help="beta (default: 1.0)")
    args = parser.parse_args()

    assert args.alpha in [0, 1]
    assert args.tau in ["max", "dense"]
    assert 0 <= args.lamb < 1
    
    # Generate iteration data
    for folder in tqdm(os.listdir(DIFF_DATA_DIR)):
        # Get BIC data
        print(f'Working on {folder}')
        [pid, vid] = folder[:-1].split("-")
        results_dict = score_eval_all(pid, vid, args.tool, args.formula, args.lamb, voting_functions[(args.alpha, args.tau)])
        results_df = pd.concat(results_dict, \
            names=['HSFL', 'score_mode', 'ensemble', 'use_br', 'use_diff', 'stage2', 'use_stopword', 'adddel']).unstack()
        
        results_df.to_hdf(os.path.join(DIFF_DATA_DIR, f'{folder}/fonte_scores.hdf'), key='data', mode='w')
    
    # Generate iteration data
    """iter_dict = dict()
    for folder in tqdm(os.listdir(DIFF_DATA_DIR)):
        # Get BIC data
        print(f'Working on {folder}')
        [pid, vid] = folder[:-1].split("-")
        bisection_all(pid, vid)"""