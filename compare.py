import argparse, os, copy, itertools, pickle
import matplotlib.pyplot as plt
from lib.experiment_utils import *
from sklearn.linear_model import LinearRegression
from scipy.stats import ttest_ind, mannwhitneyu, wilcoxon
from statistics import mean
import numpy as np

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

def elementwise_addition(series):
    print(series)
    return series.apply(lambda lists: list(np.sum(lists, axis=0)))

def compare_extra_score():
    GT = load_BIC_GT("./data/Defects4J/BIC_dataset")
    score_dict = dict()
    rank_dict = dict()

    for folder in tqdm(os.listdir(DIFF_DATA_DIR)):
        #print(f'Working on {folder}')
        [pid, vid] = folder[:-1].split("-")
        BIC = GT.set_index(["pid", "vid"]).loc[(pid, vid), "commit"]
        
        extra_scores_df = pd.read_hdf(os.path.join(DIFF_DATA_DIR, f'{folder}/scores.hdf'))

        for index, row in extra_scores_df.iterrows():
            commit_df = row['commit_hash'].dropna()
            rank_df = row['rank'].dropna()
            score_df = row['score'].dropna()

            commit_ind = commit_df.loc[commit_df == BIC].index[0]
            tup_ind = tuple(index)

            # Work on score data
            if tup_ind not in score_dict:
                score_dict[tup_ind] = [[], []]

            BIC_score = score_df.loc[commit_ind]
            score_mean = score_df.mean()
            
            score_dict[tup_ind][0] = score_dict[tup_ind][0] + [BIC_score - score_mean]
            score_dict[tup_ind][1] = score_dict[tup_ind][1] + [0 if score_mean == 0 else BIC_score / score_mean]
            
            # Work on rank data
            if tup_ind not in rank_dict:
                rank_dict[tup_ind] = [[], []]
            
            BIC_rank = rank_df.loc[commit_ind]
            rank_size = rank_df.size

            rank_dict[tup_ind][0] = rank_dict[tup_ind][0] + [BIC_rank]
            rank_dict[tup_ind][1] = rank_dict[tup_ind][1] + [BIC_rank / rank_size]
    
    # Build dataframe
    score_list = [(key[0], key[1], key[2], key[3], key[4], value[0], value[1]) for key, value in score_dict.items()]
    rank_list = [(key[0], key[1], key[2], key[3], key[4], value[0], value[1]) for key, value in rank_dict.items()]

    score_df = pd.DataFrame(score_list, columns=['use_br', 'use_diff', 'stage2', 'use_stopword', 'adddel', 'score_diff', 'score_ratio'])
    rank_df = pd.DataFrame(rank_list, columns=['use_br', 'use_diff', 'stage2', 'use_stopword', 'adddel', 'rank', 'rank_ratio'])

    for setting in ['use_br', 'use_diff', 'stage2', 'use_stopword', 'adddel']:
        print(f'Working on {setting}')

        score_true = score_df[score_df[setting] == 'True']
        score_false = score_df[score_df[setting] == 'False']

        rank_true = rank_df[rank_df[setting] == 'True']
        rank_false = rank_df[rank_df[setting] == 'False']

        #a = score_df.groupby(setting)[['score_diff', 'score_ratio']]
        b = (
            score_df.groupby(setting)
            .apply(lambda x: pd.Series({
                col: elementwise_addition(x[col]) for col in ['score_diff', 'score_ratio']
            }))
            .reset_index()
        )

        print(b)
        
        #print(b)
        
        """for _, (use_br, use_diff, stage2, use_stopword, adddel, score_diff, score_ratio) in score_diff_true.iterrows():
            print(score_diff)"""

        #print(score_diff_true)
        #print(score_diff_true.dropna())
        #print(score_diff_false)
        #print(score_diff_false.dropna())
        #t_stat, p_value = ttest_ind(score_diff_true.values, score_diff_false.values)
        #print(t_stat, p_value)
        break


    #print(score_df)
    #print(rank_df)
    
if __name__ == "__main__":
    compare_extra_score()

    
