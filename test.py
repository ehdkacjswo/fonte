import argparse, os, copy
import matplotlib.pyplot as plt
from lib.experiment_utils import *
from sklearn.linear_model import LinearRegression
from scipy.stats import wilcoxon, ttest_rel
from statistics import mean
import numpy

voting_functions = {
    # key: (alpha, tau)
    (1, 'max'): (lambda r: r.score/r.max_rank),
    (0, 'max'): (lambda r: 1/r.max_rank),
    (1, 'dense'): (lambda r: r.score/r.dense_rank),
    (0, 'dense'): (lambda r: 1/r.dense_rank),
}    

def analyze_ranks(ranks):
    # Analyze acc@n
    for i in [1, 2, 3, 5, 10]:
        print('acc@{} : {}'.format(i, sum(x <= i for x in ranks)))
    
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
def ignore_bisection(C_BIC_list, scores_list, BIC_list, BIC_rank_list, ignore=0.0):
    num_iter_list = []
    num_BIC_ignore = 0

    for (C_BIC, scores, BIC, BIC_rank) in zip(C_BIC_list, scores_list, BIC_list, BIC_rank_list):
        # Ignore lower scores
        sorted_scores = copy.deepcopy(scores)
        sorted_scores.sort()
        new_scores = [sorted_scores[0] if s <= sorted_scores[int(len(sorted_scores) * ignore)] else s for s in scores]

        # Check if BIC is ignored or not
        if sorted_scores[len(sorted_scores) - BIC_rank] <= sorted_scores[int(len(sorted_scores) * ignore)]:
            num_BIC_ignore = num_BIC_ignore + 1
        
        num_iter_list.append(weighted_bisection(C_BIC, new_scores, BIC))
    
    print('Ignoring {}percent of scores ignored {} BICs'.format(int(ignore * 100), num_BIC_ignore))
    return num_iter_list
    
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
    parser.add_argument('--no-openrewrite', action="store_true",
        help="not using openrewrite in Stage 2(default: False)")
    parser.add_argument('--output', '-o',
        help="path to output file (example: output.csv)")
    # Additional argument
    parser.add_argument('--beta', type=float, default=1.0,
        help="beta (default: 1.0)")
    args = parser.parse_args()

    assert args.alpha in [0, 1]
    assert args.tau in ["max", "dense"]
    assert 0 <= args.lamb < 1

    """res = aaa(args)

    sbfl_res = res[0]
    max_res = res[1]
    sum_res = res[2]

    for i in range(4):
        print(i)
        sbfl = [x[i] for x in sbfl_res]
        max_ = [x[i] for x in max_res]
        sum_ = [x[i] for x in sum_res]

        analyze_ranks(sbfl)
        analyze_ranks(max_)
        analyze_ranks(sum_)

        compare_iters(sbfl, max_)
        compare_iters(sbfl, sum_)"""

    # Run Fonte with various settings
    C_BIC_list_base, scores_list_base, BIC_list_base, BIC_rank_list_base = fonte(args, HSFL=True, score=None)
    C_BIC_list_score_base, scores_list_score_base, BIC_list_score_base, BIC_rank_list_score_base = fonte(args, HSFL=True, score='bug2commit')
    C_BIC_list_score_diff, scores_list_score_diff, BIC_list_score_diff, BIC_rank_list_score_diff = fonte(args, HSFL=True, score='bug2commit_diff')
    
    # Analyze ranks
    analyze_ranks(BIC_rank_list_base)
    analyze_ranks(BIC_rank_list_score_base)
    analyze_ranks(BIC_rank_list_score_diff)

    num_iters_base = ignore_bisection(C_BIC_list_base, scores_list_base, BIC_list_base, BIC_rank_list_base)
    num_iters_score_base = ignore_bisection(C_BIC_list_score_base, scores_list_score_base, BIC_list_score_base, BIC_rank_list_score_base)
    num_iters_score_diff = ignore_bisection(C_BIC_list_score_diff, scores_list_score_diff, BIC_list_score_diff, BIC_rank_list_score_diff)

    compare_iters(num_iters_base, num_iters_score_base)
    compare_iters(num_iters_base, num_iters_score_diff)
    compare_iters(num_iters_score_base, num_iters_score_diff)
    
    """analyze_ranks(BIC_rank_list_base)
    num_iters_base = ignore_bisection(C_BIC_list_base, scores_list_base, BIC_list_base, BIC_rank_list_base)

    for beta in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        print(beta)
        args.beta = beta
        C_BIC_list_hsfl, scores_list_hsfl, BIC_list_hsfl, BIC_rank_list_hsfl = fonte(args, HSFL=False, score='fbl_bert')
        analyze_ranks(BIC_rank_list_hsfl)
        num_iters_hsfl = ignore_bisection(C_BIC_list_hsfl, scores_list_hsfl, BIC_list_hsfl, BIC_rank_list_hsfl)
        compare_iters(num_iters_base, num_iters_hsfl)"""

    """C_BIC_list_base, scores_list_base, BIC_list_base, BIC_rank_list_base = fonte(args, HSFL=False, score=None)
    C_BIC_list_hsfl, scores_list_hsfl, BIC_list_hsfl, BIC_rank_list_hsfl = fonte(args, HSFL=True, score=None)
    C_BIC_list_score, scores_list_score, BIC_list_score, BIC_rank_list_score = fonte(args, HSFL=False, score='bug2commit')
    C_BIC_list_all, scores_list_all, BIC_list_all, BIC_rank_list_all = fonte(args, HSFL=True, score='bug2commit')

    # Analyze ranks
    analyze_ranks(BIC_rank_list_base)
    analyze_ranks(BIC_rank_list_hsfl)
    analyze_ranks(BIC_rank_list_score)
    analyze_ranks(BIC_rank_list_all)

    num_iters_base = ignore_bisection(C_BIC_list_base, scores_list_base, BIC_list_base, BIC_rank_list_base)
    num_iters_hsfl = ignore_bisection(C_BIC_list_hsfl, scores_list_hsfl, BIC_list_hsfl, BIC_rank_list_hsfl)
    num_iters_score = ignore_bisection(C_BIC_list_score, scores_list_score, BIC_list_score, BIC_rank_list_score)
    num_iters_all = ignore_bisection(C_BIC_list_all, scores_list_all, BIC_list_all, BIC_rank_list_all)

    compare_iters(num_iters_base, num_iters_all)
    compare_iters(num_iters_hsfl, num_iters_all)
    compare_iters(num_iters_score, num_iters_all)
    
    for ig in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        num_iters_new = ignore_bisection(C_BIC_list_all, scores_list_all, BIC_list_all, BIC_rank_list_all, ignore=ig)
        compare_iters(num_iters_all, num_iters_new)"""

    """x = numpy.array(x).reshape(-1, 1)
    y = numpy.array(y).reshape(-1, 1)
    model = LinearRegression()
    model.fit(X=x, y=y)

    pred = model.predict(x)

    plt.scatter(x, y)
    plt.plot(x, pred, color='green')
    plt.savefig('foo1.png')"""
    #df = pd.DataFrame(data=fonte_data, index=index, columns=["top_score_sum", "top_vote_sum", "BIC_score_sum", "BIC_vote_sum", "standard_iter", "weight_iter", "rank_BIC", "num_commits", "num_commits_log"])
    #df.to_csv("./total_data.csv")