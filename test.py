import argparse, os
import matplotlib.pyplot as plt
from lib.experiment_utils import *
from sklearn.linear_model import LinearRegression
from scipy.stats import wilcoxon, ttest_rel
import numpy

voting_functions = {
    # key: (alpha, tau)
    (1, 'max'): (lambda r: r.score/r.max_rank),
    (0, 'max'): (lambda r: 1/r.max_rank),
    (1, 'dense'): (lambda r: r.score/r.dense_rank),
    (0, 'dense'): (lambda r: 1/r.dense_rank),
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute commit scores")
    parser.add_argument('coredir', help="data/Defects4J/core/<pid>-<vid>b")
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
    args = parser.parse_args()

    assert args.alpha in [0, 1]
    assert args.tau in ["max", "dense"]
    assert 0 <= args.lamb < 1

    fonte(args)

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