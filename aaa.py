import argparse, os
import matplotlib.pyplot as plt
from lib.experiment_utils import *

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

    # Print BIC for current version
    GT = load_BIC_GT("./data/Defects4J/BIC_dataset")
    
    tot = 0
    cnt = 0
    fonte_data = []
    index = []

    same = 0
    better = 0
    worse = 0

    new_same = 0
    new_better = 0
    new_worse = 0

    for folder in os.listdir('data/Defects4J/core/'):
        coredir = os.path.join('data/Defects4J/core/', folder)
        bug2commit(coredir.replace('core', 'baseline'))