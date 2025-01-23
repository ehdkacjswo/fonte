import os, json, argparse, pickle, sys, itertools, subprocess, logging
import pandas as pd
from tqdm import tqdm
from interval import interval

sys.path.append('/root/workspace/diff_util/lib/')
from diff import *
from gumtree import *

DIFF_DATA_DIR = '/root/workspace/data/Defects4J/diff'
CORE_DATA_DIR = '/root/workspace/data/Defects4J/core'
BASE_DATA_DIR = '/root/workspace/data/Defects4J/baseline'
RESULT_DATA_DIR = '/root/workspace/data/Defects4J/result'

sys.path.append('/root/workspace/lib/')
from experiment_utils import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute commit scores")
    parser.add_argument('--project', '-p', type=str, default="Cli",
        help="Target project name (default: Closure)")
    parser.add_argument('--version', '-v', type=int, default=29,
        help="Target buggy version (default: 21)")
    args = parser.parse_args()

    #logging.basicConfig(filename="/root/workspace/diff_util/collector/", level=logging.INFO)

    print('Working on {}_{}b'.format(args.project, args.version))
    with open('/root/workspace/error.txt', 'a') as file:
        file.write('Working on {}_{}b\n'.format(args.project, args.version))

    savedir = os.path.join(RESULT_DATA_DIR, f'{args.project}-{args.version}b', 'vote')
    os.makedirs(savedir, exist_ok=True)

    fonte_iter, bug2commit_iter, ensemble_iter = bisection_all(args.project, str(args.version))

    print(fonte_iter)
    print(bug2commit_iter)
    print(ensemble_iter)