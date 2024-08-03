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
    
    better_count = 0
    worse_count = 0
    better_perc = 0
    worse_perc = 0

    new_same = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    new_better = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    new_worse = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    new_better_count = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    new_worse_count = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    new_better_perc = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    new_worse_perc = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    #args.skip_stage_2 = True

    for folder in os.listdir('data/Defects4J/core/'):
        coredir = os.path.join('data/Defects4J/core/', folder)
        tot = tot + 1

        # Get BIC
        [pid, vid] = coredir[20:-1].split("-")
        fault = (pid, vid)
        BIC = GT.set_index(["pid", "vid"]).loc[fault, "commit"]

        if args.skip_stage_2:
            style_change_commits = []
        else:
            style_change_commits = get_style_change_commits(coredir, args.tool, with_Rewrite=True)
        
        """fonte_data.append(data_gen(coredir, args.tool, args.formula,
        args.lamb, voting_functions[(args.alpha, args.tau)], BIC,
        use_method_level_score=False,
        excluded=style_change_commits, adjust_depth=True))

        index.append(coredir)"""
        
        # Vote for commits
        vote_df = vote_for_commits(coredir, args.tool, args.formula,
            args.lamb, voting_functions[(args.alpha, args.tau)],
            use_method_level_score=False,
            excluded=style_change_commits, adjust_depth=True, HSFL=True)
        
        # Get the candidate list of commits
        all_commits = get_all_commits(coredir)
        C_BIC = [
            c for c in all_commits
            if c in vote_df.index and c not in style_change_commits
        ]

        scores = [float(vote_df.loc[c, "vote"]) for c in C_BIC]
        BIC_index = C_BIC.index(BIC)

        standard_iter = standard_bisection(C_BIC, BIC)
        weight_iter = weighted_bisection(C_BIC, scores, BIC)

        new_scores = copy.deepcopy(scores)
        new_scores.sort()

        a = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        for i in range(len(a)):
            anew_scores = [new_scores[0] if s < new_scores[int(len(new_scores) * a[i])] else s for s in scores]
            new_iter = weighted_bisection(C_BIC, anew_scores, BIC)

            if new_iter == standard_iter:
                new_same[i] = new_same[i] + 1
            elif new_iter > standard_iter:
                new_worse[i] = new_worse[i] + 1
                new_worse_count[i] = new_worse_count[i] + standard_iter - new_iter
                new_worse_perc[i] = new_worse_perc[i] + (standard_iter - new_iter) / standard_iter * 100
            else:
                new_better[i] = new_better[i] + 1
                new_better_count[i] = new_better_count[i] + standard_iter - new_iter
                new_better_perc[i] = new_better_perc[i] + (standard_iter - new_iter) / standard_iter * 100

        #if new_iter != weight_iter:
            #print(coredir, standard_iter, weight_iter, new_iter)

        #print(coredir, get_the_number_of_total_commits(coredir), standard_iter, weight_iter, new_iter)

        if standard_iter == weight_iter:
            same = same + 1
        elif standard_iter > weight_iter:
            better = better + 1
            better_count = better_count + standard_iter - weight_iter
            better_perc = better_perc + (standard_iter - weight_iter) / standard_iter * 100
        else:
            worse = worse + 1
            worse_count = worse_count + standard_iter - weight_iter
            worse_perc = worse_perc + (standard_iter - weight_iter) / standard_iter * 100

        """plt.figure(figsize=(6, 2.5))
        plt.title(fault)
        plt.bar(range(len(scores)), scores, color=[
            "red" if i == BIC_index else "green" for i in range(len(scores))])
        plt.ylabel("Score")
        plt.xlabel("Commit Index (in Desending Order of Time)")

        plt.savefig('./fig2/{}.png'.format(folder))
        plt.close()"""
    
    print(better, same, worse)
    print(better_count / better, worse_count / worse, (better_count + worse_count) / (better + same + worse))
    print(better_perc / better, worse_perc / worse, (better_perc + worse_perc) / (better + same + worse))

    for i in range(len(new_better)):
        print(i * 0.1)
        print(new_better[i], new_same[i], new_worse[i])
        print(new_better_count[i] / new_better[i], new_worse_count[i] / new_worse[i], (new_better_count[i] + new_worse_count[i]) / (new_better[i] + new_same[i] + new_worse[i]))
        print(new_better_perc[i] / new_better[i], new_worse_perc[i] / new_worse[i], (new_better_perc[i] + new_worse_perc[i]) / (new_better[i] + new_same[i] + new_worse[i]))

    #df = pd.DataFrame(data=fonte_data, index=index, columns=["top_score_sum", "top_vote_sum", "BIC_score_sum", "BIC_vote_sum", "standard_iter", "weight_iter", "rank_BIC", "num_commits", "num_commits_log"])
    #df.to_csv("./total_data.csv")