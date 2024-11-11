import os, math, copy, json, sys
import numpy as np
import pandas as pd
from sbfl.base import SBFL

sys.path.append('/root/workspace/diff_util/lib/')
from encoder import savepath_postfix

def load_BIC_GT(dataset_dir):
    def load_BIC_data(filename):
        df = pd.read_csv(os.path.join(dataset_dir, filename), header=0)
        df["vid"] = df["vid"].astype(str)
        df["commit"] = df["commit"].apply(lambda s: str(s)[:7])
        return df

    Wen_GT = load_BIC_data(
        "wen19-defects4j-bug-inducing-commits.csv")
    # print("Wen (Original)", len(Wen_GT))
    excluded_from_Wen = load_BIC_data(
        "excluded-from-Wen.csv"
    )[["pid", "vid"]].to_records(index=False).tolist()
    to_exclude = Wen_GT.apply(lambda row:
        (row.pid, row.vid) in excluded_from_Wen, axis=1)
    Wen_GT = Wen_GT[~to_exclude]
    # print("Wen (Filtered)", len(Wen_GT))

    manual_GT = load_BIC_data(
        "manual-defects4j-bug-inducing-commits.csv").drop_duplicates()
    # print("Manual", len(manual_GT))

    # Check for the overlapped data points
    overlap_counts = 0
    for _, row in Wen_GT.iterrows():
        cor_row = manual_GT[
            (manual_GT.pid == row.pid) & (manual_GT.vid == row.vid)]
        if cor_row.shape[0] > 0:
            found_by_Wen = row.commit
            found_manually =  cor_row.commit.values[0]
            assert found_by_Wen == found_manually
            overlap_counts += 1

    GT = pd.concat([Wen_GT, manual_GT])
    GT = GT.drop_duplicates()
    assert GT[["pid", "vid"]].drop_duplicates().shape[0] == GT.shape[0]
    print(f"The BIC data is available for {len(GT)} faults. ({len(Wen_GT)} Wen + {len(manual_GT)} Manual - {overlap_counts} Overlapped)")

    GT.groupby(["pid"]).count()["vid"]
    GT.sort_values(by=["pid", "vid"], inplace=True)

    GT = GT.set_index(["pid", "vid"])
    Wen_GT = Wen_GT.set_index(["pid", "vid"])
    manual_GT = manual_GT.set_index(["pid", "vid"])

    GT["provenance"] = ""
    GT.loc[Wen_GT.index, "provenance"] += "Wen+"
    GT.loc[manual_GT.index, "provenance"] += "Manual"
    GT["provenance"] = GT["provenance"].apply(
        lambda s: s[:-1] if s[-1:] == "+" else s)
    GT = GT.reset_index()
    
    savepath = os.path.join(dataset_dir, "combined.csv")
    GT.to_csv(savepath, index=False)
    print("The combined GT data is saved to", savepath)
    return GT

def get_sbfl_scores_from_coverage(path_to_coverage, formula="Ochiai",
    covered_by_failure_only=True, in_class_only=False, use_cache=True,
    return_coverage_matrix=False):
    assert not (use_cache and return_coverage_matrix)
    savedir = os.path.join(os.path.dirname(path_to_coverage), "sbfl")
    postfix = "suspicious" if covered_by_failure_only else "all"
    postfix += "_inClassOnly" if in_class_only else ""
    savepath = os.path.join(savedir, f"{formula}_{postfix}.pkl")
    if use_cache and os.path.exists(savepath):
        sbfl_df = pd.read_pickle(savepath)
    else:
        if in_class_only:
            # only use the test cases that are in the class containing
            # at least one failing test case
            new_path_to_coverage = path_to_coverage.replace(
                ".pkl", ".in_class_only.pkl")

            if os.path.exists(new_path_to_coverage):
                cov_df = pd.read_pickle(new_path_to_coverage)
            else:
                cov_df = pd.read_pickle(path_to_coverage)

                failing_tests = cov_df.index[~cov_df["result"]]
                failing_test_classes = set(
                    [t.split("::")[0] for t in failing_tests])
                in_class = np.array([
                    t.split("::")[0] in failing_test_classes for t in cov_df.index])
                cov_df = cov_df.loc[in_class]
                cov_df.to_pickle(new_path_to_coverage)
        else:
            cov_df = pd.read_pickle(path_to_coverage)
        is_passing = cov_df["result"].values.astype(bool) # test results
        cov_df.drop("result", axis=1, inplace=True)

        if covered_by_failure_only:
            cov_df = cov_df.loc[:, cov_df.loc[~is_passing].any(axis=0)]
        sbfl = SBFL(formula=formula)
        sbfl.fit(cov_df.values, is_passing)
        sbfl_df = sbfl.to_frame(elements=cov_df.columns,
                names=["class_file", "method_name", "method_signature", "line"])
        if not os.path.exists(savedir):
            os.mkdir(savedir)
        sbfl_df.to_pickle(savepath)
    if return_coverage_matrix:
        return sbfl_df, cov_df
    else:
        return sbfl_df

def get_all_commits(fault_dir):
    with open(os.path.join(fault_dir, "commits.log"), "r") as f:
        return list(map(lambda hash: hash[:7], f.read().strip().split('\n')))

def get_the_number_of_total_commits(fault_dir):
    return len(get_all_commits(fault_dir))

def load_commit_history(fault_dir, tool):
    com_df = pd.read_pickle(os.path.join(fault_dir, tool, "commits.pkl"))
    if "class_file" not in com_df.columns:
        with open(os.path.join(fault_dir, "src_dir"), "r") as f:
            src_dir = f.read().strip()
            if src_dir[-1] != "/":
                src_dir += "/"
        com_df["class_file"] = com_df["src_path"].apply(
            lambda s: s[len(src_dir):]
        )
    com_df["commit_hash"] = com_df["commit_hash"].apply(lambda s: str(s)[:7])
    return com_df

def get_style_change_commits(fault_dir, tool, with_Rewrite=True):
    postfix = "" if with_Rewrite else "_noOpenRewrite"
    val_df = pd.read_csv(
        os.path.join(fault_dir, tool, f"validation{postfix}.csv"), 
        header=None,
        names=["commit", "src_path", "AST_diff"])

    val_df["unchanged"] = val_df["AST_diff"] == "U"
    agg_df = val_df.groupby("commit").all()[["unchanged"]]
    return agg_df.index[agg_df["unchanged"]].tolist()

# Extra score
def extra_score(data_dir, use_diff=True, tool='git', skip_stage_2=False, with_Rewrite=True, \
    use_stopword=True, adddel='all', encode_type='simple', norm_mode='score'):
    score_dict = dict()

    # Use Bug2Commit
    file_postfix = savepath_postfix(tool, skip_stage_2, with_Rewrite, use_stopword)
    diff_prefix = 'diff_' if use_diff else ''
    savepath = os.path.join(data_dir, f'{diff_prefix}ranking{file_postfix}.csv')
    df = pd.read_csv(savepath, names=["commit_hash", "rank", "score"])

    # No normalization
    if norm_mode == 'base':
        for ind in df.index:
            score_dict[df['commit_hash'][ind]] = df['score'][ind]

    # Rank
    elif norm_mode == 'rank':
        for ind in df.index:
            score_dict[df['commit_hash'][ind]] = 1 / df['rank'][ind]

    # Softmax
    elif norm_mode == 'softmax':
        score_sum = 0

        for ind in df.index:
            norm_score = math.exp(df['score'][ind])
            score_sum = score_sum + norm_score
            score_dict[df['commit_hash'][ind]] = norm_score

        if score_sum != 0:
            for commit_hash, commit_score in score_dict.items():
                score_dict[commit_hash] = commit_score / score_sum

    else:
        max_score = df['score'].max()
        for ind in df.index:
            score_dict[df['commit_hash'][ind]] = math.exp(df['score'][ind] - max_score)
    
    return score_dict

# HSFL parameter means the score will be regularized similarly as HSFL
def vote_for_commits(fault_dir, tool, formula, decay, voting_func,
    use_method_level_score=False, adjust_depth=True,
    in_class_only=False, HSFL=True, beta=0.0, \
    use_diff=True, skip_stage_2=False, with_Rewrite=True, use_stopword=True, adddel='all'):
    
    # Get commit history info
    commit_df = load_commit_history(fault_dir, tool)
    if skip_stage_2:
        excluded = []
    else:
        excluded = get_style_change_commits(fault_dir, tool, with_Rewrite=with_Rewrite)

    commit_df["excluded"] = commit_df["commit_hash"].isin(excluded)
    commit_df["new_depth"] = commit_df["depth"]

    # Adjust depth of commits by ignoring excluded commits
    if len(excluded) > 0 and adjust_depth:
        commit_df.loc[commit_df.excluded, "new_depth"] = None
        commit_df["method_identifier"] = commit_df.class_file + ":" + \
            commit_df.method_name + commit_df.method_signature + \
            ":L" + commit_df.begin_line.astype(str) + "," + commit_df.end_line.astype(str)
        for _, row in commit_df[commit_df.excluded].iterrows():
            # print(row)
            affected = (commit_df.method_identifier == row.method_identifier)\
                & (commit_df.depth > row.depth)
            commit_df.loc[affected, "new_depth"] = commit_df.loc[affected, "new_depth"] - 1

    # Get SBFL score
    sbfl_df = get_sbfl_scores_from_coverage(
        os.path.join(fault_dir, "coverage.pkl"),
        formula=formula,
        covered_by_failure_only=True,
        in_class_only=in_class_only)

    # For method level score, each statements votes for the method
    # And score of method will be sum of the votes from the statements
    if use_method_level_score:
        identifier = ["class_file", "method_name", "method_signature","begin_line", "end_line"]
        l_sbfl_df = sbfl_df.reset_index()
        l_sbfl_df["dense_rank"] = (-l_sbfl_df["score"]).rank(method="dense")
        l_sbfl_df["max_rank"] = (-l_sbfl_df["score"]).rank(method="max")

        l_sbfl_df["score"] = l_sbfl_df.apply(voting_func, axis=1)

        method_sbfl_rows = []
        for _, method in commit_df[identifier].drop_duplicates().iterrows():
            method_score = l_sbfl_df[
                (l_sbfl_df.class_file == method.class_file)
                # & (l_sbfl_df.method_name == method.method_name)
                # & (l_sbfl_df.method_signature == method.method_signature)   
                & (l_sbfl_df.line >= method.begin_line)
                & (l_sbfl_df.line <= method.end_line)
            ].score.sum()
            method_sbfl_rows.append([
                method.class_file, method.method_name, method.method_signature,
                method.begin_line, method.end_line, method_score
            ])
        sbfl_df = pd.DataFrame(method_sbfl_rows,columns=identifier+["score"])
        sbfl_df = sbfl_df.set_index(identifier)

    # Get the rank based on SBFL scores
    sbfl_df["dense_rank"] = (-sbfl_df["score"]).rank(method="dense")
    sbfl_df["max_rank"] = (-sbfl_df["score"]).rank(method="max")
    
    extra_score_dict = extra_score(fault_dir.replace('core', 'baseline'), use_diff=use_diff, skip_stage_2=skip_stage_2, with_Rewrite=with_Rewrite, use_stopword=use_stopword, adddel=adddel)
    vote_dict = dict()
    vote_rows = []
    total_vote = 0
    BIC_vote = 0

    for _, row in sbfl_df.reset_index().iterrows():
        vote = voting_func(row)
        if use_method_level_score:
            com_df = commit_df[
                (commit_df.class_file == row.class_file) \
                & (commit_df.method_name == row.method_name) \
                & (commit_df.method_signature == row.method_signature)
            ]
        else:
            com_df = commit_df[
                (commit_df.class_file == row.class_file) \
                & (commit_df.begin_line <= row.line) \
                & (commit_df.end_line >= row.line)
            ]
        
        induce_sqrt = math.sqrt(com_df.shape[0])
        induce_sqrt = 0
        list_commits = []
        sum_score = 0

        # Find commits that modified the statement(method) and not excluded
        for commit, depth in zip(com_df.commit_hash, com_df.new_depth):
            if commit not in excluded:
                list_commits.append((commit, depth))
                sum_score = sum_score + extra_score_dict.get(commit, min(extra_score_dict.values(), default=1))
                """if commit not in extra_score_dict:
                    print('FBL-BERT key doesn\'t exists : {} {}'.format(commit,fault_dir))"""
        
        induce_sqrt = math.sqrt(len(list_commits))
        #sum_score = 1 if score is None else sum_score / induce_sqrt
        sum_score = 1
        
        # Vote for commits
        for commit, depth in list_commits:
            if HSFL:
                decayed_vote = vote * ((1-decay) ** depth) * sum_score / induce_sqrt
                #decayed_vote = (vote * beta + extra_score_dict.get(commit, min(extra_score_dict.values(), default=0)) * (1 - beta)) * ((1-decay) ** depth) * sum_score / induce_sqrt
            else:
                decayed_vote = vote * ((1-decay) ** depth) * sum_score
                #decayed_vote = (vote * beta + extra_score_dict.get(commit, min(extra_score_dict.values(), default=0)) * (1 - beta)) * ((1-decay) ** depth) * sum_score
            vote_dict[commit] = vote_dict.get(commit, 0) + decayed_vote
                

        """decayed_vote = vote * ((1-decay) ** depth)
                list_commits.append(commit)
        
        for commit in list_commits:
            induce_sqrt = induce_sqrt + """

    # Apply extra score
    for commit, vote in vote_dict.items():
        vote_rows.append([commit, vote * extra_score_dict.get(commit, min(extra_score_dict.values(), default=1))])
        #vote_rows.append([commit, vote])
        #if commit not in extra_score_dict.keys():
            #print(fault_dir, commit)

    vote_df = pd.DataFrame(data=vote_rows, columns=["commit", "vote"])
    agg_vote_df = vote_df.groupby("commit").sum("vote")
    agg_vote_df.sort_values(by="vote", ascending=False, inplace=True)

    return agg_vote_df

def max_aggr_for_commits(fault_dir, tool, formula,
    use_method_level_score=False, excluded=[]):
    commit_df = load_commit_history(fault_dir, tool)

    sbfl_df = get_sbfl_scores_from_coverage(
        os.path.join(fault_dir, "coverage.pkl"),
        formula=formula,
        covered_by_failure_only=True)

    if use_method_level_score:
        identifier = ["class_file", "method_name", "method_signature","begin_line", "end_line"]
        l_sbfl_df = sbfl_df.reset_index()
        method_sbfl_rows = []
        for _, method in commit_df[identifier].drop_duplicates().iterrows():
            method_score = l_sbfl_df[
                (l_sbfl_df.class_file == method.class_file)
                & (l_sbfl_df.line >= method.begin_line)
                & (l_sbfl_df.line <= method.end_line)
            ].score.max()
            method_sbfl_rows.append([
                method.class_file, method.method_name, method.method_signature,
                method.begin_line, method.end_line, method_score
            ])
        sbfl_df = pd.DataFrame(method_sbfl_rows,columns=identifier+["score"])
        sbfl_df = sbfl_df.set_index(identifier)

    vote_rows = []
    for _, row in sbfl_df.reset_index().iterrows():
        vote = row.score
        if use_method_level_score:
            com_df = commit_df[
                (commit_df.class_file == row.class_file) \
                & (commit_df.method_name == row.method_name) \
                & (commit_df.method_signature == row.method_signature)
                & (commit_df.begin_line == row.begin_line) \
                & (commit_df.end_line == row.end_line)
            ]
        else:
            com_df = commit_df[
                (commit_df.class_file == row.class_file) \
                & (commit_df.begin_line <= row.line) \
                & (commit_df.end_line >= row.line)
            ]

        for commit in com_df.commit_hash.unique():
            vote_rows.append([commit, vote if commit not in excluded else 0])
    vote_df = pd.DataFrame(data=vote_rows, columns=["commit", "vote"])
    agg_vote_df = vote_df.groupby("commit").max("vote")
    agg_vote_df.loc[agg_vote_df.vote.isna(), "vote"] = .0
    agg_vote_df.sort_values(by="vote", ascending=False, inplace=True)
    return agg_vote_df

def standard_bisection(commits: list, BIC, verbose=False, return_pivots=False):
    assert BIC in commits
    # pre-condition: commit[i] is newer than commit[i+1]
    # return: the number of required iterations until the BIC is found

    BIC_index = commits.index(BIC)
    bad_index = 0
    good_index = len(commits)
    num_iterations = 0
    if return_pivots:
        pivots = []
    while good_index > bad_index + 1:
        num_iterations += 1
        pivot_index = int((bad_index + good_index)/2)
        if verbose:
            print(f"Test {pivot_index}")
        if return_pivots:
            pivots.append(pivot_index)
        if pivot_index > BIC_index:
            good_index = pivot_index
        elif pivot_index <= BIC_index:
            bad_index = pivot_index
    assert bad_index == BIC_index
    if return_pivots:
        return num_iterations, pivots
    else:
        return num_iterations

def weighted_bisection(commits: list, scores: list, BIC, verbose=False,
    return_pivots=False):
    assert BIC in commits
    assert len(commits) == len(scores)
    # pre-condition: commit[i] is newer than commit[i+1]
    # pre-condition: score[i] is the score of commit[i]
    # return: the number of required iterations until the BIC is found
    commits = [c for c, s in zip(commits, scores) if s > 0]
    scores = [s for s in scores if s > 0]
    BIC_index = commits.index(BIC)
    bad_index = 0
    good_index = len(commits)
    num_iterations = 0
    if return_pivots:
        pivots = []
    while good_index > bad_index + 1:
        num_iterations += 1
        """
        original implementation
        """
        min_diff, pivot_index = None, None
        for i in range(bad_index + 1, good_index):
            abs_diff = abs(
                sum(scores[bad_index:i]) - sum(scores[i:good_index]))
            if min_diff is None or min_diff > abs_diff:
                min_diff = abs_diff
                pivot_index = i
            else:
                # already pass the min, now increasing
                break
        if verbose:
            print(f"Test {pivot_index}")
        if return_pivots:
            pivots.append(pivot_index)
        if pivot_index > BIC_index:
            good_index = pivot_index
        elif pivot_index <= BIC_index:
            bad_index = pivot_index
    assert bad_index == BIC_index
    if return_pivots:
        return num_iterations, pivots
    else:
        return num_iterations

# Voting functions
voting_functions = {
    # key: (alpha, tau)
    (1, 'max'): (lambda r: r.score/r.max_rank),
    (0, 'max'): (lambda r: 1/r.max_rank),
    (1, 'dense'): (lambda r: r.score/r.dense_rank),
    (0, 'dense'): (lambda r: 1/r.dense_rank),
}

# Run fonte for every projects
# [C_BIC], [scores], [BIC], [BIC_rank]
def fonte(args, HSFL=True, use_diff=True, skip_stage_2=False, with_Rewrite=True,\
    use_stopword=True, adddel='all', ignore=[0]):

    # Return values
    C_BIC_list = []
    scores_list = []
    BIC_list = []
    BIC_rank_list = []

    # Load BIC data
    GT = load_BIC_GT("./data/Defects4J/BIC_dataset")

    # Iterate through every projects
    for folder in os.listdir('data/Defects4J/core/'):
        
        coredir = os.path.join('data/Defects4J/core/', folder)

        # Get BIC data
        [pid, vid] = coredir[20:-1].split("-")

        # diff data not fully handled
        """if not os.path.isfile(os.path.join('data/Defects4J/baseline/{}-{}b'.format(pid, vid), 'ranking_diff_Bug2Commit.csv')):
            continue"""

        fault = (pid, vid)
        BIC = GT.set_index(["pid", "vid"]).loc[fault, "commit"]
        BIC_list.append(BIC)

        if args.skip_stage_2:
            style_change_commits = []
        else:
            style_change_commits = get_style_change_commits(coredir, args.tool, with_Rewrite=True)

        vote_df = vote_for_commits(coredir, args.tool, args.formula,
            args.lamb, voting_functions[(args.alpha, args.tau)],
            use_method_level_score=False, adjust_depth=True, HSFL=HSFL,
            use_diff=use_diff, skip_stage_2=args.skip_stage_2, with_Rewrite=with_Rewrite, \
            use_stopword=use_stopword, adddel=adddel)
    
        # Get the candidate list of commits
        all_commits = get_all_commits(coredir)
        C_BIC = [
            c for c in all_commits
            if c in vote_df.index and c not in style_change_commits
        ]
        C_BIC_list.append(C_BIC)

        # Get the rank of BIC
        vote_df["rank"] = (-vote_df["vote"]).rank(method=args.tau)
        BIC_rank_list.append(int(vote_df.loc[BIC, "rank"]))
        
        # Get number of iterations
        scores = [float(vote_df.loc[c, "vote"]) for c in C_BIC]
        scores_list.append(scores)
    
    return C_BIC_list, scores_list, BIC_list, BIC_rank_list

# Check whether the methods that were modified by more likely BICs are likely to be buggy
# Statement level is impossible since the projects are re-written by OpenRewrite
# Return : Rank of buggy method based on
# Score methods : SBFL, ensemble_max, ensemble_min
# Rank methods : dense, max
# Result methods : min, max
def aaa(args, use_method_level_score=True, score='bug2commit', adjust_depth=True, in_class_only=False):

    ##### Step 1 : Initialization #####
    # Load 
    GT = load_BIC_GT("./data/Defects4J/BIC_dataset")

    # Load buggy method info
    with open("data/Defects4J/buggy_methods.json", "r") as f:
        buggy_method_infos = json.load(f)

    # Get the list of fault directories
    CORE_DATA_DIR = "./data/Defects4J/core"

    fault_dirs = {}
    for fault in os.listdir(CORE_DATA_DIR):
        fault_dir = os.path.join(CORE_DATA_DIR, fault)
        if not os.path.isdir(fault_dir):
            continue
        pid, vid = fault.split('-')
        fault_dirs[(pid, vid[:-1])] = fault_dir

    # Load buggy method infos
    results = [[], [], []]

    for _, row in GT.iterrows():
        fault = (row.pid, row.vid)

        # No data for current fault
        if fault not in fault_dirs:
            continue
        
        if use_method_level_score:
            # Load info of buggy methods for current fault
            buggy_methods = [
                (bm["class_file"], bm["method_name"], bm["arg_types"])
                for bm in buggy_method_infos[f"{row.pid}-{row.vid}b"]
            ]

            # No buggy method info
            if len(buggy_methods) == 0:
                continue
        
        """else:
            # Load info of buggy statements for current fault
            try:
                with open("./data/Defects4J/buggy-lines/{}-{}.buggy.lines".format(row.pid, row.vid), "r") as f:
                    buggy_lines = [(buggy_line.split('#')[0], int(buggy_line.split('#')[1])) for buggy_line in f.readlines()]
                    print(buggy_lines)

            # No buggy method info
            except:
                print('No such file ./data/Defects4J/buggy-lines/{}-{}.buggy.lines'.format(row.pid, row.vid))
                continue"""

        fault_dir = fault_dirs[fault]

        # Get style change commits (to exclude)
        if args.skip_stage_2:
            excluded = []
        else:
            excluded = get_style_change_commits(fault_dir, args.tool, with_Rewrite=True)
        
        # Load commit history info
        commit_df = load_commit_history(fault_dir, args.tool)

        commit_df["excluded"] = commit_df["commit_hash"].isin(excluded)
        commit_df["new_depth"] = commit_df["depth"]

        # Adjust depth of commits by ignoring excluded commits
        if len(excluded) > 0 and adjust_depth:
            commit_df.loc[commit_df.excluded, "new_depth"] = None
            commit_df["method_identifier"] = commit_df.class_file + ":" + \
                commit_df.method_name + commit_df.method_signature + \
                ":L" + commit_df.begin_line.astype(str) + "," + commit_df.end_line.astype(str)
            for _, row in commit_df[commit_df.excluded].iterrows():
                # print(row)
                affected = (commit_df.method_identifier == row.method_identifier)\
                    & (commit_df.depth > row.depth)
                commit_df.loc[affected, "new_depth"] = commit_df.loc[affected, "new_depth"] - 1

        ##### Step 2 : SBFL #####
        # Get SBFL score
        sbfl_df = get_sbfl_scores_from_coverage(
            os.path.join(fault_dir, "coverage.pkl"),
            formula=args.formula,
            covered_by_failure_only=True,
            in_class_only=in_class_only)

        # For method level score, each statements votes for the method
        # And score of method will be sum of the votes from the statements
        if use_method_level_score:
            identifier = ["class_file", "method_name", "method_signature","begin_line", "end_line"]
            l_sbfl_df = sbfl_df.reset_index()
            
            # Apply voting function to get score for each statements
            l_sbfl_df["dense_rank"] = (-l_sbfl_df["score"]).rank(method="dense")
            l_sbfl_df["max_rank"] = (-l_sbfl_df["score"]).rank(method="max")

            l_sbfl_df["score"] = l_sbfl_df.apply(voting_functions[(args.alpha, args.tau)], axis=1)

            # Evaluate score of methods by adding scores of statements
            method_sbfl_rows = []
            for _, method in commit_df[identifier].drop_duplicates().iterrows():
                method_score = l_sbfl_df[
                    (l_sbfl_df.class_file == method.class_file)
                    & (l_sbfl_df.line >= method.begin_line)
                    & (l_sbfl_df.line <= method.end_line)
                ].score.sum()
                method_sbfl_rows.append([
                    method.class_file, method.method_name, method.method_signature,
                    method.begin_line, method.end_line, method_score
                ])

            sbfl_df = pd.DataFrame(method_sbfl_rows, columns=identifier+["score"])
            sbfl_df = sbfl_df.set_index(identifier)

        ##### Step 3 : Ensemble #####
        # Ensemble the original SBFL score with extra score
        extra_score_dict = extra_score(fault_dir.replace('core', 'baseline'), score=score, norm_mode='rank')

        ensemble_max_rows = []
        ensemble_sum_rows = []

        for _, row in sbfl_df.reset_index().iterrows():

            # Commits modified the method (statement)
            if use_method_level_score:
                com_df = commit_df[
                    (commit_df.class_file == row.class_file) \
                    & (commit_df.method_name == row.method_name) \
                    & (commit_df.method_signature == row.method_signature)
                ]

            """else:
                com_df = commit_df[
                    (commit_df.class_file == row.class_file) \
                    & (commit_df.begin_line <= row.line) \
                    & (commit_df.end_line >= row.line)
                ]"""

            # Get the extra scores from commits
            max_vote = 0
            sum_vote = 0
            num_commits = 0
            
            for commit, depth in zip(com_df.commit_hash, com_df.new_depth):
                if commit not in excluded:
                    vote = extra_score_dict.get(commit, min(extra_score_dict.values(), default=1))

                    max_vote = max(max_vote, vote)
                    sum_vote = sum_vote + vote
                    num_commits = num_commits + 1
            
            # Apply normalization
            num_commits_sqrt = math.sqrt(num_commits)

            if use_method_level_score:
                #ensemble_max_rows.append([row.class_file, row.method_name, row.method_signature, row.score * (max_vote * num_commits)])
                #ensemble_sum_rows.append([row.class_file, row.method_name, row.method_signature, row.score * sum_vote])
                ensemble_max_rows.append([row.class_file, row.method_name, row.method_signature, max_vote * num_commits])
                ensemble_sum_rows.append([row.class_file, row.method_name, row.method_signature, sum_vote])
            
            """else:
                ensemble_max_rows.append([row.class_file, row.line, row.score * (max_vote * num_commits)])
                ensemble_sum_rows.append([row.class_file, row.line, row.score * sum_vote])"""
        
        # Save the data as DataFrame
        if use_method_level_score:
            ensemble_max_df = pd.DataFrame(data=ensemble_max_rows, columns=["class_file", "method_name", "method_signature", "score"])
            ensemble_sum_df = pd.DataFrame(data=ensemble_sum_rows, columns=["class_file", "method_name", "method_signature", "score"])
        
        """else:
            ensemble_max_df = pd.DataFrame(data=ensemble_max_rows, columns=["class_file", "line", "score"])
            ensemble_sum_df = pd.DataFrame(data=ensemble_sum_rows, columns=["class_file", "line", "score"])"""
        
        ensemble_max_df.sort_values(by="score", ascending=False, inplace=True)
        ensemble_sum_df.sort_values(by="score", ascending=False, inplace=True)

        ##### Step 4 : Get rank #####
        # Get the ranks for each scores (SBFL, ensemble_max, ensemble_sum)
        score_data_list = [sbfl_df, ensemble_max_df, ensemble_sum_df]
        
        for i in range(len(score_data_list)):
            score_data_list[i]["dense_rank"] = (-score_data_list[i]["score"]).rank(method="dense")
            score_data_list[i]["max_rank"] = (-score_data_list[i]["score"]).rank(method="max")
            score_data_list[i]["dense_rank_perc"] = (score_data_list[i]["dense_rank"]-1)/score_data_list[i].shape[0]
            score_data_list[i]["max_rank_perc"] = (score_data_list[i]["max_rank"]-1)/score_data_list[i].shape[0]

        # Get the rank of buggy method (statement)
        for i in range(len(score_data_list)):
            rank_df = score_data_list[i].reset_index()

            if use_method_level_score:
                rank_df["arg_types"] = rank_df["method_signature"].apply(
                    lambda s: s.split(')')[0][1:]
                )
                rank_df = rank_df.set_index(["class_file", "method_name", "arg_types"])[
                    ["dense_rank", "max_rank", "dense_rank_perc", "max_rank_perc"]]

                rank_df.sort_index()
                new_result = [math.inf, 0, math.inf, 0]
                
                for bm in buggy_methods:
                    if bm in rank_df.index:
                        new_result[0] = min(new_result[0], rank_df.loc[bm, "dense_rank"].min())
                        new_result[1] = max(new_result[1], rank_df.loc[bm, "dense_rank"].max())
                        new_result[2] = min(new_result[2], rank_df.loc[bm, "max_rank"].min())
                        new_result[3] = max(new_result[3], rank_df.loc[bm, "max_rank"].max())
                
                results[i].append(new_result)

            """else:
                rank_df = rank_df.set_index(["class_file", "line"])[
                    ["dense_rank", "max_rank", "dense_rank_perc", "max_rank_perc"]]
                
                rank_df.sort_index()
                new_result = [math.inf, 0, math.inf, 0]

                # Some buggy lines are not found
                for bl in buggy_lines:
                    if bl in rank_df.index:
                        new_result[0] = min(new_result[0], rank_df.loc[bl, "dense_rank"].min())
                        new_result[1] = max(new_result[1], rank_df.loc[bl, "dense_rank"].max())
                        new_result[2] = min(new_result[2], rank_df.loc[bl, "max_rank"].min())
                        new_result[3] = max(new_result[3], rank_df.loc[bl, "max_rank"].max())
                if new_result[1] == 0:
                    continue
                results[i].append(new_result)"""
    
    return results