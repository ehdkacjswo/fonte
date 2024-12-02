import os, math, copy, json, sys, itertools
import numpy as np
import pandas as pd
from sbfl.base import SBFL
from tqdm import tqdm

sys.path.append('/root/workspace/diff_util/lib/')
from encoder import savepath_postfix

DIFF_DATA_DIR = '/root/workspace/data/Defects4J/diff'
CORE_DATA_DIR = '/root/workspace/data/Defects4J/core'

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
    #print(f"The BIC data is available for {len(GT)} faults. ({len(Wen_GT)} Wen + {len(manual_GT)} Manual - {overlap_counts} Overlapped)")

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
    #print("The combined GT data is saved to", savepath)
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

def weighted_bisection(commits: list, scores: list, BIC, ignore_zero = True, verbose=False,
    return_pivots=False):
    assert BIC in commits
    assert len(commits) == len(scores)
    # pre-condition: commit[i] is newer than commit[i+1]
    # pre-condition: score[i] is the score of commit[i]
    # return: the number of required iterations until the BIC is found
    if ignore_zero:
        commits = [c for c, s in zip(commits, scores) if s > 0]

    if BIC not in commits:
        return None

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
        # Extra code for 0 score cases
        if not ignore_zero and scores[pivot_index] == 0:
            min_pivot_index = pivot_index
            max_pivot_index = pivot_index

            while scores[min_pivot_index] == 0 and min_pivot_index > bad_index + 1:
                min_pivot_index -= 1
            while scores[max_pivot_index] == 0 and max_pivot_index < good_index - 1:
                max_pivot_index += 1
            
            pivot_index = int((min_pivot_index + max_pivot_index) / 2)
            
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

# For a given project, generate dataframe with result scores of fonte for every settings
# Settings : ['HSFL', 'score_mode', 'ensemble', 'use_br', 'use_diff', 'stage2', 'use_stopword', 'adddel']
def score_eval_all(pid, vid, tool, formula, decay, voting_func,
    use_method_level_score=False, adjust_depth=True,
    in_class_only=False):

    fault_dir = os.path.join('/root/workspace/data/Defects4J/core', f'{pid}-{vid}b')
    extra_scores_df = pd.read_hdf(os.path.join('/root/workspace/data/Defects4J/diff', f'{pid}-{vid}b/scores.hdf'))

    # Possible settings
    use_br_list = [True, False]
    use_diff_list = [True, False]
    use_stopword_list = [True] # [True, False]
    adddel_list = ['add', 'del', 'all-uni', 'all-sep']
    extra_score_list = list(itertools.product(use_br_list, use_diff_list, use_stopword_list, adddel_list))

    score_mode_list = ['score', 'rank', 'both']
    ensemble_list = ['mul', ('add', 0.0), ('add', 0.1), ('add', 0.2), ('add', 0.3), ('add', 0.4), \
        ('add', 0.5), ('add', 0.6), ('add', 0.7), ('add', 0.8), ('add', 0.9), ('add', 1.0)]
    score_param_list = list(itertools.product(score_mode_list, ensemble_list))

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

    result_dict = dict()
    
    for stage2 in [True]: #['skip', True, False]
        # Get commit history info
        commit_df = load_commit_history(fault_dir, tool)
        if stage2 == 'skip':
            excluded = []
        else:
            excluded = get_style_change_commits(fault_dir, tool, with_Rewrite=stage2)

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
        
        for HSFL in [True, False]:
            vote_dict = dict()

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
                
                list_commits = []

                # Find commits that modified the statement(method) and not excluded
                for commit, depth in zip(com_df.commit_hash, com_df.new_depth):
                    if commit not in excluded:
                        list_commits.append((commit, depth))
                
                induce_sqrt = math.sqrt(len(list_commits))
                
                # Vote for commits
                for commit, depth in list_commits:
                    decayed_vote = vote * ((1-decay) ** depth) / (induce_sqrt if HSFL else 1)
                    vote_dict[commit] = vote_dict.get(commit, 0) + decayed_vote

            for (use_br, use_diff, use_stopword, adddel) in extra_score_list:
                extra_score_df = extra_scores_df.loc[(str(use_br), str(use_diff), str(stage2), str(use_stopword), str(adddel))]
                commit_hash_df = extra_score_df['commit_hash']
                score_df = extra_score_df['score']
                rank_df = extra_score_df['rank']

                for (score_mode, ensemble) in score_param_list:
                    result_key = (str(HSFL), str(score_mode), str(ensemble), str(use_br), str(use_diff), \
                        str(stage2), str(use_stopword), str(adddel))

                    if ensemble != 'mul':
                        if ensemble[1] == 1.0: # Use only original score
                            result_key = (str(HSFL), 'None', str(ensemble), 'None', 'None', str(stage2), 'None', 'None')

                        elif ensemble[1] == 0.0: # Use only extra score
                            result_key = ('None', str(score_mode), str(ensemble), str(use_br), str(use_diff), \
                                str(stage2), str(use_stopword), str(adddel))
                        
                        if result_key in result_dict: # Already done
                            continue

                    vote_rows = []
                
                    # Apply extra score
                    for commit, vote in vote_dict.items():
                        index = commit_hash_df.loc[commit_hash_df == commit].index[0]
                        
                        # Extra score of commit based on score mode
                        if score_mode == 'score':
                            score = score_df.loc[index]
                        elif score_mode == 'rank':
                            score = 1 / rank_df.loc[index]
                        else:
                            score = score_df.loc[index] / rank_df.loc[index]
                        
                        # Ensemble the scores based on ensemble method
                        if ensemble == 'mul':
                            vote_rows.append((commit, score * vote))
                        else:
                            vote_rows.append((commit, vote * ensemble[1] + score * (1 - ensemble[1])))

                    vote_df = pd.DataFrame(data=vote_rows, columns=["commit", "vote"])
                    vote_df.sort_values(by="vote", ascending=False, inplace=True)

                    result_dict[result_key] = vote_df
    
    return result_dict

# For a given project, generate dictionary with number of iterations of fonte for every settings
# Settings : ['HSFL', 'score_mode', 'ensemble', 'use_br', 'use_diff', 'stage2', 'use_stopword', 'adddel']
def bisection_all(pid, vid, tool='git'):
    fault_dir = os.path.join(CORE_DATA_DIR, f'{pid}-{vid}b')
    num_iters_dict = dict() # {setting : number of iterations}
    
    # Load fonte score dataframe
    fonte_scores_df = pd.read_hdf(os.path.join(DIFF_DATA_DIR, f'{pid}-{vid}b/fonte_scores.hdf'))
    
    # Get BIC
    GT = load_BIC_GT("./data/Defects4J/BIC_dataset")
    BIC = GT.set_index(["pid", "vid"]).loc[(pid, vid), "commit"]

    # Get list of every commits
    all_commits = get_all_commits(fault_dir)

    # Iterate through every settings
    for (index, row) in fonte_scores_df.iterrows():
        commit_df = row['commit']
        vote_df = row['vote']

        stage2 = index[5]
        if stage2 == 'skip':
            style_change_commits = []
        else:
            style_change_commits = get_style_change_commits(fault_dir, tool, with_Rewrite=(stage2=='True'))
        
        C_BIC = [c for c in all_commits if c in commit_df.values and c not in style_change_commits]
        scores = [vote_df.loc[commit_df.loc[commit_df == c].index[0]] for c in C_BIC]

        # Score of BIC is zero
        num_iters = (weighted_bisection(C_BIC, scores, BIC, ignore_zero=True), weighted_bisection(C_BIC, scores, BIC, ignore_zero=False))
        if num_iters[0] is None:
            with open('/root/workspace/eror.txt', 'a') as file:
                a = tuple(index)
                file.write(f'{pid}-{vid}b, True : {a}\n')
        
        if num_iters[1] is None:
            with open('/root/workspace/eror.txt', 'a') as file:
                a = tuple(index)
                file.write(f'{pid}-{vid}b, False : {a}\n')

        num_iters_dict[tuple(index)] = num_iters
    
    return num_iters_dict
