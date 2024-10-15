import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'lib')))

from experiment_utils import *

def get_coverage(fauld_dir, covered_by_failure_only=True,
    in_class_only=False, use_method_level_score=False):

    path_to_coverage = os.path.join(fault_dir, "coverage.pkl")
    postfix = "suspicious" if covered_by_failure_only else "all"
    postfix += "_inClassOnly" if in_class_only else ""

    # Load coverage matrix
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

    # Consider only statements covered by failing tests
    if covered_by_failure_only:
        cov_df = cov_df.loc[:, cov_df.loc[~is_passing].any(axis=0)]

    # For method level score, each statements votes for the method
    # And score of method will be sum of the votes from the statements
    """if use_method_level_score:
        identifier = ["class_file", "method_name", "method_signature","begin_line", "end_line"]
        l_cov_df = sbfl_df.reset_index()
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
        sbfl_df = sbfl_df.set_index(identifier)"""

    return cov_df

def commit_df_to_evolve (pid, vid, tool, with_Rewrite=True, skip_stage_2=False):
    
    # Get excluded commits
    fault_dir = os.path.join('data/Defects4J/core/{}-{}b/'.format(pid, vid))

    if skip_stage_2:
        excluded = []
    else:
        excluded = get_style_change_commits(fault_dir, tool)
    
    # Get evolve data
    cov_df = get_coverage()
    
    # Get commit history info
    commit_df = load_commit_history(fault_dir, tool)

    commit_df["excluded"] = commit_df["commit_hash"].isin(excluded)
    commit_df["new_depth"] = commit_df["depth"]

    for _, row in cov_df.reset_index().iterrows():
        print('Target file : {}, line : {}'.format(row.class_file, row.line))
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

        # Find commits that modified the statement(method) and not excluded
        for commit in zip(com_df.commit_hash):
            if commit not in excluded:
                print(commit)
        
        induce_sqrt = math.sqrt(len(list_commits))
        #sum_score = 1 if score is None else sum_score / induce_sqrt
        sum_score = 1
        
if __name__ == "__main__":
    commit_df_to_evolve('Math', 63, 'git')
