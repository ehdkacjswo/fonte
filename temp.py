from scipy import sparse
import pandas as pd
from sbfl.base import SBFL
import os

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

if __name__ == "__main__":
    cov_df = get_sbfl_scores_from_coverage('/root/workspace/data/Defects4J/core/Closure-30b/coverage.pkl')
    print(cov_df)