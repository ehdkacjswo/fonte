import os, sys
import pandas as pd

sys.path.append('/root/workspace/lib/')
from experiment_utils import get_sbfl_scores_from_coverage

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

# 
def get_style_change_commits(fault_dir, tool='git', stage2='precise'):
    if stage2 == 'skip':
        return set()

    if stage2 == 'precise':
        val_df = pd.read_csv(
            os.path.join(fault_dir, tool, f"precise_validation_noOpenRewrite.csv"), 
            header=None,
            names=["commit", "before_src_path", "after_src_path", "AST_diff"])
        
        style_df = val_df[val_df["AST_diff"] == "U"]
        return set(zip(style_df["commit"], style_df["before_src_path"], style_df["after_src_path"]))

# Deleted method level
def vote_for_commits(fault_dir, tool, formula, decay, voting_func, \
    excluded=[], adjust_depth=True, in_class_only=False):
    commit_df = load_commit_history(fault_dir, tool)
    commit_df["new_depth"] = commit_df["depth"]

    # Style change exists > Update commit depth
    if len(excluded) > 0:
        # Excluded commits have "None" 
        mask = commit_df.apply(lambda row: (row["commit_hash"], row["before_src_path"], row["after_src_path"]) in excluded, axis=1)
        commit_df.loc[mask, "new_depth"] = None

        commit_df["method_identifier"] = commit_df.class_file + ":" + \
            commit_df.method_name + commit_df.method_signature + \
            ":L" + commit_df.begin_line.astype(str) + "," + commit_df.end_line.astype(str)

        # Adjust depth for each suspicious methods
        for (src_path, method_id), group in commit_df.groupby(["src_path", "method_identifier"]):
            affected_depths = sorted(group.loc[group["new_depth"].isna(), "depth"].tolist())
            
            if not affected_depths:
                continue
                
            for idx, row in group.iterrows():
                if pd.notna(row["new_depth"]):
                    num_smaller = sum(d < row["depth"] for d in affected_depths)
                    commit_df.at[idx, "new_depth"] -= num_smaller

    sbfl_df = get_sbfl_scores_from_coverage(
        os.path.join(fault_dir, "coverage.pkl"),
        formula=formula,
        covered_by_failure_only=True,
        in_class_only=in_class_only)

    #sbfl_df["dense_rank"] = (-sbfl_df["score"]).rank(method="dense")
    sbfl_df["max_rank"] = (-sbfl_df["score"]).rank(method="max")
    vote_rows = []

    for _, row in sbfl_df.reset_index().iterrows():
        vote = voting_func(row)
        
        com_df = commit_df[
            (commit_df.class_file == row.class_file) \
            & (commit_df.begin_line <= row.line) \
            & (commit_df.end_line >= row.line)
        ]

        for commit, depth in zip(com_df.commit_hash, com_df.new_depth):
            if depth is None:
                decayed_vote = 0
            else:
                decayed_vote = vote * ((1-decay) ** depth)
            vote_rows.append([commit, decayed_vote])
            
    vote_df = pd.DataFrame(data=vote_rows, columns=["commit", "vote"])
    agg_vote_df = vote_df.groupby("commit").sum("vote")
    agg_vote_df.sort_values(by="vote", ascending=False, inplace=True)
    return agg_vote_df