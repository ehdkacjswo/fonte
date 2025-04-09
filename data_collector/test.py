import pickle
import os, sys
import pandas as pd

DIFF_DATA_DIR = '/root/workspace/data/Defects4J/diff'
CORE_DATA_DIR = '/root/workspace/data/Defects4J/core/'

# Get the unchanged file data
# [(commit, src_path)]
"""def get_style_change_data(coredir, tool='git', with_Rewrite=True):
    postfix = "" if with_Rewrite else "_noOpenRewrite"
    val_df = pd.read_csv(
        os.path.join(coredir, tool, f"validation{postfix}.csv"), 
        header=None,
        names=["commit", "src_path", "AST_diff"])
    
    unchanged_df = val_df[val_df["AST_diff"] == "U"]
    return list(zip(unchanged_df["commit"], unchanged_df["src_path"]))


def get_range_dict(pid='Closure', vid='33', tool='git'):
    commit_path = os.path.join(CORE_DATA_DIR, f'{pid}-{vid}b', tool, 'commits.pkl')
    com_df = pd.read_pickle(commit_path)
    com_df.drop_duplicates(subset=['src_path', 'begin_line', 'end_line'], inplace=True)
    range_dict = dict()

    for _, row in com_df.iterrows():
        src_path = row['src_path']
        range_val = range_dict.get(src_path, [])
        range_val.append((row['begin_line'], row['end_line']))
        range_dict[src_path] = range_val

    return range_dict

def find_rename():
    for project_dir in os.listdir(DIFF_DATA_DIR):
        [pid, vid] = project_dir[:-1].split("-")
        excluded = get_style_change_data(os.path.join(CORE_DATA_DIR, f'{pid}-{vid}b/'), 'git', True)

        diff_data_dir = os.path.join(DIFF_DATA_DIR, f'{pid}-{vid}b')
        with open(os.path.join(diff_data_dir, 'diff.pkl'), 'rb') as file:
            diff_data = pickle.load(file)
        
        for commit_hash, commit_diff in diff_data.diff_dict.items(): # Iterate through commits

            for src_path, src_diff in commit_diff.items(): # Iterate through source files editted by commit
                for (before_src_path, after_src_path), [addition, deletion] in src_diff.diff_dict.items():
                    if before_src_path != after_src_path:
                        print(project_dir, commit_hash, before_src_path, after_src_path)
                    if (commit_hash, after_src_path) in excluded: # Exclude style change
                        continue"""

if __name__ == "__main__":
    find_rename()