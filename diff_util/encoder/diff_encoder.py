import os, json, argparse, pickle, sys, itertools
import pandas as pd
from tqdm import tqdm

sys.path.append('/root/workspace/diff_util/lib/')
from diff import *
from encoder import *

DIFF_DATA_DIR = '/root/workspace/data/Defects4J/diff'
CORE_DATA_DIR = '/root/workspace/data/Defects4J/core'
BASE_DATA_DIR = '/root/workspace/data/Defects4J/baseline'

# Get the unchanged file data
# [(commit, src_path)]
def get_style_change_data(coredir, tool='git', with_Rewrite=True):
    postfix = "" if with_Rewrite else "_noOpenRewrite"
    val_df = pd.read_csv(
        os.path.join(coredir, tool, f"validation{postfix}.csv"), 
        header=None,
        names=["commit", "src_path", "AST_diff"])
    
    unchanged_df = val_df[val_df["AST_diff"] == "U"]
    return list(zip(unchanged_df["commit"], unchanged_df["src_path"]))

# Encode the raw diff data
# skip_stage_2 = Excluding style change diff, with_Rewrite = , use_stopword
# Encoded data : {commit_hash : [addition_list, deletion_list, msg_encode]}
# addition/deletion dict : [(before/after_src_path_encode, content_encode_sum)]
def encode_pid(pid, vid, tool='git', skip_stage_2=False, with_Rewrite=True, use_stopword=True):
    # Load related diff data
    diff_data_dir = os.path.join(DIFF_DATA_DIR, f'{pid}-{vid}b')
    with open(os.path.join(diff_data_dir, 'diff.pkl'), 'rb') as file:
        diff_data = pickle.load(file)
    
    # Get list of style change commits
    if skip_stage_2:
        excluded = []
    else:
        excluded = get_style_change_data(os.path.join(CORE_DATA_DIR, f'{pid}-{vid}b'), tool, with_Rewrite)

    # Merge the diff data
    diff_dict = dict()

    for commit_hash, commit_diff in diff_data.diff_dict.items(): # Iterate through commits
        addition_dict = dict()
        deletion_dict = dict()

        for src_path, src_diff in commit_diff.items(): # Iterate through source files editted by commit
            if (commit_hash, src_path) in excluded: # Exclude style change
                continue
            
            for (before_src_path, after_src_path), [addition, deletion] in src_diff.diff_dict.items():
                addition_content_dict = addition_dict.get(after_src_path, dict())
                deletion_content_dict = deletion_dict.get(before_src_path, dict())

                for line, content in addition.items():
                    if line not in addition_content_dict:
                        addition_content_dict[line] = content
                    
                    elif content != addition_content_dict[line]: # Different diff data for same source file
                        print(f'Different addition content!!! {commit_hash} {after_src_path} {line}')
                
                for line, content in deletion.items():
                    if line not in deletion_content_dict:
                        deletion_content_dict[line] = content
                    
                    elif content != deletion_content_dict[line]: # Different diff data for same source file
                        print(f'Different deletion content!!! {commit_hash} {before_src_path} {line}')
                
                addition_dict[after_src_path] = addition_content_dict
                deletion_dict[before_src_path] = deletion_content_dict
        
        diff_dict[commit_hash] = [addition_dict, deletion_dict]
    
    # Encode the merged data
    encode_dict = dict()
    encoder = Encoder()
    base_data_dir = os.path.join(BASE_DATA_DIR, f'{pid}-{vid}b/commits/')

    for commit_hash, [addition_dict, deletion_dict] in diff_dict.items():
        addition_list = []
        deletion_list = []

        # Encode addition data
        for after_src_path, addition_content_dict in addition_dict.items():
            after_src_path_encode = encoder.encode(after_src_path, use_stopword)
            addition_encode_sum = list()

            for content in addition_content_dict.values():
                addition_encode_sum = sum_encode(addition_encode_sum, encoder.encode(content, use_stopword))
            
            addition_list.append((after_src_path_encode, addition_encode_sum))
        
        # Encode deletion data
        for before_src_path, deletion_content_dict in deletion_dict.items():
            before_src_path_encode = encoder.encode(before_src_path, use_stopword)
            deletion_encode_sum = list()

            for content in deletion_content_dict.values():
                addition_encode_sum = sum_encode(deletion_encode_sum, encoder.encode(content, use_stopword))
            
            deletion_list.append((before_src_path_encode, deletion_encode_sum))
        
        # Encode message
        for filename in os.listdir(base_data_dir):
            if filename.startswith(f'c_{commit_hash}'):
                with open(os.path.join(base_data_dir, filename), "r") as file:
                    data = json.load(file)
                msg_encode = encoder.encode(data['log'], use_stopword)
                break
        
        encode_dict[commit_hash] = [addition_list, deletion_list, msg_encode]
    
    # Save encoded data and vocab
    diff_encode_dir = os.path.join(diff_data_dir, 'encode')
    os.makedirs(diff_encode_dir, exist_ok=True)

    save_postfix = savepath_postfix(tool, skip_stage_2, with_Rewrite, use_stopword)

    with open(os.path.join(diff_encode_dir, f'diff_encode{save_postfix}.pkl'), 'wb') as file:
        pickle.dump(encode_dict, file)
    with open(os.path.join(diff_encode_dir, f'vocab{save_postfix}.pkl'), 'wb') as file:
        pickle.dump(encoder.vocab, file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Encode diff data")
    parser.add_argument('--all', action="store_true",
        help="history retrieval tool, git or shovel (default: git)")
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

    skip_stage_2_list = [True, False]
    with_Rewrite_list = [True, False]
    use_stopword_list = [True, False]
    param_list = list(itertools.product(skip_stage_2_list, with_Rewrite_list, use_stopword_list))

    # Iterate through projects
    for project_dir in os.listdir(DIFF_DATA_DIR):
        print(f'Working on project {project_dir}')
        [pid, vid] = project_dir[:-1].split("-")
        
        for (skip_stage_2, with_Rewrite, use_stopword) in param_list:
            if skip_stage_2 and with_Rewrite:
                continue
            encode_pid(pid, vid, tool='git', skip_stage_2=skip_stage_2, with_Rewrite=with_Rewrite, use_stopword=use_stopword)