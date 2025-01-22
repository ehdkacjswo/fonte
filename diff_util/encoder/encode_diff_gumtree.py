import os, json, argparse, pickle, sys, itertools, subprocess
import pandas as pd
from tqdm import tqdm
from interval import interval

sys.path.append('/root/workspace/diff_util/lib/')
from diff import *
from encoder import *
from gumtree import *

DIFF_DATA_DIR = '/root/workspace/data/Defects4J/diff'
CORE_DATA_DIR = '/root/workspace/data/Defects4J/core'
BASE_DATA_DIR = '/root/workspace/data/Defects4J/baseline'

# Get the unchanged file data
# [(commit, src_path)]
def get_style_change_data(coredir, tool='git', with_Rewrite=True):
    #postfix = "" if with_Rewrite else "_noOpenRewrite"
    postfix = "" if with_Rewrite else ""
    prefix = "" if with_Rewrite else "precise_"
    val_df = pd.read_csv(
        os.path.join(coredir, tool, f"{prefix}validation{postfix}.csv"), 
        header=None,
        names=["commit", "src_path", "AST_diff"])
    
    unchanged_df = val_df[val_df["AST_diff"] == "U"]
    return list(zip(unchanged_df["commit"], unchanged_df["src_path"]))

# Encode the raw diff data
# skip_stage_2 = Excluding style change diff, with_Rewrite = , use_stopword
# Encoded data : {commit_hash : [addition_list, deletion_list, msg_encode]}
# addition/deletion dict : [(before/after_src_path_encode, content_encode_sum)]
def encode_git(pid, vid, stage2, use_stopword):
    # Load related diff data
    diff_data_dir = os.path.join(DIFF_DATA_DIR, f'{pid}-{vid}b')
    with open(os.path.join(diff_data_dir, 'git_diff.pkl'), 'rb') as file:
        git_diff = pickle.load(file)
    
    # Get list of style change commits
    if stage2 == 'skip':
        excluded = []
    else:
        excluded = get_style_change_data(os.path.join(CORE_DATA_DIR, f'{pid}-{vid}b/'), 'git', stage2)

    # Merge the diff data
    diff_dict = dict()

    for commit_hash, commit_diff in git_diff.items(): # Iterate through commits
        addition_dict = dict()
        deletion_dict = dict()
        add_data = False # Add the data only when the commit has modified relative files

        for (before_src_path, after_src_path), git_diff_dict in commit_diff.items(): # Iterate through source files editted by commit
            if (commit_hash, after_src_path) in excluded: # Exclude style change
                continue
            
            add_data = True

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

def encode_gumtree(pid, vid, stage2, use_stopword):
    
    # Helper function encoding tokens
    def encode_token(src_interval_dict, encoder, use_stopword):
        src_encode_dict = dict()

        for src_path, src_interval in src_interval_dict.items():
            encode_dict = {'src_path' : encoder.encode(src_path, use_stopword=use_stopword, update_vocab=True)}

            # Empty interval
            if len(src_interval) == 0:
                src_encode_dict[src_path] = encode_dict
                continue

            p = subprocess.Popen(f'git show {commit_hash}:{src_path}', shell=True, stdout=subprocess.PIPE)
            code_txt, _ = p.communicate()

            # Error raised while copying file
            if p.returncode != 0:
                # Actual addition occured but failed to copy file
                with open('/root/workspace/error.txt', 'a') as file:
                    file.write(f'Failed to copy file {commit_hash}:{src_path}')
                src_encode_dict[src_path] = encode_dict
                continue
            
            else:
                code_txt = code_txt.decode(encoding='utf-8', errors='ignore')
                with open('/root/workspace/tmp/tmp.java', 'w') as file:
                    file.write(code_txt)
            
            token_dict = gumtree_parse('tmp.java', src_interval)
            
            # Error while parsing
            if token_dict is None:
                src_encode_dict[src_path] = encode_dict
                continue

            for token_type, token_list in token_dict.items():
                encode_dict[token_type] = []

                for token in token_list:
                    encode_dict[token_type] = sum_encode(encode_dict[token_type], \
                        encoder.encode(token, use_stopword=use_stopword, update_vocab=True))
            
            src_encode_dict[src_path] = encode_dict
        
        return src_encode_dict

    # Load related diff data
    diff_data_dir = os.path.join(DIFF_DATA_DIR, f'{pid}-{vid}b')
    
    with open(os.path.join(diff_data_dir, 'gumtree_diff_interval.pkl'), 'rb') as file:
        gumtree_diff = pickle.load(file)
    
    # Get list of style change commits
    if stage2 == 'skip':
        excluded = []
    else:
        excluded = get_style_change_data(os.path.join(CORE_DATA_DIR, f'{pid}-{vid}b/'), 'git', stage2)

    # {commit_hash : {addition / deletion : {src_path : interval}}}
    total_interval = dict()
    
    for commit_hash, commit_diff in gumtree_diff.items(): # Iterate through commits
        addition_interval_dict = dict()
        deletion_interval_dict = dict()

        for (before_src_path, after_src_path), diff_interval in commit_diff.items(): # Iterate through source files editted by commit
            if (commit_hash, after_src_path) in excluded: # Exclude style change
                continue

            if after_src_path != '/dev/null' and len(diff_interval['addition']) > 0:
                addition_interval_dict[after_src_path] = \
                    addition_interval_dict.get(after_src_path, interval()) | diff_interval['addition']
            
            if before_src_path != '/dev/null' and len(diff_interval['deletion']):
                deletion_interval_dict[before_src_path] = \
                    deletion_interval_dict.get(before_src_path, interval()) | diff_interval['deletion']

        
        total_interval[commit_hash] = {'addition' : dict(), 'deletion' : dict()}

        for src_path, src_interval in addition_interval_dict.items():
            total_interval[commit_hash]['addition'][src_path] = src_interval
        
        for src_path, src_interval in deletion_interval_dict.items():
            total_interval[commit_hash]['deletion'][src_path] = src_interval
    
    # {commit_hash : {addition / deletion : {src_path : {src_path, token_types}}}}
    total_encode = dict()
    encoder = Encoder()

    for commit_hash, commit_interval in total_interval.items():
        total_encode[commit_hash] = dict()
        total_encode[commit_hash]['addition'] = encode_token(commit_interval['addition'])
        total_encode[commit_hash]['deletion'] = encode_token(commit_interval['deletion'])
        


            




    print(before_src_path, after_src_path)
    print(gumtree_diff_interval['addition'].format("%+g"))
    print(gumtree_diff_interval['deletion'].format("%+g"))
    
    return 1, 2

if __name__ == "__main__":
    stage2_list = [True, False] # ['skip', True, False] Skip stage or use OpenRewrite or not
    use_stopword_list = [True] # [True, False] Use stopword or not    
    param_list = list(itertools.product(stage2_list, use_stopword_list))
    
    for project_dir in tqdm(os.listdir(DIFF_DATA_DIR)):
        print(f'Working on project {project_dir}')
        [pid, vid] = project_dir[:-1].split("-")
        
        # Encode diff for every settings
        encode_dict = dict()
        vocab_dict = dict()

        for (stage2, use_stopword) in param_list:
            encode_res, vocab = encode_pid(pid=pid, vid=vid, stage2=stage2, use_stopword=use_stopword)
            encode_dict[(stage2, use_stopword)] = encode_res
            vocab_dict[(stage2, use_stopword)] = vocab

        diff_encode_dir = os.path.join(DIFF_DATA_DIR, f'{pid}-{vid}b/encode')
        os.makedirs(diff_encode_dir, exist_ok=True)

        """with open(os.path.join(diff_encode_dir, f'diff_encode.pkl'), 'wb') as file:
            pickle.dump(encode_dict, file)
        with open(os.path.join(diff_encode_dir, f'vocab.pkl'), 'wb') as file:
            pickle.dump(vocab_dict, file)"""