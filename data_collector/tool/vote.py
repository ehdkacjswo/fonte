import os, json, argparse, pickle, sys, subprocess, logging, itertools, time, re
import pandas as pd
import numpy as np
from scipy.spatial.distance import cosine
from collections import Counter

sys.path.append('/root/workspace/data_collector/lib/')
from encoder import Encoder
from vote_util import *
from utils import *
from BM25_Custom import *

DIR_NAME = '/home/coinse/doam/fonte/tmp'
DIFF_DATA_DIR = '/root/workspace/data/Defects4J/diff'
CORE_DATA_DIR = '/root/workspace/data/Defects4J/core'
BASE_DATA_DIR = '/root/workspace/data/Defects4J/baseline'
RESULT_DATA_DIR = '/root/workspace/data/Defects4J/result'
BASELINE_DATA_DIR = "/root/workspace/data/Defects4J/baseline"

# Load data
def load_data(pid, vid):
    #log('vote', '[INFO] Start loading data')
    #start_time = time.time()
    
    # Get token interval of given test method
    # Defects4J test
    def get_method_intvl(json_data, test_method):

        # Method declaration
        # (https://help.eclipse.org/latest/index.jsp?topic=%2Forg.eclipse.jdt.doc.isv%2Freference%2Fapi%2Forg%2Feclipse%2Fjdt%2Fcore%2Fdom%2FASTNode.html)
        if json_data['type'] == 'MethodDeclaration':
            for child in json_data['children']:
                if child['type'] == 'SimpleName': # Method name (May not be SimpleName in general use)
                    if child['label'] == test_method:
                        return [(int(json_data['pos']), int(json_data['pos']) + int(json_data['length']))]
                    break

            return []
        
        else:
            res = []
            for child in json_data['children']: # Gather result from children
                res += get_method_intvl(child, test_method)
            return res

    # When given test method is not present, it may be in extended class
    # (This isn't perfect solution, but works for given projects properly)
    def get_test_data(json_data, test_name, test_method):
        extension_list, method_intvl_list = [], []

        # Type (Class) declaration
        # (https://help.eclipse.org/latest/index.jsp?topic=%2Forg.eclipse.jdt.doc.isv%2Freference%2Fapi%2Forg%2Feclipse%2Fjdt%2Fcore%2Fdom%2FASTNode.html)
        if json_data['type'] == 'TypeDeclaration':
            for child in json_data['children']:

                # Class name (May not be SimpleName)
                if child['type'] == 'SimpleName':

                    # Consider target test class only (May not has label)
                    if child['label'] == test_name:
                        for _child in json_data['children']:
                            method_intvl_list += get_method_intvl(_child, test_method)

                    else:
                        break

                # Extended class (May not be SimpleType or extended class)
                if child['type'] == 'SimpleType':
                    extension_list.append(child['children'][0]['label']) # Might not work
            
            return extension_list, method_intvl_list
        
        else:
            for child in json_data['children']: # Gather result from children
                extension, method_intvl = get_test_data(child, test_name, test_method)
                extension_list += extension
                method_intvl_list += method_intvl
            
            return extension_list, method_intvl_list

    # Get code text of given failing test ()
    # Multiple tests could be essentially same test (Cli-27b)
    def get_test_code(test_path, test_name, test_method):
        
        # Test code is already found
        if (test_path, test_name, test_method) in vst_test:
            return (test_path, test_name, test_method), None

        # Copy test file
        test_src_path = os.path.join(test_path, test_name + '.java')

        try:
            with open(test_src_path, 'r', encoding="utf-8", errors="ignore") as file:
                code_txt = file.read()
        except:
            log('vote', f'[ERROR] Failed to read test file {test_src_path}')
            return None, None
        
        with open('/root/workspace/tmp/tmp.java', 'w') as file:
            file.write(code_txt)

        # Parse test file
        parse_cmd = f
        p = subprocess.Popen(['docker', 'run', '--rm', '-v', f'{DIR_NAME}:/diff', 'gumtree', 'parse', \
            '-g', 'java-jdt', '-f', 'JSON', 'tmp.java'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out_txt, err_txt = p.communicate()

        if p.returncode != 0: # Parsing error
            log('vote', f'[ERROR] Failed to parse test file {test_src_path}', out_txt, err_txt)
            return None, None

        try:
            parse_json = json.loads(out_txt.decode(encoding='utf-8', errors='ignore'))
        except:
            log('vote', f'[ERROR] Failed to decode parsed test file {test_src_path}')
            return None, None
        
        # Get extension & test method interval (if possible)
        extension_list, method_intvl_list = get_test_data(parse_json['root'], test_name, test_method)

        # Test not found, maybe on extension
        if len(method_intvl_list) == 0:
            ret_test_info, ret_test_code = None, None

            for extension in extension_list:
                test_info, test_code = get_test_code(test_path, extension, test_method)
                
                # Test not found
                if test_info is None:
                    continue
                
                # Test found
                elif ret_test_info is None or ret_test_code is None:
                    ret_test_info, ret_test_code = test_info, test_code
                
                # Multiple test found
                else:
                    log('vote', f'[ERROR] Multiple test method found {test_src_path}::{test_method}')
                    return None, None
            
            return ret_test_info, ret_test_code
        
        # Multiple test found
        elif len(method_intvl_list) > 1:
            log('vote', f'[ERROR] Multiple test method found {test_src_path}::{test_method}')
            return None, None
        
        # One test found
        else:
            return (test_path, test_name, test_method), code_txt[method_intvl_list[0][0] : method_intvl_list[0][1]]

    # Checkout Defects4J project
    p = subprocess.Popen(['sh', '/root/workspace/lib/checkout.sh', pid, vid], \
        stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out_txt, err_txt = p.communicate()

    if p.returncode != 0:
        log('vote', f'[ERROR] Checkout failed', out_txt, err_txt)
        return None, None, None
    
    # Change working directory to target Defects4J project
    try:
        os.chdir(f'/tmp/{pid}-{vid}b/')
    except:
        log('vote', '[ERROR] Moving directory failed')
        return None, None, None

    # Get path of test directory
    p = subprocess.Popen(['defects4j', 'export', '-p', 'dir.src.tests'], \
        stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    main_test_path, err_txt = p.communicate()

    if p.returncode != 0:
        log('vote', '[ERROR] Exporting test directory failed', main_test_path, err_txt)
        return None, None, None

    main_test_path = main_test_path.decode(encoding='utf-8', errors='ignore')

    # Load feature & encoder
    with open(os.path.join(DIFF_DATA_DIR, f'{pid}-{vid}b', 'feature.pkl'), 'rb') as file:
        feature_dict = pickle.load(file)
    
    with open(os.path.join(DIFF_DATA_DIR, f'{pid}-{vid}b', 'encoder.pkl'), 'rb') as file:
        encoder_dict = pickle.load(file)
    
    # Bug feature
    bug_feature_dict = dict()

    with open(os.path.join(BASELINE_DATA_DIR, f'{pid}-{vid}b', "br_long.txt"), "r") as f:
        bug_feature_dict['br_long'] = [f.read().strip()] # 1st bug report feature
    with open(os.path.join(BASELINE_DATA_DIR, f'{pid}-{vid}b', "br_short.txt"), "r") as f:
        bug_feature_dict['br_short'] = [f.read().strip()] # 2nd bug report feature
    
    # Handle failing test info
    with open(os.path.join(CORE_DATA_DIR, f'{pid}-{vid}b', "failing_tests"), "r", encoding="utf-8", errors="ignore") as f:
        fail_test = f.read().strip()
    
    for feature_type in ['test_name', 'exception', 'stack_trace', 'test_code']:
        bug_feature_dict[feature_type] = list()
    
    # Each block containing info of one failing test
    blocks = re.split(r'(?=^--- \S+::\S+)', fail_test, flags=re.MULTILINE)
    vst_test = set() # {(Test path, test name, test method)}

    for block in blocks:
        lines = block.strip()
        if len(lines) == 0:
            continue
        lines = lines.split('\n')

        # Failing test path, name and method name
        [test_path, test_method] = lines[0][4:].split('::')
        [test_path, test_name] = test_path.rsplit('.', 1)
        test_path = os.path.join(main_test_path, test_path.replace('.', '/'))
        bug_feature_dict['test_name'].append(os.path.join(test_path, test_name, test_method))

        test_info, test_code = get_test_code(test_path, test_name, test_method)

        # Test code already found
        if test_info in vst_test:
            continue
        
        # Test code not found
        elif test_code is None:
            return None, None, None

        vst_test.add(test_info)
        bug_feature_dict['test_code'].append(test_code)

        # Find starting point of stack trace
        trace_start = next((i for i, line in enumerate(lines) if re.match(r'^\tat \S+\(\S+\)', line)), len(lines))
        bug_feature_dict['exception'] += lines[1 : trace_start]
        bug_feature_dict['stack_trace'] += lines[trace_start : ]
    
    #end_time = time.time()
    #log('vote', f'[INFO] Elapsed time : {time_to_str(start_time, end_time)}')
    return feature_dict, encoder_dict, bug_feature_dict

# Bug2Commit with diff info
# 특정 버전에 대하여 vote for commits와 동일한 방법으로 evolve relationship 구축하고
# 해당 
def vote_bug2commit(total_feature_dict, total_encoder_dict, bug_feature_dict):
    log('vote', '[INFO] Bug2Commit voting')
    start_time = time.time()
    res_dict = dict()

    for stage2, setting_dict in total_feature_dict.items():
        res_dict[stage2] = dict()

        for setting, commit_dict in setting_dict.items():
            
            # Encode bug features
            encoder_setting = dict(setting)
            del encoder_setting['adddel']
            
            if encoder_setting['diff_tool'] is None: # dfa
                encoder = total_encoder_dict[stage2][frozenset({'tracker' : encoder_setting['tracker'], 'diff_tool' : 'base', 'diff_type' : 'base'}.items())]
            else:
                encoder = total_encoder_dict[stage2][frozenset(encoder_setting.items())]
            enc_bug_feature_dict = dict()

            for bug_type, bug_feature in bug_feature_dict.items():
                id_vec, non_id_vec = encoder.encode(bug_feature, update_vocab=False, \
                    mode='code' if bug_type == 'test_code' else 'text')
                enc_bug_feature_dict[bug_type] = {'id' : id_vec, 'non_id' : non_id_vec}

            # Get list of commit feature types
            commit_type_set = set()

            for feature_dict in commit_dict.values():
                commit_type_set |= set(feature_dict.keys())
            
            # For settings using identifires, add extrac setting that don't use full identifiers
            new_setting_list = [frozenset((dict(setting) | {'use_br' : True}).items()), frozenset((dict(setting) | {'use_br' : False}).items())]

            if 'diff_type' in dict(setting) and dict(setting)['diff_type'].endswith('id'):
                new_setting_list.append(frozenset((dict(setting) | {'diff_type' : 'no_' + dict(setting)['diff_type'], 'use_br' : True}).items()))
                new_setting_list.append(frozenset((dict(setting) | {'diff_type' : 'no_' + dict(setting)['diff_type'], 'use_br' : False}).items()))
                
            for new_setting in new_setting_list:
                res_dict[stage2][new_setting] = {'all' : list()}
                score_dict = res_dict[stage2][new_setting]

                use_id = 'diff_type' in dict(new_setting) and \
                    dict(new_setting)['diff_type'].endswith('id') and \
                    not dict(new_setting)['diff_type'].startswith('no_')
                use_br = dict(new_setting)['use_br']

                for commit_type in commit_type_set:
                    # Build BM25 vocabulary
                    bm25 = BM25_Encode()
                    for commit, feature_dict in commit_dict.items():
                        sub_feature_dict = feature_dict.get(commit_type, {'id' : Counter(), 'non_id' : Counter()}) # How should I handle empty type
                        #print(commit_type, sub_feature_dict)
                        if use_id:
                            bm25.add_document(sub_feature_dict['id'] + sub_feature_dict['non_id']) 
                        else:
                            bm25.add_document(sub_feature_dict['non_id'])
                    bm25.init_end()

                    # Vectorize query, commit features & evaluate similarity
                    for bug_type, bug_feature in enc_bug_feature_dict.items():
                        if not use_br and bug_type.startswith('br'):
                            continue

                        type_setting = frozenset({'commit' : commit_type, 'bug' : bug_type}.items())
                        score_dict[type_setting] = list()

                        bug_vector = bm25.vectorize_complex(\
                            bug_feature['id'] + bug_feature['non_id'] if use_id else bug_feature['non_id'])
                        
                        for commit, feature_dict in commit_dict.items():
                            if np.all(bug_vector == 0):
                                score_dict['all'].append([commit, 0])
                                score_dict[type_setting].append([commit, 0])
                                continue

                            if use_id:
                                commit_vector = bm25.vectorize_complex(sub_feature_dict['id'] + sub_feature_dict['non_id']) 
                            else:
                                commit_vector = bm25.vectorize_complex(sub_feature_dict['non_id']) 

                            if np.all(commit_vector == 0):
                                score_dict['all'].append([commit, 0])
                                score_dict[type_setting].append([commit, 0])
                            
                            else:
                                score_dict['all'].append([commit, 1 - cosine(bug_vector, commit_vector)])
                                score_dict[type_setting].append([commit, 1 - cosine(bug_vector, commit_vector)])
                
                # Format the result as DataFrame
                for type_setting, score_rows in score_dict.items():
                    vote_df = pd.DataFrame(data=score_rows, columns=["commit", "vote"])
                    vote_df = vote_df.groupby("commit").sum()
                    vote_df["rank"] = vote_df["vote"].rank(ascending=False, method="max")
                    vote_df["rank"] = vote_df["rank"].astype(int)
                    vote_df.sort_values(by="rank", inplace=True)
                    score_dict[type_setting] = vote_df

                    #print(res_dict[stage2][new_setting][type_setting])

    end_time = time.time()
    log('vote', f'[INFO] Elapsed time : {time_to_str(start_time, end_time)}')
    return res_dict

# For a given project, generate dataframe with result scores of fonte for every settings
# Settings : ['HSFL', 'score_mode', 'ensemble', 'use_br', 'use_diff', 'stage2', 'use_stopword', 'adddel']
def vote_fonte(pid, vid):
    #log('vote', '[INFO] Fonte voting')
    #start_time = time.time()
    fault_dir = os.path.join(CORE_DATA_DIR, f'{pid}-{vid}b')
    res_dict = dict()

    for stage2 in ['skip', 'precise']:
        excluded = get_style_change_commits(fault_dir, tool='git', stage2=stage2)
            
        fonte_df = vote_for_commits(fault_dir, tool='git', formula='Ochiai', decay=0.1, \
            voting_func=(lambda r: 1/r.max_rank), excluded=excluded, adjust_depth=True)
        
        # Add ranking
        fonte_df["rank"] = fonte_df["vote"].rank(ascending=False, method="max")
        fonte_df["rank"] = fonte_df["rank"].astype(int)

        res_dict[stage2] = fonte_df
    
    #end_time = time.time()
    #log('vote', f'[INFO] Elapsed time : {time_to_str(start_time, end_time)}')
    return res_dict

# Ensemble results from Bug2Commit & Fonte
def vote_ensemble(bug2commit_dict, fonte_dict):
    #log('vote', '[INFO] Vote ensembling')
    #start_time = time.time()
    res_dict = dict()
    
    for stage2, sub_dict in bug2commit_dict.items():
        fonte_df = fonte_dict[stage2]
        res_dict[stage2] = dict()

        for setting, bug2commit_df in sub_dict.items():
            merged_df = fonte_df.merge(bug2commit_df['all'], on='commit', how='left', suffixes=('_fonte', '_bug2commit'))
            merged_df['vote_bug2commit'].fillna(0, inplace=True) # 

            for beta in [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]:
                merged_df['vote'] = merged_df['vote_fonte'] * (1 + beta * merged_df['vote_bug2commit'])
                result_df = merged_df[['vote']].copy()

                result_df["rank"] = result_df["vote"].rank(ascending=False, method="max")
                result_df["rank"] = result_df["rank"].astype(int)
                result_df = result_df.sort_values(by="rank")

                # Update setting with beta
                new_setting = frozenset((dict(setting) | {'beta' : beta}).items())
                res_dict[stage2][new_setting] = result_df
    
    #end_time = time.time()
    #log('vote', f'[INFO] Elapsed time : {time_to_str(start_time, end_time)}')
    return res_dict

def main(pid, vid):
    log('vote', f'Working on {pid}_{vid}b')

    savedir = os.path.join(RESULT_DATA_DIR, f'{pid}-{vid}b', 'vote')
    os.makedirs(savedir, exist_ok=True)

    """# Load the previous result if possible
    bug2commit_save_path = os.path.join(savedir, f'bug2commit.pkl')

    if os.path.isfile(bug2commit_save_path):
        with open(bug2commit_save_path, 'rb') as file:
            res_dict = pickle.load(file)
    
    else:
        res_dict = dict()"""

    # Load feature & vocab for the project
    start_time = time.time()
    feature_dict, encoder_dict, bug_feature_dict = load_data(pid, vid)
    #print('bug_feature_dict', json.dumps(bug_feature_dict, indent=4))

    if feature_dict is None or encoder_dict is None or bug_feature_dict is None:
        return

    # Bug2Commit voting
    bug2commit_vote = vote_bug2commit(feature_dict, encoder_dict, bug_feature_dict)
    
    """with open(os.path.join(savedir, f'bug2commit.pkl'), 'wb') as file:
        pickle.dump(bug2commit_vote, file)

    # Fonte voting
    fonte_vote = vote_fonte(pid, vid)
    with open(os.path.join(savedir, 'fonte.pkl'), 'wb') as file:
        pickle.dump(fonte_vote, file)

    # Ensemble voting
    with open(os.path.join(savedir, 'ensemble.pkl'), 'wb') as file:
        pickle.dump(vote_ensemble(bug2commit_vote, fonte_vote), file)"""