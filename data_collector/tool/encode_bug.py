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
def load_bug_feature(pid, vid):
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
            log('encode_bug', f'[ERROR] Failed to read test file {test_src_path}')
            return None, None
        
        with open('/root/workspace/tmp/tmp.java', 'w') as file:
            file.write(code_txt)

        # Parse test file
        parse_cmd = f
        p = subprocess.Popen(['docker', 'run', '--rm', '-v', f'{DIR_NAME}:/diff', 'gumtree', 'parse', \
            '-g', 'java-jdt', '-f', 'JSON', 'tmp.java'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out_txt, err_txt = p.communicate()

        if p.returncode != 0: # Parsing error
            log('encode_bug', f'[ERROR] Failed to parse test file {test_src_path}', out_txt, err_txt)
            return None, None

        try:
            parse_json = json.loads(out_txt.decode(encoding='utf-8', errors='ignore'))
        except:
            log('encode_bug', f'[ERROR] Failed to decode parsed test file {test_src_path}')
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
                    log('encode_bug', f'[ERROR] Multiple test method found {test_src_path}::{test_method}')
                    return None, None
            
            return ret_test_info, ret_test_code
        
        # Multiple test found
        elif len(method_intvl_list) > 1:
            log('encode_bug', f'[ERROR] Multiple test method found {test_src_path}::{test_method}')
            return None, None
        
        # One test found
        else:
            return (test_path, test_name, test_method), code_txt[method_intvl_list[0][0] : method_intvl_list[0][1]]
    
    log('encode_bug', '[INFO] Start loading data')
    start_time = time.time()

    # Get path of test directory
    p = subprocess.Popen(['defects4j', 'export', '-p', 'dir.src.tests'], \
        stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    main_test_path, err_txt = p.communicate()

    if p.returncode != 0:
        log('encode_bug', '[ERROR] Exporting test directory failed', main_test_path, err_txt)
        return None, None, None

    main_test_path = main_test_path.decode(encoding='utf-8', errors='ignore')
    
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
    
    end_time = time.time()
    log('encode_bug', f'[INFO] Elapsed time : {time_to_str(start_time, end_time)}')
    return bug_feature_dict

def main(pid, vid):
    log('encode_bug', f'Working on {pid}_{vid}b')

    diff_data_dir = os.path.join(DIFF_DATA_DIR, f'{pid}-{vid}b')
    os.makedirs(diff_data_dir, exist_ok=True)

    # Load encoder & bug feature
    with open(os.path.join(DIFF_DATA_DIR, f'{pid}-{vid}b', 'encoder.pkl'), 'rb') as file:
        encoder_dict = pickle.load(file)
    
    bug_feature = load_bug_feature(pid, vid)
    
    # Encode bug feature
    res_dict = dict()

    for stage2, setting_dict in encoder_dict.items():
        res_dict.setdefault(stage2, dict())

        for setting, encoder in setting_dict.items():
            res_dict[stage2].setdefault(setting, dict())

            for bug_type, feature in bug_feature.items():
                id_vec, non_id_vec = encoder.encode(feature, update_vocab=False, mode='code' if bug_type == 'test_code' else 'text')
                res_dict[stage2][setting][bug_type] = {'id' : id_vec, 'non_id' : non_id_vec}

    with open(os.path.join(diff_data_dir, 'bug_feature.pkl'), 'wb') as file:
        pickle.dump(res_dict, file)