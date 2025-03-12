from scipy import sparse
import pandas as pd
from sbfl.base import SBFL
import os, pickle, math, re, subprocess, sys, shutil, json
from intervaltree import Interval, IntervalTree
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

sys.path.append('/root/workspace/lib/')
from experiment_utils import load_BIC_GT

DIFF_DATA_DIR = '/root/workspace/data/Defects4J/diff'
RESULT_DATA_DIR = '/root/workspace/data/Defects4J/result'
CORE_DATA_DIR = '/root/workspace/data/Defects4J/core'

#test_regex = re.compile()
# Directory mounted on "original" adfasdf
DIR_NAME = '/home/coinse/doam/fonte/tmp'

def get_method_intvl(json_data, test_method):
    
    # Method declaration
    if json_data['type'] == 'MethodDeclaration':
        for child in json_data['children']:
            
            # Method name
            if child['type'] == 'SimpleName':
                #print(child['label'], test_method)
                #print(len(child['label']), len(test_method))
                if child['label'] == test_method:
                    return [(int(json_data['pos']), int(json_data['pos']) + int(json_data['length']))]
                break

        return []
    
    # 
    else:
        res = []

        for child in json_data['children']:
            res += get_method_intvl(child, test_method)  

        return res

def get_extension(json_data, test_name):
    if json_data['type'] == 'TypeDeclaration':
        for child in json_data['children']:
            
            if child['type'] == 'SimpleName':
                if child['label'] != test_name:
                    break
            
            if child['type'] == 'SimpleType':
                #print(child)
                return child['children'][0]['label']
        
        return None
    
    else:
        for child in json_data['children']:
            res = get_extension(child, test_name)
            if res is not None:
                return res
        
        return None

def get_test_code(test_path, test_name, test_method):
    #print(test_path, test_name, test_method)

    try:
        with open(os.path.join(test_path, test_name + '.java'), 'r', encoding="utf-8", errors="ignore") as file:
            code_txt = file.read()
        with open('/root/workspace/tmp/tmp.java', 'w') as file:
            file.write(code_txt)
    except:
        print(f'No such file {test_path}/{test_name}.java')
        return None

    parse_cmd = f'docker run --rm -v {DIR_NAME}:/diff gumtree parse -g java-jdt -f JSON tmp.java'
    p = subprocess.Popen(parse_cmd, shell=True, stdout=subprocess.PIPE)
    stdout, _ = p.communicate()

    if p.returncode != 0: # Parsing error
        return None

    try:
        parse_json = json.loads(stdout.decode(encoding='utf-8', errors='ignore'))
    except:
        print('Decoding error')
        return None # Decoding error
    
    method_intvl = get_method_intvl(parse_json['root'], test_method)
    #print(json.dumps(parse_json, indent=4))

    # Test not found, maybe on other
    if len(method_intvl) == 0:
        extension = get_extension(parse_json['root'], test_name)
        if extension is not None:
            return get_test_code(test_path, extension, test_method)
    
    else:
        return method_intvl
    
    return None

    
    #with open(test_path, "r") as file:
    #    code_txt = file.read()
    #print(code_txt[res[0]:res[1]])
    #print(stdout)
    #id_intvl_dict = {'class':CustomInterval(), 'method':CustomInterval(), 'variable':CustomInterval(), 'comment':CustomInterval()}

if __name__ == "__main__":
    all_GT = load_BIC_GT("/root/workspace/data/Defects4J/BIC_dataset")
    GT = all_GT[all_GT['provenance'].str.contains("Manual", na=False)]
    #update = False

    for _, row in GT.iterrows():
        pid, vid = row.pid, row.vid
        #pid, vid = 'Compress', '7'
        print(f'Working on {pid}-{vid}b')

        # Checkout Defects4J project
        p = subprocess.Popen(['sh', '/root/workspace/lib/checkout.sh', pid, vid], \
            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out_txt, err_txt = p.communicate()

        if p.returncode != 0:
            print('Checkout failed')
            continue
        
        # Change working directory to target Defects4J project
        try:
            os.chdir(f'/tmp/{pid}-{vid}b/')
        except:
            print('Directory moving failed')
            continue

        # Get test directory
        p = subprocess.Popen(['defects4j', 'export', '-p', 'dir.src.tests'], \
            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        test_path, err_txt = p.communicate()

        if p.returncode != 0:
            print('Getting test path failed')
            continue

        main_test_path = test_path.decode(encoding='utf-8', errors='ignore')
        
        with open(os.path.join(CORE_DATA_DIR, f'{pid}-{vid}b', "failing_tests"), "r") as f:
            txt = f.read().strip()

        blocks = re.split(r'(?=^--- \S+::\S+)', txt, flags=re.MULTILINE)

        for block in blocks:
            lines = block.strip()
            if len(lines) == 0:
                continue

            lines = lines.split('\n')
            
            trace_start = next((i for i, line in enumerate(lines) if re.match(r'^\tat \S+\(\S+\)', line)), len(lines))

            #print('Failing test)', lines[0])
            [test_path, test_method] = lines[0][4:].split('::')
            [test_path, test_name] = test_path.rsplit('.', 1)
            test_path = os.path.join(main_test_path, test_path.replace('.', '/'))
            # test_path가 project마다 다를수도 있다
            test_method = lines[0][lines[0].find('::') + 2:]
            #print(test_name)

            res = get_test_code(test_path, test_name, test_method)

            if res is None:
                print(f'Failed for {pid}-{vid} {test_path}/{test_name}.java::{test_method}')
            elif len(res) == 0:
                print(f'Test not found for {pid}-{vid} {test_path}/{test_name}.java::{test_method}')
            elif len(res) > 1:
                print(f'Multple test found for {pid}-{vid} {test_path}/{test_name}.java::{test_method}')

            #print('Error message)', lines[1:trace_start])
            #print('Stack trace)', lines[trace_start:])
        #break
        