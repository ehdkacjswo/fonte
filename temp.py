from scipy import sparse
import pandas as pd
from sbfl.base import SBFL
import os, pickle, math, re, subprocess, sys, shutil, json
from intervaltree import Interval, IntervalTree
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

sys.path.append('/root/workspace/docker/workspace/collect_utils/')
from javaparser import NodeRanges

DIFF_DATA_DIR = '/root/workspace/data/Defects4J/diff'
RESULT_DATA_DIR = '/root/workspace/data/Defects4J/result'
CORE_DATA_DIR = '/root/workspace/data/Defects4J/core'

#test_regex = re.compile()
# Directory mounted on "original" adfasdf
DIR_NAME = '/home/coinse/doam/fonte/tmp'

def get_test_code(test_path, test_name):

    try:
        shutil.copy(test_path, '/root/workspace/tmp/tmp.java')
    
    # For other errors
    except:
        print("Error occurred while copying file.")
        return None

    parse_cmd = f'docker run --rm -v {DIR_NAME}:/diff gumtree parse -g java-jdt -f JSON tmp.java'
    p = subprocess.Popen(parse_cmd, shell=True, stdout=subprocess.PIPE)
    stdout, _ = p.communicate()

    if p.returncode != 0: # Parsing error
        return None

    try:
        parse_json = json.loads(stdout.decode(encoding='utf-8', errors='ignore'))
    except:
        return None # Decoding error
    print(json.dumps(parse_json, indent=4))
    #id_intvl_dict = {'class':CustomInterval(), 'method':CustomInterval(), 'variable':CustomInterval(), 'comment':CustomInterval()}
    
    """# Return type + Test name + Parameter + (Extend, Throw, etc...) + '{'
    #test_pattern = rf'(\b\w+\s+){test_name}\s*\([^)]*\)\s*{{'
    test_pattern = rf'\bpublic\s+void\s+{test_name}\s*\([^)]*\)\s*throws\s+Exception\s*{{'

    # Search for the method start
    match = re.search(test_pattern, code_txt)
    if not match:
        return None  # Method not found

    start_index = match.start()  # Start of method
    brace_count = 0
    end_index = start_index

    # Find the method end by tracking braces
    for i in range(start_index, len(code_txt)):
        if code_txt[i] == '{':
            brace_count += 1
        elif code_txt[i] == '}':
            brace_count -= 1
            if brace_count == 0:
                end_index = i + 1  # Include closing brace
                break

    return code_txt[start_index:end_index].strip()"""

if __name__ == "__main__":
    pid, vid = 'Closure', '62'

    # Checkout Defects4J project
    """p = subprocess.Popen(['source', '$HOME/.sdkman/bin/sdkman-init.sh', '&&', 'sdk', 'use', 'java', '11.0.12-open'], \
        stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out_txt, err_txt = p.communicate()

    if p.returncode != 0:
        sys.exit(0)"""

    # Checkout Defects4J project
    p = subprocess.Popen(['sh', '/root/workspace/lib/checkout.sh', pid, vid], \
        stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out_txt, err_txt = p.communicate()

    if p.returncode != 0:
        sys.exit(0)
    
    # Change working directory to target Defects4J project
    try:
        os.chdir(f'/tmp/{pid}-{vid}b/')
    except:
        sys.exit(0)

    with open(os.path.join(CORE_DATA_DIR, f'{pid}-{vid}b', "failing_tests"), "r") as f:
        txt = f.read().strip()

    blocks = re.split(r'(?=^--- \S+::\S+)', txt, flags=re.MULTILINE)

    for block in blocks:
        lines = block.strip()
        if len(lines) == 0:
            continue

        lines = lines.split('\n')
        
        trace_start = next((i for i, line in enumerate(lines) if re.match(r'^\tat \S+\(\S+\)', line)), len(lines))

        print('Failing test)', lines[0])
        test_path = lines[0][4:lines[0].find('::')]
        test_path = os.path.join('test/', test_path.replace('.', '/') + '.java')
        test_name = lines[0][lines[0].find('::') + 2:]
        print(test_name)

        get_test_code(test_path, test_name)

        #print('Error message)', lines[1:trace_start])
        #print('Stack trace)', lines[trace_start:])
        
        