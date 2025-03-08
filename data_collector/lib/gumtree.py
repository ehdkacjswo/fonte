import subprocess, json, re, sys
from charset_normalizer import from_path

sys.path.append('/root/workspace/data_collector/lib/')
from utils import CustomInterval

# Directory mounted on "original" adfasdf
DIR_NAME = '/home/coinse/doam/fonte/tmp'

def file_encode(no_after_src, no_before_src):
    if not no_after_src:
        encoded_code = from_path('/root/workspace/tmp/after.java').best()
        
        if encoded_code is not None:
            with open('/root/workspace/tmp/after.java', 'w', encoding='utf-8') as file:
                file.write(str(encoded_code))
    
    if not no_before_src:
        encoded_code = from_path('/root/workspace/tmp/before.java').best()
        
        if encoded_code is not None:
            with open('/root/workspace/tmp/before.java', 'w', encoding='utf-8') as file:
                file.write(str(encoded_code))

# Return addition, deletion token range
# addition, deletion = Token range
def gumtree_diff(before_src_path, after_src_path):
    diff_cmd = f'docker run --rm -v {DIR_NAME}:/diff gumtree textdiff -g java-jdtc -m gumtree-simple -f JSON {before_src_path} {after_src_path}'
    p = subprocess.Popen(diff_cmd, shell=True, stdout=subprocess.PIPE)
    stdout, _ = p.communicate()

    # Diff error
    if p.returncode != 0: 
        #with open('/root/workspace/error.txt', 'a') as file:
        #    file.write(f'GumTreeDiff returns error : {stdout}\n')
        return None, None

    try:
        diff_json = json.loads(stdout.decode(encoding='utf-8', errors='ignore'))
    except:
        #with open('/root/workspace/error.txt', 'a') as file:
        #    file.write(f'Diff output decoding error\n')
        return None, None # Decoding error
    
    tree_pattern = r".+\[(\d+),(\d+)\]"
    add_intvl = CustomInterval()
    del_intvl = CustomInterval()

    # {'action': 'action', 'tree': 'type:'}
    for action in diff_json['actions']:
        # Insertion (tree, node) action
        if action['action'].startswith('insert'):
            match = re.match(tree_pattern, action['tree'])

            if match:
                start_pos = int(match.group(1))
                end_pos = int(match.group(2))
                add_intvl |= CustomInterval(start_pos, end_pos - 1)
            
        # Update node action
        # {'action': 'update-node', 'tree': 'SimpleName: classNames [2810,2820]', 'label': 'className'}
        elif action['action'] == 'update-node':
            match = re.match(tree_pattern, action['tree'])

            if match:
                start_pos = int(match.group(1))
                end_pos = int(match.group(2))
                #gumtree_update_dict[CustomInterval(start_pos, end_pos - 1)] = action['label']
                del_intvl |= CustomInterval(start_pos, end_pos - 1)
                add_intvl |= CustomInterval(action['targetPos'], action['targetPos'] + action['targetLength'] - 1)

        # Deletion (tree, node) action
        # {'action': 'delete-tree', 'tree': 'ReturnStatement [14519,14532]'}
        elif action['action'].startswith('delete'):
            match = re.match(tree_pattern, action['tree'])

            if match:
                start_pos = int(match.group(1))
                end_pos = int(match.group(2))
                del_intvl |= CustomInterval(start_pos, end_pos - 1)
        
        """
        # Move action
        # {'action': 'move-tree', 'tree': 'IfStatement [14204,14459]', 'parent': 'Block [12965,14472]', 'at': 7}
        elif action['action'] == 'move-tree':
            match = re.match(tree_pattern, action['tree'])

            if match:
                start_pos = int(match.group(1))
                end_pos = int(match.group(2))
        """

    return add_intvl, del_intvl

# Parse the file and returns in json format
def gumtree_parse(filename):
    parse_cmd = f'docker run --rm -v {DIR_NAME}:/diff gumtree parse -g custom-jdt -f JSON {filename}'
    p = subprocess.Popen(parse_cmd, shell=True, stdout=subprocess.PIPE)
    stdout, _ = p.communicate()

    if p.returncode != 0: # Parsing error
        with open('/root/workspace/error.txt', 'a') as file:
            file.write(f'Parse returns error : {stdout}\n')
        return None

    try:
        parse_json = json.loads(stdout.decode(encoding='utf-8', errors='ignore'))
    except:
        with open('/root/workspace/error.txt', 'a') as file:
            file.write(f'Parse output decoding error\n')
        return None # Decoding error
    #print(json.dumps(parse_json, indent=4))
    id_intvl_dict = {'class':CustomInterval(), 'method':CustomInterval(), 'variable':CustomInterval(), 'comment':CustomInterval()}

    for data in parse_json:
        if 'isClass' in data:
            id_intvl_dict['class'] |= CustomInterval(int(data['pos']), int(data['pos']) + int(data['length']) - 1)
        elif 'isMethod' in data:
            id_intvl_dict['method'] |= CustomInterval(int(data['pos']), int(data['pos']) + int(data['length']) - 1)
        elif 'isComment' in data:
            id_intvl_dict['comment'] |= CustomInterval(int(data['pos']), int(data['pos']) + int(data['length']) - 1)
        elif 'isVariable' in data:
            id_intvl_dict['variable'] |= CustomInterval(int(data['pos']), int(data['pos']) + int(data['length']) - 1)
    
    """with open(f'/root/workspace/tmp/{filename}', 'rb') as file:
        filedata = file.read().decode(encoding='utf-8', errors='ignore')
    
    #for a in filedata:
    #    print(a)"""

    return id_intvl_dict

if __name__ == "__main__":
    addition_interval = CustomInterval()
    deletion_interval = CustomInterval()

    addition_interval.interval_data = interval[-inf, inf]
    deletion_interval.interval_data = interval[-inf, inf]

    gumtree_diff(False, False)
