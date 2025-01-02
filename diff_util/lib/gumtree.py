import subprocess, json, re
from intervaltree import Interval, IntervalTree

# Directory mounted on "original" adfasdf
DIR_NAME = '/home/coinse/doam/fonte/tmp'

def get_all_labels(tree_json):
    if 'label' in tree_json:
        res = [tree_json['label']]
    
    else:
        res = []

    
    for child in tree_json['children']:
        res += all_label(child)
    
    return res

# Return list of every labels in given token range of the tree
def get_labels_in_range(tree_json, token_range):
    
    # Check if given interval is subset of token range
    def interval_in_range(begin, length):
        interval = Interval(begin, begin + length)
        for sub_range in token_range:
            if sub_range.contains_interval(interval):
                return True
        return False

    label_list = []

    if 'label' in tree_json and interval_in_range(int(tree_json['pos'], int(tree_json['length']))):
        label_list = [tree_json['label']]

    for child in tree_json['children']:
        if interval_in_range(int(child['pos']), int(child['length']))
        if token_range.overlaps(int(child['pos']), int(child['pos']) + int(child['length'])):
            label_list += get_labels_in_range(child, token_range)

    return label_list

def token_range(filename):
    res = [(-1, -1)]

    with open(filename, 'r') as file:
        lines = file.readlines()
    
    a = 0
    for line in lines:
        res.append((res[-1][1] + 1, res[-1][1] + len(line)))
        print(line, res[-1])
        a += len(line)
    print(a)
    return res

def tree_json(filepath):
    parse_cmd = f'docker run --rm -v {DIR_NAME}:/diff gumtreediff/gumtree parse -g java-jdt -f JSON {filepath}'
    p = subprocess.Popen(parse_cmd, shell=True, stdout=subprocess.PIPE)
    stdout, _ = p.communicate()

    if p.returncode != 0: # Parsing error
        return None

    try:
        return json.loads(stdout.decode(encoding='utf-8', errors='ignore'))
    except:
        return None # Decoding error

# 
def diff_json(no_after_src, no_before_src, addition_range, deletion_range):
    if addition and deletion:
        # Perform diff
        diff_cmd = f'docker run --rm -v {DIR_NAME}:/diff gumtreediff/gumtree textdiff -g java-jdt -m gumtree-simple -f JSON before.java after.java'
        p = subprocess.Popen(diff_cmd, shell=True, stdout=subprocess.PIPE)
        stdout, _ = p.communicate()

        if p.returncode != 0: # Diff error
            return None

        try:
            diff_json = json.loads(stdout.decode(encoding='utf-8', errors='ignore'))
        except:
            return None # Decoding error
        
        tree_pattern = r".+\[(\d+),(\d+)\]"
        addition_range = IntervalTree()
        deletion_tree = IntervalTree()

        # 각자 별개가 아니라 range 모아서 한번에
        for action in diff_json['actions']:
            print(action)
            # Insertion (tree, node) action
            if action['action'].startswith('insert'):
                #print(action)
                match = re.match(tree_pattern, action['tree'])

                if match:
                    start_pos = int(match.group(1))
                    end_pos = int(match.group(2))
                    addition_range[start_pos : end_pos] = True
                    #print(find_tree(target_jsons[1]['root'], start_pos, end_pos))
                
            # Update node action
            # {'action': 'update-node', 'tree': 'SimpleName: classNames [2810,2820]', 'label': 'className'}
            elif action['action'] == 'update-node':
                #print(action)
                match = re.match(tree_pattern, action['tree'])

                if match:
                    start_pos = int(match.group(1))
                    end_pos = int(match.group(2))
                    print(find_tree(target_jsons[0]['root'], start_pos, end_pos))
            
            """elif action['action'] == 'move-tree':
                #print(action)
                match = re.match(tree_pattern, action['tree'])

                if match:
                    start_pos = int(match.group(1))
                    end_pos = int(match.group(2))
                    print(find_tree(target_jsons[0]['root'], start_pos, end_pos))"""
            
            # Deletion (tree, node) action
            elif action['action'].startswith('delete'):
                #print(action)
                match = re.match(tree_pattern, action['tree'])

                if match:
                    start_pos = int(match.group(1))
                    end_pos = int(match.group(2))
                    print(find_tree(target_jsons[0]['root'], start_pos, end_pos))

    

# Gumtree is executed by host's 
def gumtree_diff(dir_name='/home/coinse/doam/fonte', old_file='old.java', new_file='new.java'):
    # Perform diff
    diff_cmd = f'docker run --rm -v {dir_name}:/diff gumtreediff/gumtree textdiff -g java-jdt -m gumtree-simple -f JSON {old_file} {new_file}'
    p = subprocess.Popen(diff_cmd, shell=True, stdout=subprocess.PIPE)
    stdout, _ = p.communicate()

    try:
        diff_json = json.loads(stdout.decode(encoding='utf-8', errors='ignore'))
    except UnicodeDecodeError as e:
        print(cmd)
        raise e
    
    # Parse files
    parse_cmd = 'docker run --rm -v {0}:/diff gumtreediff/gumtree parse -g java-jdt -f JSON {1}'

    target_files = [old_file, new_file]
    target_jsons = []
    token_ranges = []

    for filename in target_files:
        cmd = parse_cmd.format(dir_name, filename)

        p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
        stdout, _ = p.communicate()

        try:
            target_jsons.append(json.loads(stdout.decode(encoding='utf-8', errors='ignore')))
        except UnicodeDecodeError as e:
            print(cmd)
            raise e
        
        # Get token range
        token_ranges.append([(-1, -1)])

        with open(filename, 'r') as file:
            lines = file.readlines()
        
        for line in lines:
            token_ranges[-1].append((token_ranges[-1][-1][1] + 1, token_ranges[-1][-1][1] + len(line)))
    
    print(target_jsons[0]['root']['pos'], target_jsons[0]['root']['length'])
    print(token_ranges[0][-1])
            
    
    deletion = []
    addition = []

    tree_pattern = r".+\[(\d+),(\d+)\]"

    # 각자 별개가 아니라 range 모아서 한번에
    for action in diff_json['actions']:
        print(action)
        # Insertion (tree, node) action
        if action['action'].startswith('insert'):
            #print(action)
            match = re.match(tree_pattern, action['tree'])

            if match:
                start_pos = int(match.group(1))
                end_pos = int(match.group(2))
                print(find_tree(target_jsons[1]['root'], start_pos, end_pos))
              
        # Update node action
        elif action['action'] == 'update-node':
            #print(action)
            match = re.match(tree_pattern, action['tree'])

            if match:
                start_pos = int(match.group(1))
                end_pos = int(match.group(2))
                print(find_tree(target_jsons[0]['root'], start_pos, end_pos))
        
        elif action['action'] == 'move-tree':
            #print(action)
            match = re.match(tree_pattern, action['tree'])

            if match:
                start_pos = int(match.group(1))
                end_pos = int(match.group(2))
                print(find_tree(target_jsons[0]['root'], start_pos, end_pos))
        
        # Deletion (tree, node) action
        elif action['action'].startswith('delete'):
            #print(action)
            match = re.match(tree_pattern, action['tree'])

            if match:
                start_pos = int(match.group(1))
                end_pos = int(match.group(2))
                print(find_tree(target_jsons[0]['root'], start_pos, end_pos))
        
        # move-node, move-tree
        # addition-tree, addition-node : Super class for move, insert

    
    #print(diff_json)

if __name__ == "__main__":
    gumtree_diff()