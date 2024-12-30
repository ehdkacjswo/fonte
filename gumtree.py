import subprocess, json, re

def all_label(tree_json):
    if 'label' in tree_json:
        res = [tree_json['label']]
    
    else:
        res = []

    
    for child in tree_json['children']:
        res += all_label(child)
    
    return res

# Get 
def find_tree(tree_json, start_pos, end_pos):
    while int(tree_json['pos']) != start_pos or int(tree_json['pos']) + int(tree_json['length']) != end_pos:
        start_ind = 0
        end_ind = len(tree_json['children']) - 1

        while start_ind < end_ind:
            pivot_ind = int((start_ind + end_ind) / 2)
            child = tree_json['children'][pivot_ind]

            # Pivot contains target position range
            if int(child['pos']) <= start_pos and int(child['pos']) + int(child['length']) >= end_pos:
                start_ind = pivot_ind
                end_ind = pivot_ind
                break
            
            # Pivot is after target position ragne
            elif int(child['pos']) > start_pos:
                end_ind = pivot_ind - 1
                continue
            
            # Pivot is before target position range
            else:
                start_ind = pivot_ind + 1
                continue

        tree_json = tree_json['children'][start_ind]
    return all_label(tree_json)

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
        cmd = parse_cmd.format(dir_name, old_file)

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