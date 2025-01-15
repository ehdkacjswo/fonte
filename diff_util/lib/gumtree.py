from __future__ import annotations

import subprocess, json, re
from interval import interval, inf
from typing import Optional
from charset_normalizer import from_path

# Directory mounted on "original" adfasdf
DIR_NAME = '/home/coinse/doam/fonte/tmp'

# -0.5, +0.5 해서 강제로 겹치게
# 1. Building line range (Add range)
# 2. With each line in line range, build token range (Add range, number in)
# 3. Get intersection of two ranges (Intersection)
# 4. Find every labels in token range (Intersect, range in)
# Possible issue : [n.5, n.5 from intersection]

class CustomInterval():
    # 
    def wide_interval(self, start:Optional[int]=None, end:Optional[int]=None):
        if start is None:
            return interval()
        elif end is None:
            return interval[start - 0.5, start + 0.5]
        else:
            return interval[start - 0.5, end + 0.5]

    def __init__(self, start:Optional[int]=None, end:Optional[int]=None):
        self.interval_data = self.wide_interval(start, end)

    # Add interval
    def add_interval(self, start:Optional[int]=None, end:Optional[int]=None):
        self.interval_data |= self.wide_interval(start, end)
    
    # Intersection
    def __and__(self, other:CustomInterval):
        ret = CustomInterval()
        ret.interval_data = self.interval_data & other.interval_data
        return ret
    
    # 
    def __contains__(self, item):
        if isinstance(item, int):
            return interval[item] in self.interval_data
        elif isinstance(item, CustomInterval):
            return item.interval_data in self.interval_data
        else: # Error
            return None
    
    def is_empty(self):
        return len(self.interval_data) == 0

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

# Return addition, deletion, update token range
# addition, deletion = Token range
# update = {interval(token pos, pos + length) : updated label}
def gumtree_diff_token_range(no_after_src, no_before_src, addition_range, deletion_range, use_comment=False):
    
    # Convert line level range to token level range
    def line_to_token_range(lines, line_range):
        token_range = CustomInterval()
        token_cnt = 0
        
        # Add range for line in given range
        for line_cnt, line in enumerate(lines):
            if line_cnt + 1 in line_range:
                token_range.add_interval(token_cnt, token_cnt + len(line) - 1)
            token_cnt += len(line)

        return token_range
    
    # Convert line level range to token level range
    if no_after_src:
        addition_token_range = CustomInterval()
    else:
        with open('/root/workspace/tmp/after.java', 'r', encoding='utf-8', errors='ignore') as file:
            after_lines = file.readlines()
        addition_token_range = line_to_token_range(after_lines, addition_range)
    
    if no_before_src:
        deletion_token_range = CustomInterval()
    else:
        with open('/root/workspace/tmp/before.java', 'r', encoding='utf-8', errors='ignore') as file:
            before_lines = file.readlines()
        deletion_token_range = line_to_token_range(before_lines, deletion_range)
    
    # File creation/deletion (Update has to be empty)
    if no_after_src or no_before_src:
        return addition_token_range, deletion_token_range, dict()

    # File modification
    else:
        diff_cmd = f'docker run --rm -v {DIR_NAME}:/diff gumtree textdiff -g java-jdt -m gumtree-simple -f JSON before.java after.java'
        p = subprocess.Popen(diff_cmd, shell=True, stdout=subprocess.PIPE)
        stdout, _ = p.communicate()

        if p.returncode != 0: # Diff error
            print('OJODSFO')
            return None

        try:
            diff_json = json.loads(stdout.decode(encoding='utf-8', errors='ignore'))
        except:
            return None # Decoding error
        
        print(diff_json)
        tree_pattern = r".+\[(\d+),(\d+)\]"
        gumtree_addition_range = CustomInterval()
        gumtree_deletion_range = CustomInterval()
        gumtree_update_dict = dict()

        # {'action': 'action', 'tree': 'type:'}
        for action in diff_json['actions']:
            # Insertion (tree, node) action
            if action['action'].startswith('insert'):
                match = re.match(tree_pattern, action['tree'])

                if match:
                    start_pos = int(match.group(1))
                    end_pos = int(match.group(2))
                    gumtree_addition_range.add_interval(start_pos, end_pos - 1)
                
            # Update node action
            # {'action': 'update-node', 'tree': 'SimpleName: classNames [2810,2820]', 'label': 'className'}
            elif action['action'] == 'update-node':
                #print(action)
                match = re.match(tree_pattern, action['tree'])

                if match:
                    start_pos = int(match.group(1))
                    end_pos = int(match.group(2))
                    gumtree_update_dict[CustomInterval(start_pos, end_pos - 1)] = action['label']

            # Deletion (tree, node) action
            # {'action': 'delete-tree', 'tree': 'ReturnStatement [14519,14532]'}
            elif action['action'].startswith('delete'):
                match = re.match(tree_pattern, action['tree'])

                if match:
                    start_pos = int(match.group(1))
                    end_pos = int(match.group(2))
                    gumtree_deletion_range.add_interval(start_pos, end_pos - 1)
            
            """
            # {'action': 'move-tree', 'tree': 'IfStatement [14204,14459]', 'parent': 'Block [12965,14472]', 'at': 7}
            elif action['action'] == 'move-tree':
                #print(action)
                match = re.match(tree_pattern, action['tree'])

                if match:
                    start_pos = int(match.group(1))
                    end_pos = int(match.group(2))
                    print(find_tree(target_jsons[0]['root'], start_pos, end_pos))
            """
        
        # Get intersection
        gumtree_update_dict = {update_range : label for (update_range, label) in gumtree_update_dict.items() if update_range in deletion_token_range}

        addition_token_range &= gumtree_addition_range
        deletion_token_range &= gumtree_deletion_range

        return addition_token_range, deletion_token_range, gumtree_update_dict

# Parse the file and returns in json format
def gumtree_parse(filename, token_range):
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
        label_list = []

        if 'label' in tree_json and CustomInterval(int(tree_json['pos']), int(tree_json['pos']) + int(tree_json['length']) - 1) in token_range:
            label_list = [tree_json['label']]

        for child in tree_json['children']:
            if not (token_range & CustomInterval(int(child['pos']), int(child['pos']) + int(child['length']) - 1)).is_empty():
                label_list += get_labels_in_range(child, token_range)

        return label_list

    parse_cmd = f'docker run --rm -v {DIR_NAME}:/diff gumtree parse -g java-jdt -f JSON {filename}'
    p = subprocess.Popen(parse_cmd, shell=True, stdout=subprocess.PIPE)
    stdout, _ = p.communicate()

    if p.returncode != 0: # Parsing error
        print('OJODSFO')
        return None

    try:
        tree_json = json.loads(stdout.decode(encoding='utf-8', errors='ignore'))
    except:
        return None # Decoding error
    #print(json.dumps(tree_json, indent=4))
    print(tree_json)
    
    return get_labels_in_range(tree_json['root'], token_range)

# addition [class, method, variable, comment], deletion [class, method, variable, comment]
def gumtree_diff(no_after_src=False, no_before_src=False, addition_range=interval[-inf,inf], deletion_range=interval[-inf,inf], use_comment=False):
    #file_encode(no_after_src, no_before_src)
    
    # Get token ranges
    addition_token_range, deletion_token_range, update_dict = \
        gumtree_diff_token_range(no_after_src, no_before_src, addition_range, deletion_range)
    
    # Get tokens
    if no_after_src:
        addition_tokens = []
    else:
        addition_tokens = gumtree_parse('after.java', addition_token_range)

    if no_before_src:
        deletion_tokens = []
    else:
        deletion_tokens = gumtree_parse('before.java', deletion_token_range)
    
    return addition_tokens, deletion_tokens

if __name__ == "__main__":
    addition_range = CustomInterval()
    deletion_range = CustomInterval()

    addition_range.interval_data = interval[-inf, inf]
    deletion_range.interval_data = interval[-inf, inf]

    gumtree_diff(False, False, addition_range, deletion_range)