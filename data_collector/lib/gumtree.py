from __future__ import annotations

import subprocess, json, re
from interval import interval, inf
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
    def wide_interval(self, start=None, end=None):
        if start is None:
            return interval()
        elif end is None:
            return interval[start - 0.5, start + 0.5]
        else:
            return interval[start - 0.5, end + 0.5]

    def __init__(self, start=None, end=None):
        self.interval_data = self.wide_interval(start, end)
    
    # Intersection / Union
    def __and__(self, other:CustomInterval):
        ret = CustomInterval()
        ret.interval_data = self.interval_data & other.interval_data
        return ret
    
    def __rand__(self, other:CustomInterval):
        return self & other
    
    def __or__(self, other:CustomInterval):
        ret = CustomInterval()
        ret.interval_data = self.interval_data | other.interval_data
        return ret
    
    def __ror__(self, other:CustomInterval):
        return self | other
    
    def __contains__(self, item):
        if isinstance(item, int):
            return item in self.interval_data
        elif isinstance(item, CustomInterval):
            return item.interval_data in self.interval_data
        else: # Error
            return None
    
    def is_empty(self):
        for sub_interval in self.interval_data:
            if sub_interval[0] != sub_interval[1]:
                return False

        return True
    
    def components(self):
        return self.interval_data.components
    
    def __iter__(self):
        return iter(self.interval_data)
    
    def __str__(self):
        ret = ""
        for sub_intvl in self.interval_data:
            if sub_intvl[0] != sub_intvl[1]:
                ret += f'({int(sub_intvl[0]) + 1}, {int(sub_intvl[1])})'

        return '[' + ret + ']'
    
    def __repr__(self):
        ret = ""
        for sub_intvl in self.interval_data:
            if sub_intvl[0] != sub_intvl[1]:
                ret += f"({'-inf' if sub_intvl[0] == -inf else int(sub_intvl[0]) + 1}, {'inf' if sub_intvl[1] == inf else int(sub_intvl[1]) + 1})"

        return '[' + ret + ']'

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
def gumtree_diff(no_after_src, no_before_src):
    # File creation/deletion (Update has to be empty)
    if no_after_src:
        return CustomInterval(), CustomInterval(-inf, inf)
    
    elif no_before_src:
        return CustomInterval(-inf, inf), CustomInterval()

    # File modification
    else:
        diff_cmd = f'docker run --rm -v {DIR_NAME}:/diff gumtree textdiff -g java-jdtc -m gumtree-simple -f JSON before.java after.java'
        p = subprocess.Popen(diff_cmd, shell=True, stdout=subprocess.PIPE)
        stdout, _ = p.communicate()

        if p.returncode != 0: # Diff error
            with open('/root/workspace/error.txt', 'a') as file:
                file.write(f'GumTreeDiff returns error : {stdout}\n')
            return None, None

        try:
            diff_json = json.loads(stdout.decode(encoding='utf-8', errors='ignore'))
        except:
            with open('/root/workspace/error.txt', 'a') as file:
                file.write(f'Diff output decoding error\n')
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
