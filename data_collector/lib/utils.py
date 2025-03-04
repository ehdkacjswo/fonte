from __future__ import annotations

import subprocess, sys, time
from interval import interval, inf

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

def log(filename, txt, out_txt=None, err_txt=None):
    with open(f'/root/workspace/data_collector/log/{filename}.log', 'a') as file:
        file.write(txt + '\n')

        if out_txt is not None:
            file.write('[ERROR] OUT\n' + out_txt.decode(encoding='utf-8', errors='ignore') + '\n')
        
        if err_txt is not None:
            file.write('[ERROR] ERR\n' + err_txt.decode(encoding='utf-8', errors='ignore') + '\n')

def time_to_str(start_time, end_time):
    hour, remainder = divmod(int(end_time - start_time), 3600)
    minute, second = divmod(remainder, 60)
    ms = int((end_time - start_time) * 1000) % 1000

    return f'{hour}h {minute}m {second}s {ms}ms'

def get_src_from_commit(commit, src_path):
    # Get the target file text
    p = subprocess.Popen(['git', 'show', f'{commit}:{src_path}'], \
    stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    code_txt, err_txt = p.communicate()

    # Error raised while copying file
    if p.returncode != 0:
        return None
    
    return code_txt.decode(encoding='utf-8', errors='ignore')