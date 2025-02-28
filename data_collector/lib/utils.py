import subprocess, sys, time
sys.path.append('/root/workspace/data_collector/lib/')
from gumtree import CustomInterval

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

# Code txt could be possibly "None" (Failed to get code data)
def get_tokens_intvl(code_txt, intvl):
    # When interval is empty, it doesn't matter whether code_txt is None
    if intvl.is_empty():
        return []
    
    # When interval is not empty, code_txt must not be None
    if code_txt is None:
        return None

    return [''.join(code_txt[int(sub_intvl[0]) + 1 : int(sub_intvl[1]) + 1]) for sub_intvl in intvl]