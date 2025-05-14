import sys

sys.path.append('/root/workspace/analysis/lib/')
#from util import *
from result_gen import *
from analysis import *
from compare import *

def RQ(settings):
    # Print metrics
    for (diff_type, stage2, method, set_dict) in settings:
        print(f'[{diff_type}]')
        print_metric(method=method, stage2=stage2, bisect_setting=set_dict)

    # Compare settings
    for ind in range(len(settings) - 1):
        print(f'Comparing {settings[ind][0]}, {settings[ind + 1][0]}')
        compare_settings(org_method=settings[ind][2], new_method=settings[ind + 1][2], \
            org_setting=(settings[ind][3] | {'stage2' : settings[ind][1]}), \
            new_setting=(settings[ind + 1][3] | {'stage2' : settings[ind + 1][1]}))

if __name__ == "__main__":
    RQ1 = \
        [('Bug2Commit (Bug report)', 'precise', 'bug2commit', {'tracker': 'git', 'diff_tool': None, 'use_br' : False, 'beta': 0.1}), \
        ('Fonte', 'precise', 'fonte', {}), \
        ('Bug2Commit (Bug report)', 'precise', 'bug2commit', {'tracker': 'git', 'diff_tool': None, 'use_br' : False, 'beta': 0.1}), \
        ('New Bug2Commit', 'precise', 'bug2commit', {'tracker': 'git', 'diff_tool': 'gumtree', 'diff_type': 'gumtree_id', 'use_br' : False, 'classify_id' : True, 'use_id' : True, 'beta': 0.1})]

    # Fine-grained
    RQ2 = \
        [('No Diff', 'precise', 'bug2commit', {'tracker': 'git', 'diff_tool': None, 'use_br' : False, 'beta': 0.1}), \
        ('Whole File', 'precise', 'bug2commit', {'tracker': 'git', 'diff_tool': 'file', 'diff_type': 'base', 'use_br' : False, 'beta': 0.1}), \
        ('Git Diff', 'precise', 'bug2commit', {'tracker': 'git', 'diff_tool': 'base', 'diff_type': 'base', 'use_br' : False, 'beta': 0.1}), \
        ('GumTree Diff', 'precise', 'bug2commit', {'tracker': 'git', 'diff_tool': 'gumtree', 'diff_type': 'base', 'use_br' : False, 'beta': 0.1})]

    RQ3_greedy = \
        [('No Identifier', 'precise', 'bug2commit', {'tracker': 'git', 'diff_tool': 'gumtree', 'diff_type' : 'greedy_id', 'use_br' : False, 'use_id' : False, 'beta': 0.1}), \
        ('Use Identifier', 'precise', 'bug2commit', {'tracker': 'git', 'diff_tool': 'gumtree', 'diff_type': 'greedy_id', 'use_br' : False, 'use_id' : True, 'beta': 0.1})]
    
    # Use full identifier
    RQ3_gumtree = \
        [('No Identifier', 'precise', 'bug2commit', {'tracker': 'git', 'diff_tool': 'gumtree', 'diff_type' : 'gumtree_id', 'use_br' : False, 'classify_id' : False, 'use_id' : False, 'beta': 0.1}), \
        ('Use Identifier', 'precise', 'bug2commit', {'tracker': 'git', 'diff_tool': 'gumtree', 'diff_type': 'gumtree_id', 'use_br' : False, 'classify_id' : False, 'use_id' : True, 'beta': 0.1})]
    
    RQ4 = \
        [('No classifying', 'precise', 'bug2commit', {'tracker': 'git', 'diff_tool': 'gumtree', 'diff_type': 'gumtree_id', 'use_br' : False, 'classify_id' : False, 'use_id' : True, 'beta': 0.1}), \
        ('Classifying', 'precise', 'bug2commit', {'tracker': 'git', 'diff_tool': 'gumtree', 'diff_type': 'gumtree_id', 'use_br' : False, 'classify_id' : True, 'use_id' : True, 'beta': 0.1})]
    
    RQ(RQ1)