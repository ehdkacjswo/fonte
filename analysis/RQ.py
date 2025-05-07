import sys

sys.path.append('/root/workspace/analysis/lib/')
#from util import *
from result_gen import *
from analysis import *
from compare import *

def RQ(settings):
    # Print metrics
    for (diff_type, stage2, set_dict) in settings:
        print(f'[{diff_type}]')
        print_metric(method='bug2commit', stage2=stage2, bisect_setting=set_dict)

    # Compare settings
    for ind in range(len(settings) - 1):
        print(f'Comparing {settings[ind][0]}, {settings[ind + 1][0]}')
        compare_settings(org_method='bug2commit', new_method='bug2commit', \
            org_setting=(settings[ind][2] | {'stage2' : settings[ind][1]}), \
            new_setting=(settings[ind + 1][2] | {'stage2' : settings[ind + 1][1]}))

if __name__ == "__main__":
    # Fine-grained
    RQ2 = \
        [('No Diff', 'precise', {'tracker': 'git', 'diff_tool': None, 'use_br' : False, 'beta': 0.1}), \
        ('Whole File', 'precise', {'tracker': 'git', 'diff_tool': 'file', 'diff_type': 'base', 'use_br' : False, 'beta': 0.1}), \
        ('Git Diff', 'precise', {'tracker': 'git', 'diff_tool': 'base', 'diff_type': 'base', 'use_br' : False, 'beta': 0.1}), \
        ('GumTree Diff', 'precise', {'tracker': 'git', 'diff_tool': 'gumtree', 'diff_type': 'base', 'use_br' : False, 'beta': 0.1})]

    RQ3_greedy = \
        [('No Identifier', 'precise', {'tracker': 'git', 'diff_tool': 'gumtree', 'diff_type' : 'greedy_id', 'use_br' : False, 'use_id' : False, 'beta': 0.1}), \
        ('Use Identifier', 'precise', {'tracker': 'git', 'diff_tool': 'gumtree', 'diff_type': 'greedy_id', 'use_br' : False, 'use_id' : True, 'beta': 0.1})]
    
    # Use full identifier
    RQ3_gumtree = \
        [('No Identifier', 'precise', {'tracker': 'git', 'diff_tool': 'gumtree', 'diff_type' : 'gumtree_id', 'use_br' : False, 'classify_id' : False, 'use_id' : False, 'beta': 0.1}), \
        ('Use Identifier', 'precise', {'tracker': 'git', 'diff_tool': 'gumtree', 'diff_type': 'gumtree_id', 'use_br' : False, 'classify_id' : False, 'use_id' : True, 'beta': 0.1})]
    
    RQ4 = \
        [('No classifying', 'precise', {'tracker': 'git', 'diff_tool': 'gumtree', 'diff_type': 'gumtree_id', 'use_br' : False, 'classify_id' : False, 'use_id' : True, 'beta': 0.1}), \
        ('Classifying', 'precise', {'tracker': 'git', 'diff_tool': 'gumtree', 'diff_type': 'gumtree_id', 'use_br' : False, 'classify_id' : True, 'use_id' : True, 'beta': 0.1})]
    
    RQ(RQ4)