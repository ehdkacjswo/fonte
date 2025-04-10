import sys

sys.path.append('/root/workspace/analysis/lib/')
#from util import *
from result_gen import *
from analysis import *
from compare import *

def RQ1():
    settings = \
        [('No Diff', {'tracker': 'git', 'diff_tool': None, 'use_br' : False, 'beta': 0.1}), \
        ('Whole File', {'tracker': 'git', 'diff_tool': 'file', 'diff_type': 'base', 'use_br' : False, 'beta': 0.1}), \
        ('Git Diff', {'tracker': 'git', 'diff_tool': 'base', 'diff_type': 'base', 'use_br' : False, 'beta': 0.1}), \
        ('GumTree Diff', {'tracker': 'git', 'diff_tool': 'gumtree', 'diff_type': 'base', 'use_br' : False, 'beta': 0.1})]
    
    # Print metrics
    for (diff_type, set_dict) in settings:
        print(f'[{diff_type}]')
        print_metric(method='bug2commit', stage2='precise', bisect_setting=set_dict)

    # Compare settings
    for ind in range(len(settings) - 1):
        print(f'Comparing {settings[ind][0]}, {settings[ind + 1][0]}')
        compare_settings(org_method='bug2commit', new_method='bug2commit', \
            org_setting=(settings[ind][1] | {'stage2' : 'precise'}), \
            new_setting=(settings[ind + 1][1] | {'stage2' : 'precise'}))

def RQ2():
    settings = \
        [('No Identifier', {'tracker': 'git', 'diff_tool': 'gumtree', 'diff_type' : 'gumtree_id', 'adddel': 'all_uni', 'use_br' : False, 'classify_id' : False, 'use_id' : False, 'beta': 0.1}), \
        ('Use Identifier', {'tracker': 'git', 'diff_tool': 'gumtree', 'diff_type': 'gumtree_id', 'adddel': 'all_uni', 'use_br' : False, 'classify_id' : False, 'use_id' : True, 'beta': 0.1})]
    # Print metrics

    for (diff_type, set_dict) in settings:
        print(f'[{diff_type}]')
        print_metric(method='bug2commit', stage2='precise', bisect_setting=set_dict)

    # Compare settings
    for ind in range(len(settings) - 1):
        print(f'Comparing {settings[ind][0]}, {settings[ind + 1][0]}')
        compare_settings(org_method='bug2commit', new_method='bug2commit', \
            org_setting=(settings[ind][1] | {'stage2' : 'precise'}), \
            new_setting=(settings[ind + 1][1] | {'stage2' : 'precise'}))

def RQ3():
    settings = \
        [('No Identifier', {'tracker': 'git', 'diff_tool': 'gumtree', 'diff_type': 'gumtree_id', 'adddel': 'all_uni', 'use_br' : False, 'classify_id' : False, 'use_id' : True, 'beta': 0.1}), \
        ('Use Identifier', {'tracker': 'git', 'diff_tool': 'gumtree', 'diff_type': 'gumtree_id', 'adddel': 'all_uni', 'use_br' : False, 'classify_id' : True, 'use_id' : True, 'beta': 0.1})]
    # Print metrics

    for (diff_type, set_dict) in settings:
        print(f'[{diff_type}]')
        print_metric(method='bug2commit', stage2='precise', bisect_setting=set_dict)

    # Compare settings
    for ind in range(len(settings) - 1):
        print(f'Comparing {settings[ind][0]}, {settings[ind + 1][0]}')
        compare_settings(org_method='bug2commit', new_method='bug2commit', \
            org_setting=(settings[ind][1] | {'stage2' : 'precise'}), \
            new_setting=(settings[ind + 1][1] | {'stage2' : 'precise'}))


if __name__ == "__main__":
    RQ2()