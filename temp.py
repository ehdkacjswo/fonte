from scipy import sparse
import pandas as pd
from sbfl.base import SBFL
import os
from intervaltree import Interval, IntervalTree

CORE_DIR = '/root/workspace/data/Defects4J/core'

# Total number of validations, Results changed + New style change

def get_style_change_commits(with_Rewrite=True):
    res = dict()
    
    for project in os.listdir(CORE_DIR):
        if with_Rewrite:
            val_df = pd.read_csv(
                os.path.join(CORE_DIR, project, 'git', "validation.csv"), 
                header=None,
                names=["commit", "src_path", "AST_diff"])
            
            for row in val_df.itertuples(index=True):
                res[(project, row.commit, row.src_path, row.src_path)] = row.AST_diff
        
        else:
            val_df = pd.read_csv(
                #os.path.join(CORE_DIR, project, 'git', "precise_validation.csv"), 
                os.path.join(CORE_DIR, project, 'git', "validation_noOpenRewrite.csv"), 
                header=None,
                #names=["commit", "before_src_path", "after_src_path", "AST_diff"])
                names=["commit", "src_path", "AST_diff"])
            
            for row in val_df.itertuples(index=True):
                #res[(project, row.commit, row.before_src_path, row.after_src_path)] = row.AST_diff
                res[(project, row.commit, row.src_path, row.src_path)] = row.AST_diff

    return res

if __name__ == "__main__":
    basic = get_style_change_commits(True)
    precise = get_style_change_commits(False)

    print(f"Total number of validation : {len(basic)} {len(precise)}")

    changed = []
    new_change = []
    err_num = 0
    new_err_num = 0

    """for (project, commit, before_src_path, after_src_path), result in basic.items():
        if (project, commit, before_src_path, after_src_path) not in precise:
            print(commit, before_src_path, after_src_path)"""

    precise_err_prj = set()
    basic_err_prj = set()

    for (project, commit, before_src_path, after_src_path), result in precise.items():
        if result == 'E':
            print(project, commit, before_src_path, after_src_path)
        
        """if result == 'U' and basic[(project, commit, before_src_path, after_src_path)] == 'E':
            print((project, commit, before_src_path, after_src_path))"""
            
        
        """elif result == 'U':
            if (project, commit, before_src_path, after_src_path) not in basic:
                print(project, commit, before_src_path, after_src_path)
                print(before_src_path == after_src_path)
            elif basic[(project, commit, before_src_path, after_src_path)] != 'U':
                print(project, commit, before_src_path, after_src_path)
                print(before_src_path == after_src_path)"""
        
        """if (project, commit, before_src_path, after_src_path) in basic:
            if result != basic[(project, commit, before_src_path, after_src_path)]:
                print(project, commit, before_src_path, after_src_path, result, basic[(project, commit, before_src_path, after_src_path)])"""
    
    for (project, commit, before_src_path, after_src_path), result in basic.items():
        if result == 'E':   
            basic_err_prj.add(project)
            err_num += 1
    
    print(err_num, basic_err_prj)
    print(new_err_num, precise_err_prj)
    print(len(basic_err_prj), len(precise_err_prj))
    print(len(basic_err_prj | precise_err_prj))
    print(basic_err_prj | precise_err_prj)
    
    #print(err_num)