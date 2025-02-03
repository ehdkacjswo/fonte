import os
import pandas as pd

CORE_DATA_DIR = '/root/workspace/data/Defects4J/core'

if __name__ == "__main__":
    # Style change detection results
    precise_same = {'U' : 0, 'C' : 0, 'N' : 0, 'E' : 0}
    precise_diff = {'U' : 0, 'C' : 0, 'N' : 0, 'E' : 0}
    original = {'U' : 0, 'C' : 0, 'N' : 0, 'E' : 0}

    # Extra data
    to_U, to_C = 0, 0
    missing = 0

    for project in os.listdir(CORE_DATA_DIR):
        #project = 'Cli-29b'
        proj_dir = os.path.join(CORE_DATA_DIR, project, 'git')

        # Validation data
        with open(os.path.join(proj_dir, 'precise_validation.csv'), 'r') as file:
            a = file.readlines()
        
        with open(os.path.join(proj_dir, 'precise_validation_noOpenRewrite.csv'), 'r') as file:
            b = file.readlines()
        
        if len(a) != len(b):
            print(f'[{project}] Error on precise validation')
            continue
        
        a_dict = {}
        
        # Precise validation check
        for x, y in zip(a, b):
            x = x.rstrip().split(',')
            y = y.rstrip().split(',')
            
            # Source path not changed
            if x[1] == x[2]:
                a_dict[(x[0], x[1])] = x[3]
                precise_same[x[3]] += 1
            
            # Source path changed
            else:
                precise_diff[x[3]] += 1
                if x[3] == 'U':
                    print(f'[{project}] Different path update {x[0]} : {x[1]}, {x[2]}')

            # Error check
            if x != y:
                print(f'[{project}] Astyle changed result')
                print(x)
                print(y)

            if x[-1] == 'E':
                print(f'[{project}] Precise validaiton with astyle causes error')
                print(x)

            if y[-1] == 'E':
                print(f'[{project}] Precise validaiton without astyle causes error')
                print(y)
        
        # Original results
        with open(os.path.join(proj_dir, 'validation_noOpenRewrite.csv'), 'r') as file:
            c = file.readlines()
        
        c_dict = {}

        # Original validation check
        for z in c:
            z = z.rstrip().split(',')
            key = (z[0], z[1])
            c_dict[key] = z[2]
            original[z[2]] += 1

            if key not in a_dict:
                if z[2] != 'N':
                    print(f'[{project}] Missing precise style change')
                    print(z)
            
            elif z[2] != a_dict[key]:
                #print(f'[{project}] Precise style change different')
                #print(f'{key} > precise:{a_dict[key]}, original:{z[2]}')

                if a_dict[key] == 'U':
                    to_U += 1
                
                elif a_dict[key] == 'C':
                    to_C += 1
        
        for key in a_dict.keys():
            if key not in c_dict:
                #print(f'[{project}] Data not in original')
                #print(key, a_dict[key])
                missing += 1
    
    print('Precise - Same path) ', precise_same)
    print('Precise - Different path) ', precise_diff)
    print('Original) ', original)
    print(f'Change to style change : {to_U}, to modification : {to_C}')
    print(missing)