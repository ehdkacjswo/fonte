import os

CORE_DATA_DIR = '/root/workspace/data/Defects4J/core'

if __name__ == "__main__":
    for project in os.listdir(CORE_DATA_DIR):
        print(f'Working on project : {project}')
        #project = 'Cli-29b'
        proj_dir = os.path.join(CORE_DATA_DIR, project, 'git')

        with open(os.path.join(proj_dir, 'precise_validation.csv'), 'r') as file:
            a = file.readlines()
        
        with open(os.path.join(proj_dir, 'precise_validation_noOpenRewrite.csv'), 'r') as file:
            b = file.readlines()
        
        if len(a) != len(b):
            print('Different length!')
        
        for c, d in zip(a, b):
            if c.endswith('E'):
                print('Rewrite', c)
            if d.endswith('E'):
                print('NoRewrite', d)
            if c != d:
                print('Different!')
    
        #break
