import os, shutil
import pandas as pd

# (commit_hash, pid) : [vid]
# pid_vid has commit_hash as past version
commits_dict = {}

diff_dir = '/root/workspace/data/Defects4J/diff/'
projects = ['Cli', 'Closure', 'Codec', 'Compress', 'Gson', 'JacksonCore', 'Jsoup', 'Lang', 'Math', 'Mockito', 'Time']

# Check only target projects
for dir_name in os.listdir(diff_dir):
    try:
        project = dir_name[:dir_name.index('-')]
    except:
        continue

    print(f'Working on {dir_name}')
    
    total_dir = os.path.join(diff_dir, project)
    project_dir = os.path.join(diff_dir, dir_name)
    os.makedirs(os.path.join(total_dir, project), exist_ok=True)

    # Get the list of current commits for target version
    for commit in os.listdir(project_dir):
        if os.path.exists(os.path.join(total_dir, commit)):
            continue
        
        # Copy the entire directory
        shutil.copytree(os.path.join(project_dir, commit), os.path.join(total_dir, commit))