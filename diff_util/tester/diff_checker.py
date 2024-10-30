import os
import pandas as pd

# (commit_hash, pid) : [vid]
# pid_vid has commit_hash as past version
commits_dict = {}

diff_dir = '/root/workspace/data/Defects4J/diff/'
core_dir = '/root/workspace/data/Defects4J/core/'
projects = ['Cli', 'Closure', 'Codec', 'Compress', 'Gson', 'JacksonCore', 'Jsoup', 'Lang', 'Math', 'Mockito', 'Time']

# Check the data
# 1. Different parent/diff/msg for same commit
# 2. diff/msg file existence
# 3. Multiple parents
# 4. List of commits same as core/{pid}-{vid}b/commits.log
for project in projects:
    print(f'Working on {project}')

    diff_dict = {}
    parent_dict = {}
    msg_dict = {}

    commit_set = set()
    commit_log_set = set()

    # Check only target projects
    for dir_name in os.listdir(diff_dir):
        if not dir_name.startswith(project):
            continue

        print(f'Inspecting {dir_name}')
        project_dir = os.path.join(diff_dir, dir_name)

        diff_parent = False
        diff_not_exist = False

        # Get the list of current commits for target version
        for commit in os.listdir(project_dir):
            commit_set.add(commit)
            commit_dir = os.path.join(project_dir, commit)

            diff_exist = False
            msg_exist = False
            new_commit = False
            num_parent = 0

            # Files for the commit
            for filename in os.listdir(commit_dir):
                # Diff data
                if filename.endswith('.csv'):
                    diff_exist = True
                    num_parent += 1

                    # New current commit
                    if commit not in diff_dict:
                        new_commit = True
                        parent_dict[commit] = []
                    
                    if new_commit:
                        parent_dict[commit].append(filename)
                        diff_dict[(commit, filename)] = pd.read_csv(os.path.join(commit_dir, filename))
                    
                    else:
                        # Commit already insepected, but new parent
                        if filename not in parent_dict[commit]:
                            print(f'Different parent for {commit}...{filename}')
                            parent_dict[commit].append(filename)
                            diff_dict[(commit, filename)] = pd.read_csv(os.path.join(commit_dir, filename))
                        
                        # Two parents have different diff
                        elif not diff_dict[(commit, filename)].equals(pd.read_csv(os.path.join(commit_dir, filename))):
                            print(f'Different diff for {commit}...{filename}')

                elif filename == 'message.txt':
                    msg_exist = True
                    with open(os.path.join(commit_dir, filename), 'r') as file:
                        content = file.read()

                    # New current commit
                    if commit not in msg_dict:
                        new_commit = True
                        msg_dict[commit] = content
                    
                    # Different message
                    elif msg_dict[commit] != content:
                        print(f'Different message for {commit}...{filename}')
                
            if not diff_exist:
                print(f'No diff data for {commit}')

            if not msg_exist:
                print(f'No message for {commit}')
                
            if num_parent > 1:
                print(f'Multiple parents for {commit}:{num_parent}')
        
        with open(os.path.join(core_dir, dir_name, 'commits.log'), 'r') as file:
            for line in file.readlines():
                commit_log_set.add(line.strip())
        
        if commit_set != commit_log_set:
            print(f'List of commits unconsistency for {dir_name}')