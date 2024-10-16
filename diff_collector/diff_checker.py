import os

# (commit_hash, pid) : [vid]
# pid_vid has commit_hash as past version
commits_dict = {}

core_dir = '/root/workspace/data/Defects4J/core/'
diff_dir = '/root/workspace/data/Defects4J/diff/'

# Check data for every commits exists
for dir_name in os.listdir(core_dir):
    core_bug_dir = os.path.join(core_dir, dir_name)

    if not os.path.isdir(core_bug_dir):
        continue

    diff_bug_dir = os.path.join(diff_dir, dir_name)

    if not os.path.isdir(diff_bug_dir):
        raise Exception('Diff data for {} not generated'.format(dir_name))
    
    # Check the commits
    
    # Load the list of past commits
    with open(os.path.join(core_bug_dir, 'commmits.log')) as file:
        commit_logs = file.readlines()
    
    for commit in commit_logs:
        diff_commit_dir = os.path.join(diff_bug_dir, commit)

        if not os.path.isdir(diff_commit_dir):
            raise Exception('Diff data for {}_{} not generated'.format(dir_name, commit))
        
        if not os.path.isfile(os.path.join(diff_commit_dir, 'addition.pkl'))

print('')