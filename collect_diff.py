import io
import json
import logging
import os
import re

import pandas as pd
from github import Github, Auth, Commit

# Additional code for diff added Bug2Commit
proj_url_dict = {'Cli' : 'apache/commons-cli', \
'Closure' : 'google/closure-compiler', \
'Codec' : 'apache/commons-codec', \
'Compress' : 'apache/commons-compress', \
'Gson' : 'google/gson', \
'JacksonCore' : 'FasterXML/jackson-core', \
'Jsoup' : 'jhy/jsoup' , \
'Lang' : 'apache/commons-lang', \
'Math' : 'apache/commons-math', \
'Time' : 'JodaOrg/joda-time'}

def collect_commits(pid, vid, divide_commits=True):
    logger.info('Processing {0}'.format(project_name))

    project_dpath = os.path.join('../data', project_name)
    if divide_commits is True:
        output_dpath = os.path.join(project_dpath, 'hunks/')
    else:
        output_dpath = os.path.join(project_dpath, 'commits/')
    limit_ts = get_limit_ts(project_dpath)
    if not os.path.exists(output_dpath):
        os.makedirs(output_dpath)

    repo, ref_sha = load_repo(repo_url, ref)
    sha_cnt = dict()

    for commit, log, sha, date, metainfo in get_commits(repo, ref_sha, divide_commits=divide_commits):
        if sha not in sha_cnt:
            sha_cnt[sha] = 0
        else:
            sha_cnt[sha] = sha_cnt[sha] + 1

        cnt = sha_cnt[sha]
        if divide_commits is False and cnt == 1:
            raise RuntimeError('More than one commit with the same sha? Something is wrong')

        if valid_commit(date, limit_ts):
            with open(os.path.join(output_dpath, 'c_{0}_{1}.json'.format(sha, cnt)), 'w') as f:
                json.dump({'sha': sha, 'log': log, 'commit': commit, 'timestamp': date, 'metainfo': metainfo}, f)
    
if __name__ == "__main__":

    # Authenticate user and create github object
    print(os.getenv("GITHUB_API_KEY"))
    auth = Auth.Token(os.getenv("GITHUB_API_KEY"))
    g = Github(auth=auth)

    repo = g.get_repo('google/closure-compiler')

    # repo.compare(<Later commit sha>, <>)
    for file in repo.compare('3771c57be864851f5cf2fc4151f0bda628d089ad', '1d9ecb5328bbde5ee84dbfe2d74bbd318b89b6d7').files:
        print(file.filename)
        print(file.patch)

