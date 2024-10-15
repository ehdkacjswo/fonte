import os, re

core_dir = '/root/workspace/data/Defects4J/core/'
data_dict = {}

for dir_name in os.listdir(core_dir):
    dir_path = os.path.join(core_dir, dir_name)

    # checking if it is a file
    if os.path.isdir(dir_path):
        match = re.fullmatch(r"(\w+)-(\d+)b", dir_name)
        if match:
            pid = match.group(1)
            vid = match.group(2)

            if pid not in data_dict:
                data_dict[pid] = {}


        else:
            print('DFSDF {}'.format(dir_name))