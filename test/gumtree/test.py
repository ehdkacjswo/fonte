import sys

sys.path.append('/root/workspace/data_collector/lib/')
from gumtree import *

#print(gumtree_parse('test.java'))

with open('/root/workspace/tmp/test.java', 'rb') as file:
    filedata = file.read().decode(encoding='utf-8', errors='ignore')

filedata = filedata.splitlines(True)
for line, string in enumerate(filedata):
    for ind, a in enumerate(string):
        print(line, ind, a)
