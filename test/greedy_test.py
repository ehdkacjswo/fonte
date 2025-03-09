import io
import json
import logging
import os
import re
import argparse
import subprocess
import sys
import pickle
import tempfile
import time
import math

import pandas as pd
from interval import inf

sys.path.append('/root/workspace/data_collector/lib/')
from gumtree import *
from utils import *
from encoder import java_keywords

CORE_DATA_DIR = '/root/workspace/data/Defects4J/core/'
DIFF_DATA_DIR = '/root/workspace/data/Defects4J/diff/'

# String : " + (Escape sequence + Any string) + "
# (https://docs.oracle.com/javase/specs/jls/se8/html/jls-3.html#jls-3.10.5)

# Character : ' + (Escape sequence | One character) + '
# (https://docs.oracle.com/javase/specs/jls/se8/html/jls-3.html#jls-3.10.4)

# Line comments : // + (Ends with newline)
# Block comments/Javadoc : /* + (Any string) + */
# (https://docs.oracle.com/javase/specs/jls/se8/html/jls-3.html#jls-3.7)

# Annotation : @ + (Possible whitespace) + Identifier name
# (https://docs.oracle.com/javase/specs/jls/se8/html/jls-9.html#jls-9.7)

non_id_regex = re.compile(r'''
    ("(?:\\.|[^"\\])*")            # Double-quoted string
    | ('(?:\\.|[^'\\])')           # Single character literals
    | (//.*?$)                     # Line comment
    | (/\*[\s\S]*?\*/)             # Block comment / Javadoc
    | (@\s*[A-Za-z_$][A-Za-z0-9_$]*) # Annotation
''', re.MULTILINE | re.VERBOSE)

# Identifier : Alphabet + Decimal number + _ + $ (Number not allowed at the start)
# (https://docs.oracle.com/javase/specs/jls/se8/html/jls-3.html#jls-3.8)

id_regex = re.compile(r'(?<!\w)[A-Za-z_$][A-Za-z0-9_$]*(?!\w)')

def convert_ind(ind):
    if ind == -inf:
        return 0
    if ind == inf:
        return len(code_txt)
    
    return math.floor(ind) + 1

if __name__ == "__main__":
    # Extract interval of char, str, line/block comment, Javadoc, annotation

    with open('/root/workspace/test/greedy.java', 'r') as file:
        code_txt = file.read()
    non_id_intvl = CustomInterval()

    for match in non_id_regex.finditer(code_txt):
        non_id_intvl |= CustomInterval(match.start(), match.end() - 1)
    


    # Extract annotations
    """annotation_regex = re.compile(r'@\s*([A-Za-z_$][A-Za-z0-9_$]*)')

    annotations = set()
    for match in annotation_regex.finditer(java_code):
        start, end = match.start(), match.end()

        if any(s <= start < e for s, e in non_identifier_ranges):
            continue  # Ignore if inside comments/strings

        annotation = match.group(1)  
        annotations.add(annotation)"""

    # Extract all potential identifiers
    id_intvl = CustomInterval()

    for match in id_regex.finditer(code_txt):
        sub_intvl = CustomInterval(match.start(), match.end() - 1)

        if (non_id_intvl & sub_intvl).is_empty():
            if match.group(0) in java_keywords:
                continue  # Ignore Java keywords
            
            id_intvl |= sub_intvl
        
    for sub_intvl in non_id_intvl:
        print('Non_Id)', code_txt[convert_ind(sub_intvl[0]) : convert_ind(sub_intvl[1])])
    
    for sub_intvl in id_intvl:
        print('Id)', code_txt[convert_ind(sub_intvl[0]) : convert_ind(sub_intvl[1])])