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
from spiral import ronin
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer


import pandas as pd
from interval import inf

sys.path.append('/root/workspace/data_collector/lib/')
from utils import *

CORE_DATA_DIR = '/root/workspace/data/Defects4J/core/'
DIFF_DATA_DIR = '/root/workspace/data/Defects4J/diff/'

keyword_set = {'abstract', 'assert', 'boolean', 'break', 'byte', 'case', 'catch', 'char', 'class', \
    'const', 'continue', 'default', 'do', 'double', 'else', 'enum', 'extends', 'final', 'finally', 'float', \
    'for', 'if', 'goto', 'implements', 'import', 'instanceof', 'int', 'interface', 'long', 'native', 'new', \
    'package', 'private', 'protected', 'public', 'return', 'short', 'static', 'strictfp', 'super', 'switch', \
    'synchronized', 'this', 'throw', 'throws', 'transient', 'try', 'void', 'volatile', 'while', \
    'true', 'false', 'null'}
stopword_set = set(stopwords.words('english'))
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

# Assume given code is syntatically
def extract_identifiers(code_txt):
    length = len(code_txt)
    id_intvl, non_id_intvl = CustomInterval(), CustomInterval()
    
    ind = 0
    while ind < length:
        char = code_txt[ind]

        if char == '"' or char == "'":
            start, quote = ind, char
            ind += 1

            # Closing quote must not be preceded by '\'
            while ind < length:
                if code_txt[ind] == quote:
                    escape_cnt, tmp_ind = 0, ind - 1

                    while code_txt[tmp_ind] == '\\':
                        escape_cnt += 1
                        tmp_ind -= 1
                    
                    # Even though closing quote is preceded by '\', it's okay if it's already escaped.
                    if escape_cnt % 2 == 0:
                        break
                    
                ind += 1
            
            if ind < length:
                non_id_intvl |= CustomInterval(start, ind + 1)
                ind += 1
                continue
            
            else:
                #log('greedy_id', f'[ERROR] Character / String literal not closed {commit}:{src_path}\n{code_txt[start : ind + 1]}')
                return None, None

        # Line Comments (//...)
        if char == '/' and ind + 1 < length and code_txt[ind + 1] == '/':
            start = ind
            ind += 2

            while ind < length and code_txt[ind] != '\n':
                ind += 1
            
            non_id_intvl |= CustomInterval(start, ind - 1)
            print('Line comment) ', code_txt[start:ind])
            ind += 1
            continue

        # Block / Javadoc comment
        if char == '/' and ind + 1 < length and code_txt[ind + 1] == '*':
            start = ind
            ind += 2

            # Find closing string '*/'
            while ind + 1 < length and not (code_txt[ind] == '*' and code_txt[ind + 1] == '/'):
                ind += 1

            if ind + 1 < length: # Comment closed properly
                non_id_intvl |= CustomInterval(start, ind + 1)
                print('Line comment) ', code_txt[start:ind+2])
                ind += 2
                continue
            
            else: # 
                #print('greedy_id', f'[ERROR] Block / Javadoc comment not closed')
                return None, None

        # Handle Identifiers (어디까지 identifier로 생각해야되지지)
        if char.isalpha() or char == '_' or char == '$':
            start = ind

            while ind < length and (code_txt[ind].isalnum() or code_txt[ind] in "_$"):
                ind += 1

            if code_txt[start : ind] not in keyword_set:
                id_intvl |= CustomInterval(start, ind - 1)
                print('ID) ', code_txt[start:ind])

            continue
        
        ind += 1

    return id_intvl, non_id_intvl

if __name__ == "__main__":
    # Extract interval of char, str, line/block comment, Javadoc, annotation

    with open('/root/workspace/test/greedy_id/greedy.java', 'r') as file:
        code_txt = file.read()
    """non_id_intvl = CustomInterval()

    for match in non_id_regex.finditer(code_txt):
        non_id_intvl |= CustomInterval(match.start(), match.end() - 1)"""
    


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
    """id_intvl = CustomInterval()

    for match in id_regex.finditer(code_txt):
        sub_intvl = CustomInterval(match.start(), match.end() - 1)

        if (non_id_intvl & sub_intvl).is_empty():
            if match.group(0) in java_keywords:
                continue  # Ignore Java keywords
            
            id_intvl |= sub_intvl"""
    
    id_intvl, non_id_intvl = extract_identifiers(code_txt)
        
    #for sub_intvl in non_id_intvl:
    #    print('Non_Id)', code_txt[convert_ind(sub_intvl[0]) : convert_ind(sub_intvl[1])])
    
    #for sub_intvl in id_intvl:
    #    print('Id)', code_txt[convert_ind(sub_intvl[0]) : convert_ind(sub_intvl[1])])
    
    #stemmer = PorterStemmer()
    #for keyword in stopword_set:
    #    print(keyword, stemmer.stem(keyword))
    
