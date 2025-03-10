from scipy import sparse
import pandas as pd
from sbfl.base import SBFL
import os, pickle, math
from intervaltree import Interval, IntervalTree
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

DIFF_DATA_DIR = '/root/workspace/data/Defects4J/diff'
RESULT_DATA_DIR = '/root/workspace/data/Defects4J/result'

if __name__ == "__main__":
    a = '132134'

    print(a[1:3])
    print(type(a[1:3]))
    print(math.floor(-0.5))

    stopword_list = stopwords.words('english')
    stemmer = PorterStemmer()

    for stopword in stopword_list:
        a = stemmer.stem(stopword)

        if stopword != a:
            print('[ERROR]', stopword, a)
        else:
            print('[INFO]', stopword, a)
        
        