from scipy import sparse
import pandas as pd
from sbfl.base import SBFL
import os
from intervaltree import Interval, IntervalTree

if __name__ == "__main__":
    a = IntervalTree([Interval(0, 1)])
    b = IntervalTree([Interval(0, 1)])
    print(Interval(0, 0.1) in a)
    print(a.overlaps(-1, 0.1))
    #print(os.path.getsize("/root/workspace/data/Defects4J/diff/Time-7b/diff.pkl"))