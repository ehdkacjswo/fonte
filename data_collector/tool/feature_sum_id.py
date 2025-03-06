import os, json, argparse, pickle, sys, itertools, math
import pandas as pd
from tqdm import tqdm

sys.path.append('/root/workspace/data_collector/lib/')
from encoder import *
from utils import *

DIFF_DATA_DIR = '/root/workspace/data/Defects4J/diff'
    
# Create new setting that doesn't distinguish types of identifiers
# Should be added to encoding step later...
def main(pid, vid):
    log('feature_sum_id', f'Working on {pid}_{vid}b')

    diff_data_dir = os.path.join(DIFF_DATA_DIR, f'{pid}-{vid}b')
    os.makedirs(diff_data_dir, exist_ok=True)

    start_time = time.time()

    # Load feature data
    with open(os.path.join(diff_data_dir, 'feature.pkl'), 'rb') as file:
        feature_dict = pickle.load(file)

    for stage2, stage2_dict in feature_dict.items():
        temp_dict = dict() # Dictionary with new settings

        for setting, setting_dict in stage2_dict.items():
            new_setting = dict(setting)

            # Consider only the identifier settings
            if new_setting.get('diff_type', None) == 'id': 
                new_setting = frozenset((new_setting | {'diff_type' : 'id_all'}).items())
                temp_dict[new_setting] = dict()

                for commit, commit_dict in setting_dict.items():
                    temp_dict[new_setting][commit] = dict()

                    for feature_type, feature in commit_dict.items():
                        # Sum up only the identifiers
                        if not (feature_type == 'msg' or feature_type.endswith('path')):
                            if '_' in feature_type: # all_sep contains prefix 'add_' or 'del_'
                                new_type = feature_type.split('_', 1)[0] + '_id'
                            else:
                                new_type = 'id'
                            
                            temp_dict[new_setting][commit][new_type] = sum_encode(temp_dict[new_setting][commit].get(new_type, []), feature)

                        # Non identifer features remain same
                        else:
                            temp_dict[new_setting][commit][feature_type] = feature
        
        # Add new settings
        for setting, setting_dict in temp_dict.items():
            stage2_dict[setting] = setting_dict
    
    with open(os.path.join(diff_data_dir, 'feature.pkl'), 'wb') as file:
        pickle.dump(feature_dict, file)
    
    end_time = time.time()
    log('feature_sum_id', f'{time_to_str(start_time, end_time)}')

    