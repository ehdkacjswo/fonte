import sys, logging
sys.path.append('/root/workspace/data_collector/tool/')
import collect_diff, stage2, encode, get_feature, vote, bisection

sys.path.append('/root/workspace/lib/')
from experiment_utils import load_BIC_GT

if __name__ == "__main__":
    GT = load_BIC_GT("/root/workspace/data/Defects4J/BIC_dataset")

    for _, row in GT.iterrows():
        pid, vid = row.pid, row.vid
        #pid, vid = 'Closure', '10'
        
        if pid == 'Closure' and vid == '131':
            continue

        #collect_diff.main(pid, vid)
        stage2.main(pid, vid)
        encode.main(pid, vid)
        get_feature.main(pid, vid)    
        vote.main(pid, vid)
        bisection.main(pid, vid)

        #break