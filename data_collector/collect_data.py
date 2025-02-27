import sys, logging
sys.path.append('/root/workspace/data_collector/tool/')
import track_history, stage2, encode, get_feature, vote, bisection, parse_gumtree

sys.path.append('/root/workspace/lib/')
from experiment_utils import load_BIC_GT

if __name__ == "__main__":
    GT = load_BIC_GT("/root/workspace/data/Defects4J/BIC_dataset")
    update = False

    for _, row in GT.iterrows():
        pid, vid = row.pid, row.vid
        #pid, vid = 'Cli', '29'
        
        #if pid == 'Lang' and vid == '21':
        #    continue

        track_history.main(pid, vid)
        parse_gumtree.main(pid, vid)
        #stage2.main(pid, vid)
        #encode.main(pid, vid)
        #get_feature.main(pid, vid)    
        #vote.main(pid, vid)
        #bisection.main(pid, vid)

        #break