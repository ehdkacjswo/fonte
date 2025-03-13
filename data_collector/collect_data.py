import sys, logging
sys.path.append('/root/workspace/data_collector/tool/')
import track_history, parse_gumtree, greedy_id, gen_intvl, encode, vote, bisection

sys.path.append('/root/workspace/lib/')
from experiment_utils import load_BIC_GT

if __name__ == "__main__":
    all_GT = load_BIC_GT("/root/workspace/data/Defects4J/BIC_dataset")
    GT = all_GT[all_GT['provenance'].str.contains("Manual", na=False)]
    #update = False
    skip = True

    for _, row in GT.iterrows():
        pid, vid = row.pid, row.vid
        pid, vid = 'Cli', '29'

        #if pid == 'Jsoup' and vid == '47':
        #    skip = False
        
        #if skip:
        #    continue

        #track_history.main(pid, vid)
        #parse_gumtree.main(pid, vid)
        #greedy_id.main(pid, vid)
        #gen_intvl.main(pid, vid)
        #encode.main(pid, vid)
        vote.main(pid, vid)
        #bisection.main(pid, vid)

        break