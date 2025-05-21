import sys, logging, subprocess, os, shutil
sys.path.append('/root/workspace/data_collector/tool/')
import track_history, parse_gumtree, greedy_id, gen_intvl, encode, encode_bug, vote, bisection

sys.path.append('/root/workspace/lib/')
from experiment_utils import load_BIC_GT

if __name__ == "__main__":
    
    # Consider only manually selected BIC
    all_GT = load_BIC_GT("/root/workspace/data/Defects4J/BIC_dataset")
    GT = all_GT[all_GT['provenance'].str.contains("Manual", na=False)]
    
    """
    #GT = all_GT[~all_GT['provenance'].str.contains("Manual", na=False)]
    #update = False
    skip = True
    """

    for _, row in GT.iterrows():
        pid, vid = row.pid, row.vid
        """
        pid, vid = 'Closure', '62'
        if pid == 'Closure' and vid == '60':
            skip = False
            #continue
        
        if skip:
            continue
        """

        # Checkout Defects4J project
        shutil.rmtree(f'/tmp/{pid}-{vid}b/', ignore_errors=False)
        p = subprocess.Popen(['sh', '/root/workspace/lib/checkout.sh', pid, vid], \
            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out_txt, err_txt = p.communicate()

        if p.returncode != 0:
            print(f'[ERROR] Checkout failed {pid}-{vid}b')
            continue
        
        # Change working directory to target Defects4J project
        try:
            os.chdir(f'/tmp/{pid}-{vid}b/')
        except:
            print(f'[ERROR] Moving directory failed {pid}-{vid}b')
            continue

        #track_history.main(pid, vid)
        #parse_gumtree.main(pid, vid)
        #greedy_id.main(pid, vid)
        #gen_intvl.main(pid, vid)
        #encode.main(pid, vid)

        # Executing encode_bug right after encode raises error
        # For practical use, execute encode first
        #encode_bug.main(pid, vid)
        vote.main(pid, vid)
        bisection.main(pid, vid)

        #break