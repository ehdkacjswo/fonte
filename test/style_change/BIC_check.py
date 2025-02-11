import subprocess, os, sys, re

sys.path.append('/root/workspace/lib/')
from experiment_utils import load_BIC_GT

if __name__ == "__main__":
    GT = load_BIC_GT("/root/workspace/data/Defects4J/BIC_dataset")

    for _, row in GT.iterrows():
        pid, vid, BIC = row.pid, row.vid, row.commit
        print(f'Working on {pid}-{vid}b : {BIC}')

        #pid = 'Closure'
        #vid = '131'
        #BIC = 'a2715a5'

        # Checkout Defects4J project
        p = subprocess.Popen(f'sh /root/workspace/lib/checkout.sh {pid} {vid}', \
            shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out_txt, err_txt = p.communicate()

        if p.returncode != 0:
            print('[ERROR] Checkout failed')
            continue
        
        # Change working directory
        try:
            os.chdir(f'/tmp/{pid}-{vid}b/')
        except:
            log('[ERROR] Moving directory failed')
            continue
            #sys.exit(1)

        # Run git diff
        p = subprocess.Popen(f'git diff {BIC}^ {BIC} -M -C', \
            shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out_txt, err_txt = p.communicate()

        if p.returncode != 0:
            print('[ERROR] Git diff failed')
            sys.exit(p.returncode)
        
        # Find changed files
        pattern = r"diff --git a/([^\s]+) b/([^\s]+)"
        matches = re.findall(pattern, out_txt.decode(encoding='utf-8', errors='ignore'))

        tot, style_true, style_false = 0, 0, 0

        for before_src_path, after_src_path in matches:
            tot += 1

            # Copy after source
            p = subprocess.Popen(f'git show {BIC}:{after_src_path}', shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            after_code, err_txt = p.communicate()

            if p.returncode != 0:
                print(f'[ERROR] After source copy failed) {BIC}:{after_src_path}')
                continue

            after_code = after_code.decode(encoding='utf-8', errors='ignore')
            with open('/root/workspace/tmp/after.java', 'w') as file:
                file.write(after_code)

            # Copy before source
            p = subprocess.Popen(f'git show {BIC}~1:{before_src_path}', shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            before_code, err_txt = p.communicate()

            if p.returncode != 0:
                print(f'[ERROR] Before source copy failed) {BIC}~1:{before_src_path}')
                continue

            before_code = before_code.decode(encoding='utf-8', errors='ignore')
            with open('/root/workspace/tmp/before.java', 'w') as file:
                file.write(before_code)
            
            # Isomorphic testing
            p = subprocess.Popen('docker run --rm -v /home/coinse/doam/fonte/tmp:/diff gumtree isotest \
                -g java-jdtnc before.java after.java', shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            is_style_change, err_txt = p.communicate()

            if p.returncode != 0:
                print(f'[ERROR] Isomorphic testing failed) {BIC} {before_src_path} {after_src_path}')
                continue
            
            is_style_change = is_style_change.decode(encoding='utf-8', errors='ignore')

            if is_style_change == 'true':
                style_true += 1
            
            elif is_style_change == 'false':
                print(f"{BIC} {before_src_path} {after_src_path} {is_style_change}")
                style_false += 1
            
        print(tot, style_true, style_false)
