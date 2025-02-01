import subprocess, os

if __name__ == "__main__":
    pid = 'Closure'
    vid = '30'
    commit_hash = 'e33e925'
    src_path = 'src/com/google/javascript/jscomp/DefinitionsRemover.java'

    # Checkout Defects4J project
    p = subprocess.Popen(f'sh /root/workspace/data_collector/tool/checkout.sh {pid} {vid}', \
        shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out_txt, err_txt = p.communicate()

    if p.returncode != 0:
        exit(1)
    
    # Change working directory
    try:
        os.chdir(f'/tmp/{pid}-{vid}b/')
    except:
        exit(1)

    p = subprocess.Popen(f'git show {commit_hash}:{src_path}', shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    code_txt, err_txt = p.communicate()

    # Error raised while copying file
    if p.returncode != 0:
        exit(1)
    
    else:
        #code_txt = code_txt.decode(encoding='utf-8', errors='ignore')

        with open('/root/workspace/data_collector/copy1.java', 'wb') as file:
            file.write(code_txt)