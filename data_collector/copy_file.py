import subprocess, os

if __name__ == "__main__":
    #a2715a5,src/com/google/javascript/jscomp/Tracer.java,src/com/google/javascript/jscomp/Tracer.java,U
    #a2715a5,src/com/google/javascript/rhino/TokenStream.java,src/com/google/javascript/rhino/TokenStream.java,U
    pid = 'Csv'
    vid = '12'
    commit_hash = '75f39a8'
    before_src_path = 'src/main/java/org/apache/commons/csv/CSVRecord.java'
    after_src_path = 'src/main/java/org/apache/commons/csv/CSVRecord.java'

    # Checkout Defects4J project
    p = subprocess.Popen(f'sh /root/workspace/lib/checkout.sh {pid} {vid}', \
        shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out_txt, err_txt = p.communicate()

    if p.returncode != 0:
        print('Checkout failed')
        sys.exit(1)
    
    # Change working directory
    try:
        os.chdir(f'/tmp/{pid}-{vid}b/')
    except:
        print('Could not move working directory')
        sys.exit(1)

    # Copy current version of file
    p = subprocess.Popen(['git', 'show', f'{commit_hash}:{after_src_path}'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    code_txt, err_txt = p.communicate()

    # Error raised while copying file
    if p.returncode != 0:
        print(f'No such file on {commit_hash}')
    
    else:
        with open('/root/workspace/data_collector/after.java', 'wb') as file:
            file.write(code_txt)
    
    # Copy before version of file
    p = subprocess.Popen(['git', 'show', f'{commit_hash}~1:{before_src_path}'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    code_txt, err_txt = p.communicate()

    # Error raised while copying file
    if p.returncode != 0:
        print(f'No such file on {commit_hash}~1')
    
    else:
        with open('/root/workspace/data_collector/before.java', 'wb') as file:
            file.write(code_txt)