import subprocess, os

if __name__ == "__main__":
    #a2715a5,src/com/google/javascript/jscomp/Tracer.java,src/com/google/javascript/jscomp/Tracer.java,U
    #a2715a5,src/com/google/javascript/rhino/TokenStream.java,src/com/google/javascript/rhino/TokenStream.java,U
    pid = 'Closure'
    vid = '131'
    commit_hash = 'a2715a5'
    before_src_path = 'test/com/google/javascript/jscomp/deps/JsFunctionParserTest.java'
    after_src_path = 'test/com/google/javascript/jscomp/deps/JsFunctionParserTest.java '

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

    # Copy current version of file
    p = subprocess.Popen(f'git show {commit_hash}:{after_src_path}', shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    code_txt, err_txt = p.communicate()

    # Error raised while copying file
    if p.returncode != 0:
        exit(1)
    
    else:
        with open('/root/workspace/data_collector/after.java', 'wb') as file:
            file.write(code_txt)
    
    # Copy before version of file
    p = subprocess.Popen(f'git show {commit_hash}~1:{before_src_path}', shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    code_txt, err_txt = p.communicate()

    # Error raised while copying file
    if p.returncode != 0:
        exit(1)
    
    else:
        with open('/root/workspace/data_collector/before.java', 'wb') as file:
            file.write(code_txt)