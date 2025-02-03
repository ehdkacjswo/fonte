import subprocess, sys
from charset_normalizer import from_path

if __name__ == "__main__":
    # Encode files
    for filename in ['before', 'after']:
        encoded_code = from_path(f'/root/workspace/tmp/{filename}.java').best()
            
        if encoded_code is not None:
            with open(f'/root/workspace/tmp/{filename}.java', 'w', encoding='utf-8') as file:
                file.write(str(encoded_code))
        
        else:
            sys.exit(1)

    p = subprocess.Popen('docker run --rm -v /home/coinse/doam/fonte/tmp:/diff gumtree isotest \
      -g java-jdtnc before.java after.java', shell=True, stdout=subprocess.PIPE)

    code_txt, _ = p.communicate()

    # Error raised while copying file
    if p.returncode != 0:
        sys.exit(1)
    
    else:
        print(code_txt.decode(encoding='utf-8', errors='ignore'))