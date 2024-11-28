class Diff_commit: # Class containg diff data of commit
    class Diff_src: # Class containg diff data of source file
        def __init__(self):
            self.diff_dict = dict()
        
        def add_file_info(self, before_src_path, after_src_path):
            file_info_key = (before_src_path, after_src_path)
            if (file_info_key) not in self.diff_dict:
                self.diff_dict[file_info_key] = [dict(), dict()] # Addition, Deletion {line : content}
            
        def add_diff(self, before_src_path, after_src_path, line, content, adddel='add'):
            file_info_key = (before_src_path, after_src_path)
            dict_idx = 0 if adddel == 'add' else 1 # Select addition, deletion

            if line not in self.diff_dict[file_info_key][dict_idx]:
                self.diff_dict[file_info_key][dict_idx][line] = content

            if line in self.diff_dict[file_info_key][dict_idx]:
                if self.diff_dict[file_info_key][dict_idx][line] != content: # Same line, but different content
                    with open('/root/workspace/eror.txt', 'a') as file:
                        file.write(f'Different diff content for same line num: {before_src_path},{after_src_path}, {line}\n')
        
        def self_print(self):
            for (before_src_path, after_src_path) in self.diff_dict.keys():
                print(f'Before path : {before_src_path}, After path : {after_src_path}')

                addition = self.diff_dict[(before_src_path, after_src_path)][0]
                deletion = self.diff_dict[(before_src_path, after_src_path)][1]

                print('Addition)')
                for line, content in addition.items():
                    print(line, content)
                    
                print('Deletion)')
                for line, content in deletion.items():
                    print(line, content)

    def __init__(self):
        self.diff_dict = dict()

    def add_commit(self, commit_hash, src_path):
        if commit_hash not in self.diff_dict:
            self.diff_dict[commit_hash] = dict()

        if src_path not in self.diff_dict[commit_hash]:
            self.diff_dict[commit_hash][src_path] = self.Diff_src()
    
    def add_file_info(self, commit_hash, src_path, before_src_path, after_src_path):
        self.diff_dict[commit_hash][src_path].add_file_info(before_src_path, after_src_path)
    
    def add_diff(self, commit_hash, src_path, before_src_path, after_src_path, line, content, adddel='add'):
        self.diff_dict[commit_hash][src_path].add_diff(before_src_path, after_src_path, line, content, adddel)

    def self_print(self):
        for commit_hash in self.diff_dict.keys():
            print(f'Commit : {commit_hash}')

            for src_path in self.diff_dict[commit_hash].keys():
                print(f'src_path : {src_path}')
                self.diff_dict[commit_hash][src_path].self_print()

class Diff_commit_encode: # Class containg encoded diff data of commit
    class Diff_src: # Class containg diff data of source file
        def __init__(self):
            self.diff_dict = dict()
        
        def add_file_info(self, before_src_path, after_src_path):
            file_info_key = (before_src_path, after_src_path)
            if (file_info_key) not in self.diff_dict:
                self.diff_dict[file_info_key] = [dict(), dict()] # Addition, Deletion {line : content}
            
        def add_diff(self, before_src_path, after_src_path, line, content, adddel='add'):
            file_info_key = (before_src_path, after_src_path)
            dict_idx = 0 if adddel == 'add' else 1 # Select addition, deletion

            if line not in self.dict[file_info_key][dict_idx]:
                self.diff_dict[file_info_key][dict_idx][line] = content

            if line in self.diff_dict[file_info_key][dict_idx]:
                if self.diff_dict[file_info_key][dict_idx][line] != content: # Same line, but different content
                    with open('/root/workspace/eror.txt', 'a') as file:
                        file.write(f'Different diff content for same line num: {before_src_path},{after_src_path}, {line}\n')
        
        def self_print(self):
            for (before_src_path, after_src_path) in self.diff_dict.keys():
                print(f'Before path : {before_src_path}, After path : {after_src_path}')

                addition = self.diff_dict[(before_src_path, after_src_path)][0]
                deletion = self.diff_dict[(before_src_path, after_src_path)][1]

                print('Addition)')
                for line, content in addition.items():
                    print(line, content)
                    
                print('Deletion)')
                for line, content in deletion.items():
                    print(line, content)

    def __init__(self):
        self.diff_dict = dict()

    def add_commit(self, commit, src_path):
        if commit not in self.diff_dict:
            self.diff_dict[commit[:7]] = dict()

        if src_path not in self.diff_dict[commit[:7]]:
            self.diff_dict[commit[:7]][src_path] = self.Diff_src()
    
    def add_file_info(self, commit, src_path, before_src_path, after_src_path):
        self.diff_dict[commit[:7]][src_path].add_file_info(before_src_path, after_src_path)
    
    def add_diff(self, commit, src_path, before_src_path, after_src_path, line, content, adddel='add'):
        self.diff_dict[commit[:7]][src_path].add_diff(before_src_path, after_src_path, line, content, adddel)

    def self_print(self):
        for commit in self.diff_dict.keys():
            print(f'Commit : {commit}')

            for src_path in self.diff_dict[commit].keys():
                print(f'src_path : {src_path}')
                self.diff_dict[commit][src_path].self_print()