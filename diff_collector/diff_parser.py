import os, re, sys, pickle
from collections.abc import Iterable
from spiral import ronin

# Function to extract diff information from Java files


class Diff():
    def __init__(self):
        # Modified contents {(file_path, num_line) : content}
        """self.deletion = {}
        self.addition = {}"""

        self.rows 

        # self.init()
    
    # Parse the diff
    def parse_diff(diff):
        # [(addition, old_file_path, new_file_path, modified line number, modified contend)]
        rows = []

        for diff_item in diff:
            # Ignore non-java files
            if not diff_item.a_path.endswith('.java') and not diff_item.b_path.endswith('.java'):
                continue
                
            # Get the raw diff for this file
            raw_diff = diff_item.diff.decode('utf-8')  # Convert bytes to string
                
            # Regex to find line numbers and modifications
            diff_block_regex = r'@@ -(\d+),?\d* \+(\d+),?\d* @@'
            lines = raw_diff.splitlines()

            # Track line number on old/new file
            cur_old_line = None
            cur_new_line = None

            for line in lines:
                # Identify start of a diff block with line numbers
                match = re.match(diff_block_regex, line)
                if match:
                    cur_old_line = int(match.group(1))  # Start line for original (deleted) lines
                    cur_new_line = int(match.group(2))  # Start line for new (added) lines
                    continue
                    
                # Deleted line (--- indicates old file path)
                if line.startswith('-') and not line.startswith('---'):
                    cur_old_line += 1
                    self.deletion[(diff_item.a_path, cur_old_line)] = line[1:].strip()
                    
                # Added line (+++ indicates new file path)
                elif line.startswith('+') and not line.startswith('+++'):
                    cur_new_line += 1
                    self.addition[(diff_item.b_path, cur_new_line)] = line[1:].strip()         
                    
                # Unchanged line
                elif not line.startswith('+') and not line.startswith('-'):
                    cur_old_line += 1
                    cur_new_line += 1

    # Initialization for new change block
    """def init(self):
        # Current stage of parsing
        # 1 : Old file, 2 : New file, 3 : Line info, 4 : Content info, 5 : Skip (Non-java file)
        self.stage = 1

        # Path of modified file
        self.old_path = None
        self.new_path = None

        # Starting line of diff content
        self.start_line_old = None
        self.start_line_new = None

        # Current line of diff content
        self.cur_line_old = None
        self.cur_line_new = None


        # Number of lines modified
        self.num_line_old = None
        self.num_line_new = None"""

    """# Parse the given diff
    def parse_diff(self, diff):
        # String is given
        if isinstance(diff, str):
            try:
                with open(diff) as file:
                    diff_lines = file.readlines()
            except:
                diff_lines = diff.splitlines()

        # Iterable object is given
        elif isinstance(diff, Iterable):
            diff_lines = diff
        
        # Unexpected type of object is given
        else:
            raise Exception('Unexpected type of input for Diff : {}'.format(type(diff)))
            return

        # Start parsing
        for line in diff_lines:
            # New change block starts
            if line.startswith('diff --git'):
                self.init()

            else:
                self.add_line(line)"""

    """
    Format of "git --diff old_commit...new_commit"
    git --diff a/ b/ 
    [deleted, new] file mode 
    index ..
    --- [a/file_path or /dev/null]
    +++ [b/file_path or /dev/null]
    [- {Deleted content} or {Unchanged content} or + {Added content}]
    """
    """def add_line(self, line):
        # Skip the non-java file
        if self.stage == 5:
            return
            
        # File path on older commit
        if line.startswith('--- '):
            assert(self.stage == 1)
            self.stage = 2

            if line[4:6] == 'a/':
                self.old_path = line[5:]
            else:
                self.old_path = line[4:]
            
        # File path on newer commit
        elif line.startswith('+++ '):
            assert(self.stage == 2)
            self.stage = 3

            if line[4:6] == 'b/':
                self.new_path = line[5:]
            else:
                self.new_path = line[4:]
            
            # Check if the target file is java
            # If it's not, skip the rest of the changes
            if not self.old_path.endswith('.java') or not self.new_path.endswith('.java'):
                self.stage = 5
            
        # Line info of files
        elif line.startswith('@'):
            assert(self.stage == 3 or self.stage == 4)
            self.stage = 4
            self.start_line_old = None
            self.start_line_new = None

            for line_info in re.split(r'(?=[+-,])', re.sub(r'[@\s]', '', line)):
                line_info = re.sub(r'[,\s]+', '', line_info)
            
                if self.start_line_old is None and line_info.startswith('-'):
                    self.start_line_old = int(line_info[1:])
                    self.cur_line_old = self.start_line_old
            
                elif self.start_line_new is None and line_info.startswith('+'):
                    self.start_line_new = int(line_info[1:])
                    self.cur_line_new = self.start_line_new
            
        # Deleted line on old version
        elif line.startswith('-'):
            assert(self.stage == 4)
            assert((self.old_path, self.cur_line_old) not in self.deletion)
            
            self.deletion[(self.old_path, self.cur_line_old)] = line[1:]
            self.cur_line_old = self.cur_line_old + 1
            
        # Added line on new version
        elif line.startswith('+'):
            assert(self.stage == 4)
            assert((self.new_path, self.cur_line_new) not in self.addition)

            self.addition[(self.new_path, self.cur_line_new)] = line[1:]
            self.cur_line_new = self.cur_line_new + 1
            
        # Maintained line
        elif self.stage == 4:
            if self.cur_line_old is not None:
                self.cur_line_old = self.cur_line_old + 1

            if self.cur_line_new is not None:
                self.cur_line_new = self.cur_line_new + 1"""
    
    def save(self, save_dir):
        os.makedirs(save_dir, exist_ok=True)

        with open(os.path.join(save_dir, 'deletion.pkl'), 'wb') as f:
            pickle.dump(self.deletion, f)
        
        with open(os.path.join(save_dir, 'addition.pkl'), 'wb') as f:
            pickle.dump(self.addition, f)
        
        # Check if dumping is done properly
        try:
            with open(os.path.join(save_dir, 'deletion.pkl'), 'rb') as f:
                assert(pickle.load(f) == self.deletion)
        
            with open(os.path.join(save_dir, 'addition.pkl'), 'rb') as f:
                assert(pickle.load(f) == self.addition)
        
        except:
            with open('/root/workspace/data/Defects4J/diff/error.txt', 'a') as file:
                file.write(save_dir)
            
