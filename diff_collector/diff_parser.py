import os, re
from collections.abc import Iterable
from spiral import ronin

class Diff():
    def __init__(self):
        # Modified contents {(file_path, num_line) : content}
        self.deletion = {}
        self.addition = {}

        self.init()

    # Initialization for new change block
    def init(self):
        # Current stage of parsing
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

        """
        # Number of lines modified
        self.num_line_old = None
        self.num_line_new = None
        """

    # Parse the given diff
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
                self.add_line(line)

    """
    Format of "git --diff old_commit...new_commit"
    git --diff a/ b/ 
    [deleted, new] file mode 
    index ..
    --- [a/file_path or /dev/null]
    +++ [b/file_path or /dev/null]
    [- {Deleted content} or {Unchanged content} or + {Added content}]
    """
    def add_line(self, line):
        # File path on older commit
        if line.startswith('--- '):
            assert(self.stage == 1)
            self.stage = 2

            if line[4:6] == 'a/':
                self.old_path = line[5:]
            else:
                self.old_path = line[4:]
            
            print('Old path : {}'.format(self.old_path))
            
        # File path on newer commit
        elif line.startswith('+++ '):
            assert(self.stage == 2)
            self.stage = 3

            if line[4:6] == 'b/':
                self.new_path = line[5:]
            else:
                self.new_path = line[4:]
            
            print('New path : {}'.format(self.new_path))
            
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
            
        # Deletion on old version
        elif line.startswith('-'):
            assert(self.stage == 4)
            assert((self.old_path, self.cur_line_old) not in self.deletion)
            
            self.deletion[(self.old_path, self.cur_line_old)] = line[1:]
            self.cur_line_old = self.cur_line_old + 1
            
        # Addition on new version
        elif line.startswith('+'):
            assert(self.stage == 4)
            assert((self.new_path, self.cur_line_new) not in self.addition)

            self.addition[(self.new_path, self.cur_line_new)] = line[1:]
            self.cur_line_new = self.cur_line_new + 1
            
        elif self.stage == 4:
            if self.cur_line_old is not None:
                self.cur_line_old = self.cur_line_old + 1

            if self.cur_line_new is not None:
                self.cur_line_new = self.cur_line_new + 1
            
        """match = re.fullmatch(r'diff --git a\/([^ ]+) b\/([^ ]+)', line)

        if match:
            match = match.groups()
            assert(len(match) == 2)

            self.old_path = match[0]
            self.new_path = match[1]

        else:
            return"""
