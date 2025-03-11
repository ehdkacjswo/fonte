import regex as re
from spiral import ronin
from collections import Counter
from typing import Literal
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
# Same stemmer with ronin (https://github.com/casics/spiral/blob/master/spiral/ronin.py#L410)

# Java keywords + Boolean / Null literals
# (https://docs.oracle.com/javase/specs/jls/se8/html/jls-3.html#jls-3.9)
keyword_set = {'abstract', 'assert', 'boolean', 'break', 'byte', 'case', 'catch', 'char', 'class', \
    'const', 'continue', 'default', 'do', 'double', 'else', 'enum', 'extends', 'final', 'finally', 'float', \
    'for', 'if', 'goto', 'implements', 'import', 'instanceof', 'int', 'interface', 'long', 'native', 'new', \
    'package', 'private', 'protected', 'public', 'return', 'short', 'static', 'strictfp', 'super', 'switch', \
    'synchronized', 'this', 'throw', 'throws', 'transient', 'try', 'void', 'volatile', 'while', \
    'true', 'false', 'null'}

stopword_set = set(stopwords.words('english'))
stemmer = SnowballStemmer('english')

# Stemming from ronin
# (https://github.com/casics/spiral/blob/master/spiral/ronin.py#L672)
def stem(token):
    # The NLTK stemmer sometimes gives weird results, and the particular
    # case of weird technical terms ending in 's' has been most troublesome.
    if len(token) > 1 and token[-1:] == 's':
        return token[:-1]
    else:
        return stemmer.stem(token)

# Class that encodes string while expanding the vocabulary
class Encoder():
    def __init__(self, vocab={}, id_vocab={}):
        self.non_id_vocab = vocab # {word : id}
        self.id_vocab = id_vocab 

    # Update vocab False > 존재하지 않는 토큰은 무시해야 한다 > 개별 Encoding 불가가
    # True > 존재하지 않는 토큰큰
    # Encode token
    def encode_tokens(self, token_list, update_vocab, is_id):
        token_id_list = list()

        if is_id:
            for token in token_list:
                token_id = None

                # Identifier on id vocabulary
                if token in self.id_vocab: 
                    token_id = self.id_vocab[token]

                # Identifier on non_id vocabulary
                elif token in self.non_id_vocab: 
                    # Do I have to consider update_vocab?
                    # Keeping it in non_id is ok?
                    token_id = self.non_id_vocab[token]
                    self.id_vocab[token] = token_id
                
                # Identifier not on vocabulary, but update it
                elif update_vocab: 
                    token_id = len(self.id_vocab) + len(self.non_id_vocab)
                    self.id_vocab[token] = token_id
                
                if token_id is not None:
                    token_id_list.append(token_id)
        
        else:
            for token in token_list:
                token_id = None

                # Token on non_id vocabulary
                if token in self.non_id_vocab: 
                    token_id = self.non_id_vocab[token]
                
                # Token not on vocabulary, but update it
                elif update_vocab: 
                    token_id = len(self.id_vocab) + len(self.non_id_vocab)
                    self.non_id_vocab[token] = token_id
                
                if token_id is not None:
                    token_id_list.append(token_id)
            
        return token_id_list

    # Encode the list of texts based on their type
    def encode(self, text_list, update_vocab, mode : Literal['id', 'code', 'text']):

        # For the id, keep the full identifier
        if mode == 'id':
            id_list = self.encode_tokens(text_list, update_vocab, is_id=True)

        else:
            id_list = list()

        # Remove special characters & split
        # Keep _ and $ since they are available on identifiers (To avoid removing keywords in identifires)
        # 
        token_list = list()

        for text in text_list:
            token_list += re.sub(r'[^A-Za-z0-9_$]', ' ', text).split()

        # Remove keywords from the code
        if mode == 'code': 
            token_list = [token for token in token_list if token not in keyword_set]
        
        # Ronin split
        ronin_token_list = []
        for token in token_list:
            ronin_token_list += ronin.split(token)
        
        # Filtering + Stemming
        final_token_list = []

        for token in ronin_token_list:
            if len(token) <= 1 or token.isdigit(): # Ignore digit and short characters
                continue
            
            token = token.lower()
            if token in stopword_set: # Ignore stopwords
                continue
            
            token = stem(token)

            if len(token) > 1:
                final_token_list.append(token)
        
        final_token_list = self.encode_tokens(final_token_list, update_vocab, is_id=False)
        #print(id_list)
        #print(final_token_list)
        
        # Find identifiers or not
        #for token in stem_token_list:
        return Counter(id_list), Counter(final_token_list)

# Return the sum of two encoded vectors
def sum_encode(vec1, vec2):
    res_dict = dict()

    for id, cnt in vec1:
        res_dict[id] = cnt
    
    for id, cnt in vec2:
        res_dict[id] = res_dict.get(id, 0) + cnt
    
    return list(res_dict.items())