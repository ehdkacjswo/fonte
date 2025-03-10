import regex as re
from spiral import ronin
from collections import Counter
from typing import Literal
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Java keywords + Boolean / Null literals
# (https://docs.oracle.com/javase/specs/jls/se8/html/jls-3.html#jls-3.9)
keyword_set = {'abstract', 'assert', 'boolean', 'break', 'byte', 'case', 'catch', 'char', 'class', \
    'const', 'continue', 'default', 'do', 'double', 'else', 'enum', 'extends', 'final', 'finally', 'float', \
    'for', 'if', 'goto', 'implements', 'import', 'instanceof', 'int', 'interface', 'long', 'native', 'new', \
    'package', 'private', 'protected', 'public', 'return', 'short', 'static', 'strictfp', 'super', 'switch', \
    'synchronized', 'this', 'throw', 'throws', 'transient', 'try', 'void', 'volatile', 'while', \
    'true', 'false', 'null'}

stopword_set = set(stopwords.words('english'))
stemmer = PorterStemmer()

# Class that encodes string while expanding the vocabulary
class Encoder():
    def __init__(self, vocab={}, id_vocab={}):
        self.vocab = vocab # {word : id}
        self.id_vocab = id_vocab 
        #self.stemmer = PorterStemmer()
    
    # Get id of given word
    # 
    def get_id(self, word, update_vocab, mode : Literal['id', 'code', 'text']):
        
        # Word exists in vocabulary
        if word in self.all_vocab:
            word_id = self.all_vocab[word]

            # 
            if mode =='id' and update_vocab and (word not in self.id_vocab):
                self.id_vocab[word] = word_id
        
        else:
            if update_vocab:
                word_id = len(self.all_vocab)
                self.all_vocab[word] = word_id

                if mode == 'id':
                    self.id_vocab[word] = word_id
            
            else:
                return None

    # Update vocab False > 존재하지 않는 토큰은 무시해야 한다 > 개별 Encoding 불가가
    # True > 존재하지 않는 토큰큰
    # Encode token
    def encode_token(self, token, is_id):
        
        if is_id:
            self.id_vocab.setdefault(token, len(self.vocab) + len(self.id_vocab))
            return self.id_vocab[token]
            if token in self.id_vocab:
                return self.id_vocab[token]
            
            else:
                self.id_vocab[token]


    # REMEBER!!!!!!! FILTERING BEFORE STEMMING!!!!!!
    # update_vocab : False > Find ID too
    def tokenize(self, text, update_vocab, mode : Literal['id', 'code', 'text']):
        token_id_list = list()

        # For the id, keep the full identifier
        # Maybe already in vocab
        if mode == 'id':
            if update_vocab:
                self.id_vocab.setdefault(text, len(self.vocab) + len(self.id_vocab))
                token_id_list = [self.id_vocab[text]]
            
            elif text in self.id_vocab:
                token_id_list = [self.id_vocab[text]]

        # Remove special characters & split
        # Keep _ and $ since they are available on id names (Handled on ronin)
        token_list = re.sub(r'[^A-Za-z0-9_$]', ' ', text).split()

        if mode == 'code': # Remove keywords from the code
            token_list = [token for token in token_list if token not in keyword_set]
        
        # Perform ronin split
        ronin_token_list = []
        for token in token_list:
            ronin_token_list += ronin.split(token)
        
        # Last filtering + 
        stem_token_list = []

        for token in ronin_token_list:
            if len(token) <= 1 or token.isdigit(): # Ignore digit and
                continue
            
            token = token.lower()
            if token in stopword_set:
                continue
            
            token = stemmer.stem(token)

            if len(token) > 1:
                stem_token_list.append(token)
            
        return token_list

    # Encode the input and list of used word index and count
    def encode(self, text, use_stopword=True, update_vocab=True):
        encode_res = []
        text = self.tokenize(text, use_stopword)

        for word, cnt in Counter(text).items():
            if word in self.vocab: # Word in vocab
                encode_res.append((self.vocab[word], cnt))
                
            elif update_vocab: # New word
                encode_res.append((len(self.vocab), cnt))
                self.vocab[word] = len(self.vocab)
        
        return encode_res

# Return the sum of two encoded vectors
def sum_encode(vec1, vec2):
    res_dict = dict()

    for id, cnt in vec1:
        res_dict[id] = cnt
    
    for id, cnt in vec2:
        res_dict[id] = res_dict.get(id, 0) + cnt
    
    return list(res_dict.items())