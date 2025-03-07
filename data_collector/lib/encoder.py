import regex as re
from spiral import ronin
from collections import Counter
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Class that encodes string while expanding the vocabulary
class Encoder():
    stopword_list = stopwords.words('english')
    keyword_list = ['abstract', 'assert', 'boolean', 'break', 'byte', 'case', 'catch', 'char', 'class', \
        'const', 'continue', 'default', 'do', 'double', 'else', 'enum', 'extends', 'final', 'finally', 'float', \
        'for', 'goto', 'if', 'implements', 'import', 'instanceof', 'int', 'interface', 'long', 'native', 'new', \
        'package', 'private', 'protected', 'public', 'return', 'short', 'static', 'strictfp', 'super', 'switch', \
        'synchronized', 'this', 'throw', 'throws', 'transient', 'try', 'void', 'volatile', 'while']

    def __init__(self, vocab={}):
        self.vocab = vocab # {word : id}
        self.stemmer = PorterStemmer()
    
    def tokenize(self, text, use_stopword=True):
        # Remove characters except alphabets and numbers
        text = re.sub(r'[^A-Za-z0-9]', ' ', text) 

        token_list = ronin.split(text) # Split the text
        #print(token_list)
        token_list = [self.stemmer.stem(token.lower()) for token in token_list] # Apply lowercase
        #token_list = [token.lower() for token in token_list] # Apply lowercase
        #print(token_list)

        # Remove single character, numbers and stopwords
        if use_stopword:
            token_list = [token for token in token_list if \
                (len(token) > 1 and \
                not token.isdigit() and \
                token not in Encoder.stopword_list and \
                token not in Encoder.keyword_list)]
            
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