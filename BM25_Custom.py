import numpy as np
import math
from collections import Counter
from rank_bm25 import BM25Okapi

# BM25Okapi for encoded version
class BM25_Encode(BM25Okapi):
    def __init__(self, k1=1.5, b=0.75, epsilon=0.25):
        # BM25 init
        self.corpus_size = 0

        # BM25Okapi init
        self.k1 = k1
        self.b = b
        self.epsilon = epsilon

        # Additional
        self.nd = {} # {word : val}
        self.num_doc = 0
    
    # Encode tokenized document
    """def encode(self, document):
        res = [0] * self.vocab_size

        for word, cnt in Counter(document).items():
            try:
                res[self.vocab[word]] = cnt
            except:
                continue
        
        return res"""

    # Vectorize encoded features
    def vectorize_complex(self, features):
        vector_sum = np.zeros(len(self.nd))

        for feature in features:
            doc_len = 0
            feature_dict = dict()

            for word, freq in feature:
                feature_dict[word] = freq
                doc_len += freq

            vector = [self.idf[word] * (feature_dict.get(word, 0) * (self.k1 + 1) /
                (self.idf[word] + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl))) for word in self.nd.keys()]

            vector_sum = np.add(vector_sum, np.array(vector))
        return np.divide(vector_sum, len(features))
    
    # Add encoded document
    def add_document(self, document):
        self.corpus_size += 1

        for ind, cnt in document:
            self.nd[ind] = self.nd.get(ind, 0) + 1
            self.num_doc += cnt
    
    def _calc_idf(self):
        """
        Calculates frequencies of terms in documents and in corpus.
        This algorithm sets a floor on the idf values to eps * average_idf
        """
        # collect idf sum to calculate an average idf for epsilon value
        idf_sum = 0
        self.idf = dict()

        # collect words with negative idf to set them a special epsilon value.
        # idf can be negative if word is contained in more than half of documents
        negative_idfs = []
        len_idf = 0

        for word, freq in self.nd.items():
            # No count
            if freq == 0:
                self.idf[word] = 0
                continue

            len_idf += 1
            idf = math.log(self.corpus_size - freq + 0.5) - math.log(freq + 0.5)
            self.idf[word] = idf
            idf_sum += idf

            if idf < 0:
                negative_idfs.append(word)

        self.average_idf = idf_sum / len_idf

        eps = self.epsilon * self.average_idf
        for word in negative_idfs:
            self.idf[word] = eps
    
    # End initialziation process
    def init_end(self):
        self.avgdl = self.num_doc / self.corpus_size
        self._calc_idf()

"""
class BM25_Custom(BM25Okapi):
    def __init__(self, tokenizer=None, k1=1.5, b=0.75, epsilon=0.25):
        # BM25 init
        self.corpus_size = 0
        self.avgdl = 0
        self.doc_freqs = []
        self.idf = {}
        self.doc_len = []
        self.tokenizer = tokenizer

        # BM25Okapi init
        self.k1 = k1
        self.b = b
        self.epsilon = epsilon

        # Additional
        self.nd = {}
        self.num_doc = 0

    # Add encoded
    def add_document(self, document):
        self.doc_len.append(len(document))
        self.num_doc += len(document)

        frequencies = {}
        for word in document:
            if word not in frequencies:
                frequencies[word] = 0
            frequencies[word] += 1
        self.doc_freqs.append(frequencies)

        for word, freq in frequencies.items():
            try:
                self.nd[word]+=1
            except KeyError:
                self.nd[word] = 1

        self.corpus_size += 1
    
    # End initialziation process
    def init_end(self):
        self.avgdl = self.num_doc / self.corpus_size
        self.vocab = list(set(sum([list(doc.keys()) for doc in self.doc_freqs], [])))
        super()._calc_idf(self.nd)
"""