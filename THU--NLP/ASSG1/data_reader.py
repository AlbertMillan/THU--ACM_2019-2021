import numpy as np
import gensim
from gensim.corpora import WikiCorpus
import torch
from torch.utils.data import Dataset
import itertools

import os.path
import os
import sys


np.random.seed(12345)

class DataReader:
    NEGATIVE_TABLE_SIZE = 1e6
    
    def __init__(self, inFile, txtFile=None):
        
        self.negatives = []
        self.discards = []
        self.negpos = 0

        self.word2id = dict()
        self.id2word = dict()
        self.token_count = 0
        self.word_freq = dict()
        self.train_text = []
        
        self.inFile = inFile
        self.txtFile = txtFile
        self.xml_to_txt()
        self.read_words()
        self.init_table_negatives()
#         self.init_table_discards()
#         self.test()
        
        
        
    def xml_to_txt(self):
        if not os.path.isfile(self.txtFile):
            print(">>> Converting XML to TXT file...")

            i = 0

            wiki = WikiCorpus(self.inFile, lemmatize=False, dictionary={})
            
            
            output = open(self.txtFile, 'w')

            with open(self.txtFile, 'w') as f:

                for text in wiki.get_texts():
                    f.write(' '.join(text) + '\n')
                    i = i + 1

                    if (i % 10000 == 0):
                        print("Saved {} articles".format(i))

        else:
            print(">>> Located TXT file {}...".format(self.txtFile))
            
            
    def read_words(self, thr_freq=5):
        word_freq = dict()
        
        with open(self.txtFile, 'r') as f:
            # self.sentences_count = len(f.readLines())
            corpus = np.array(f.read().split(' '))
            
            print("Total Words:",len(corpus))
            
            # 0. Prepare variables
            unique_elements, counts_elements = np.unique(corpus, return_counts=True)
            word_freq = {word : counts_elements[i] for i, word in enumerate(unique_elements)}
            word2Ind = {word : i for i, word in enumerate(unique_elements)}
            
            
            
            # 1. Remove those words that barely appear in the corpus
                
            print("Removing Low Freq words:",len(corpus))
                
            del_idx = list([])
                
            del_el = set(word for word, freq in word_freq.items() if freq < thr_freq)

            for i, word in enumerate(corpus):
                if word in del_el:
                    del_idx.append(i)
                        
            corpus = np.delete(corpus, del_idx)
                
            print("Reduced Corpus:",len(corpus))
                
                   
            # 2. Subsampling
            del_idx = list([])
            corpus_size = len(corpus)

            for i, word in enumerate(corpus):
                idx = word2Ind[word]
                freq = counts_elements[idx] / corpus_size
                prob = (np.sqrt(freq / 1e-3) + 1) * (1e-3 / freq)
                prob = 1.0 - prob

                if np.random.uniform() <= prob:
                    del_idx.append(i)
                    
            print("Deleting Sampled words...")

            corpus_sampled = np.delete(corpus, del_idx)
            print("Sampled Corpus Size:",len(corpus_sampled))
            
            
            # 2.5 Removing words with less than 3 characters
            corpus_sampled = [word for word in corpus_sampled if len(word) >= 3]
    
            # 3. Word2Ind & Ind2Word
            tokens = list({word for word in corpus_sampled})
            tokens.sort()
            self.token_count = len(tokens)
        
            print("Creating word2Ind vectors...")
            for i, token in enumerate(tokens):
                self.word2id[token] = i
                self.id2word[i] = token
                
                
            # 4. Remove sequences of repeated words
            del_rep = []
            i = 0
            while i < (len(corpus_sampled) - 1 ):
                j = i + 1
                while corpus_sampled[i] == corpus_sampled[j]:
                    del_rep.append(j)
                    j = j + 1
                i += 1
            
            print("Deleted Consecutive Words:", len(del_rep))
            corpus_sampled = np.delete(corpus_sampled, del_rep)
            print("Final Word Corpus Size:", len(corpus_sampled))
            
                
            # 5. Create train_text
            print("Creating train_text array...")
            self.train_text = [self.word2id[word] for word in corpus_sampled]
            
            # I use the word_freq after processing. Other examples exists where they use it before
            unique_elements, counts_elements = np.unique(corpus_sampled, return_counts=True)
            self.word_freq = {word : counts_elements[i] for i, word in enumerate(unique_elements)}
            
            print("Tokens:", self.token_count)
            
    
    def init_table_negatives(self):
        pow_freq = np.array( list(self.word_freq.values())) ** 0.75
        sum_pow_freq = np.sum(pow_freq)
        ratio = pow_freq / sum_pow_freq
        count = np.round(ratio * DataReader.NEGATIVE_TABLE_SIZE)
        self.max_sample_id = len(count)
        for wid, c in enumerate(count):
            self.negatives += [wid] * int(c)
        self.negatives = np.array(self.negatives)
        np.random.shuffle(self.negatives)
        
    
    # Pick a random number.Those elements that appear more have a higher probability.
    def get_negatives(self, target, N, K=5):
        res = np.random.choice(self.negatives, size=(N, K))
        for i in range(len(res)):
            for j in range(K):
                if res[i,j] == target:
                    res[i,j] =  np.random.choice(self.negatives, size=1)
                    
        return res
    
    
    def test(self):
        print("Table:")
        print(self.train_text[22045:22085])
        print([self.id2word[self.train_text[idx]] for idx in range(22045, 22085)]) 
        
        sys.exit()
        
        
#================================================================================

class Word2VecDataset(Dataset):
    def __init__(self, data, window_size):
        self.data = data
        self.window_size = window_size
        
    def __len__(self):
        return len(self.data.train_text)
    
    # Returns centerInd(v), contextInd(u) & negSamples
    def __getitem__(self, idx):
        
        v = self.data.train_text[idx]
        
        u = self.data.train_text[ max(0, idx - self.window_size) : 
                                  min(idx + self.window_size,len(self.data.train_text) )]
        
        u = [i for i in u if i != v]
        
        v = [v] * len(u)
        
        neg = self.data.get_negatives(v[0], len(v)) 
        
        return (v, u, neg)
    
    
    @staticmethod
    def collate(batches):
        v_vec = [batch[0] for batch in batches]
        u_vec = [batch[1] for batch in batches]
        neg_mat = [batch[2] for batch in batches]
        
        v_vec = list(itertools.chain(*v_vec))
        u_vec = list(itertools.chain(*u_vec))
        
        neg_mat = np.vstack(neg_mat)
      
        return torch.LongTensor(v_vec), torch.LongTensor(u_vec), torch.LongTensor(neg_mat)