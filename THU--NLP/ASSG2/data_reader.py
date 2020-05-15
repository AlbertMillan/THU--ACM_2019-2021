import numpy as np
import torch
from torchtext import data, vocab, datasets
from torchtext.vocab import Vectors, GloVe
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import pandas as pd
import re

import os.path
import os
import sys


np.random.seed(12345)

class DataReader:
    """
    self.TEXT :  TorchText variable to store the vocabulatry: word2id (stoi), id2word (itos) and embeddings 
                 (vectors) for the unique words in training and validation set.
                
    self.LABEL : TorchText variable storing the vocab for label data
    
    self.trainds : Training dataset in TorchText TabularDataset format. Refer to doc.
    
    self.valds : Validation dataset in TorchText TabularDataset format. Refer to doc.
    
    self.train_data : Training data in batches.
    
    self.val_data : Validation data in batches. 
    
    self.paths : Dict containing the paths of train, val and test files
    
    self.data :  Dict containing a tuple (pruned sentences, labels) for train, val and test datasets.
    
    """
    
    def __init__(self, paths):
        
        self.TEXT = None
        self.LABEL = None
        
        self.trainds = None
        self.valds = None
        self.testds = None
        
        self.train_data = None
        self.val_data = None
        
        self.paths = paths
        self.data = self.read_files()
        
        self.to_csv()
        self.init_torch_vars()
#         self.test()
        
        
     
    def init_torch_vars(self):
        """
        Field :       A class that stores information on how to preprocess the data.
        
        build_vocab : 1) Dictionary mapping all the unique words present in the train_data to an idx
                      2) Use pre-trained (GloVe) word embedding to map the index to the corresponding 
                          word embedding.
        
        """
        print(">>> Creating Torch Variables...")
        
#         Maybe I need to include the vocab object from the embeddings...?
        self.TEXT = data.Field(sequential=True, tokenize=(lambda x: x.split()), 
                               include_lengths=True, 
                               batch_first=True)
        
        self.LABEL = data.LabelField()
        
        
        train_val_fields = [
            ('Sentence', self.TEXT),
            ('Label', self.LABEL)
        ]
        
        self.trainds, self.valds, self.testds = data.TabularDataset.splits(path='./datasets', format='csv',
                                                                           train='train.csv',
                                                                           validation='val.csv', 
                                                                           test='test.csv', 
                                                                           fields=train_val_fields,
                                                                           skip_header=True)
        
#         self.TEXT.build_vocab(self.trainds, self.valds, vectors=Vectors('glove.6B.300d.txt', './'))
        self.TEXT.build_vocab(self.trainds, self.valds, vectors=GloVe(name='6B', dim=300))
        self.LABEL.build_vocab(self.trainds)
        
        print("Length of Text Vocabulary: ", len(self.TEXT.vocab))
        print("Example Word 'Rock': ", self.TEXT.vocab.stoi['rock'])
        print("Vector size of Text Vocabulary: ", self.TEXT.vocab.vectors.size())
        print("Label Length: ", len(self.LABEL.vocab))
        
        

    
    def set_training_data(self, batch_size, device):
        
        if not self.train_data or not self.val_data:
        
            self.train_data, _ = data.BucketIterator.splits(datasets=(self.trainds, self.valds),
                                                                        batch_size=batch_size,
                                                                        sort_key=(lambda x: len(x.Sentence)),
                                                                        sort_within_batch=True,
                                                                        device=device,
                                                                        repeat=False)
            
            self.val_data = data.BucketIterator(dataset=(self.valds),
                                                batch_size=batch_size,
                                                sort_key=(lambda x: len(x.Sentence)),
                                                sort_within_batch=True,
                                                device=device,
                                                repeat=False)
            
            self.test_data = data.BucketIterator(dataset=(self.testds),
                                                batch_size=batch_size,
                                                sort_key=(lambda x: len(x.Sentence)),
                                                sort_within_batch=True,
                                                device=device,
                                                repeat=False)
                                               
                
        print("Train size:",len(self.trainds), 
              "Val size:",len(self.valds),
              "Test size:",len(self.testds))
        
        print("Train Batch size:", len(self.train_data), 
              "Val Batch size:", len(self.val_data),
              "Test Batch size:",len(self.test_data))
    
    
    # 
    def read_files(self):
        print(">>> Reading Data...")
        assert len(self.paths) == 3, "ERR: Insufficient paths for train, val and test data..."
        lines = 0
        sentences = []
        all_sentences = []
        all_labels = []
        
        stopset = set(stopwords.words('english'))
        lemmatizer = WordNetLemmatizer()
        
        for key, path in self.paths.items():
                   
            with open(path, 'r') as f:
                   
                file_sentences = []
                file_labels = []
                
                for line in f:
                    
                    lines = lines + 1
                    
                    # Class label located at second position in the line
                    label = line[1]
                    file_labels.append(int(label))
                    
                    
                    # Parse line for sentence
                    # Remove Digits and find only alpha tokens
                    sentence = re.sub("\d+", "", line)
                    sentence = re.findall(r'\w+', sentence)
                    
                    # Remove stop words, convert to lower-case and lemmatize
#                     sentence = [lemmatizer.lemmatize(w.lower()) for w in sentence if w not in stopset and len(w) > 1]
                    sentence = [w.lower() for w in sentence]
                    
                    
                    sentence = " ".join(sentence)
                    
                    sentences.append(sentence)
                    file_sentences.append(sentence)
                
            all_sentences.append(file_sentences)
            all_labels.append(file_labels)
        
        return {
            'train': (all_sentences[0], all_labels[0]),
            'val':   (all_sentences[1], all_labels[1]),
            'test':  (all_sentences[2], all_labels[2]),
        }
                    
                
    def to_csv(self):
        for key, arrs in self.data.items():
            arr1 = np.array(arrs[0]).reshape(-1,1)
            arr2 = np.array(arrs[1]).reshape(-1,1)
        
            df = pd.DataFrame(np.concatenate((arr1, arr2), axis=1), 
                              columns=['Sentence','Label'])

            df.to_csv(os.path.join('datasets', key+'.csv'), index=False)
        
        
#================================================================================

class BatchGenerator:
    
    def __init__(self, train_data, x_field, y_field):
        self.train_data = train_data
        self.x_field = x_field
        self.y_field = y_field
        
    
    def __len__(self):
        return len(self.train_data)
    
    
    def __iter__(self):
        for batch in self.train_data:
            X = getattr(batch, self.x_field)
            y = getattr(batch, self.y_field)
            yield(X, y)