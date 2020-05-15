import numpy as np
import torch
from torchtext import data, vocab, datasets
from torchtext.vocab import Vectors, GloVe
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
    
    """
    
    def __init__(self):
        
        self.TEXT = None
        self.LABEL = None
        
        self.train_data = None
        self.val_data = None
        self.test_data = None
        
    
    def init_dataset(self, batch_size, device):
        # set up fields
        self.TEXT = data.Field(lower=True, include_lengths=True, batch_first=True)
        self.LABEL = data.Field(sequential=False, dtype=torch.long)

        # make splits for data
        self.train_data, self.val_data, self.test_data = datasets.SST.splits(self.TEXT, self.LABEL, fine_grained=True, train_subtrees=False)

        # build the vocabulary
        self.TEXT.build_vocab(self.train_data, vectors=GloVe(name='6B', dim=300))
#         self.TEXT.build_vocab(self.train_data, vectors=Vectors(name='vector.txt', cache='./data'))
        self.LABEL.build_vocab(self.train_data)

        # make iterator for splits
        train_iter, val_iter, test_iter = data.BucketIterator.splits((self.train_data, self.val_data, self.test_data), 
                                                                     batch_size=batch_size,
                                                                     sort_key=(lambda x: len(x.text)),
                                                                     sort_within_batch=True,
                                                                     device=device,
                                                                     repeat=False)
        
        print("Train size:",len(self.train_data), 
              "Val size:",len(self.val_data),
              "Test size:",len(self.test_data))
        
        print("Train Batch size:", len(train_iter), 
              "Val Batch size:", len(val_iter),
              "Test Batch size:",len(test_iter))
        
        print("Length of Text Vocabulary: ", len(self.TEXT.vocab))
        print("Example Word 'Rock': ", self.TEXT.vocab.stoi['rock'])
        print("Vector size of Text Vocabulary: ", self.TEXT.vocab.vectors.size())
        print("Label Length: ", len(self.LABEL.vocab))
        
        return train_iter, val_iter, test_iter
        
# #================================================================================

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