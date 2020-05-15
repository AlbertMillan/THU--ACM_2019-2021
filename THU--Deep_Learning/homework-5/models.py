import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import sys

"""
    u_embedding: Embedding for center word.
    v_embedding: Embedding for neighbor words.
"""


class RNN(nn.Module):
    
    def __init__(self, emb_size, emb_dimension, pretrained_emb, output_size, 
                 num_layers=2, hidden_size=256, dropout=0.0):
        super(RNN, self).__init__()
        
        self.hidden_size = hidden_size
        self.sup = num_layers * 2             # num_layers * num_directions
        
        # Loading pre-trained embeddings
        self.embeddings = nn.Embedding(emb_size, emb_dimension)
        self.embeddings.weight = nn.Parameter(pretrained_emb, requires_grad=False)
        
        # Bi-directional layer
        self.rnn = nn.RNN(emb_dimension, hidden_size, num_layers=num_layers, bidirectional=True)
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout)
        
        # Linear Layers
        self.fc1 = nn.Linear(self.sup*hidden_size, int(hidden_size/2))
        self.fc2 = nn.Linear(int(hidden_size/2), output_size)
        
        # Cross-Entropy Loss
        self.softmax = nn.CrossEntropyLoss()
        
        
    def linear_block(self, x, linear_layer):
        linear_out = linear_layer(x)
        linear_out = F.relu(linear_out)
        linear_out = self.dropout(linear_out)
        return linear_out
        
        
    def forward(self, input_sentences, y_batch):
        
        # Get embeddings of each of the words in the sentences
        sentence_of_emb = self.embeddings(input_sentences)
        
        # [sentence_length, batch_size, emb_dim]
        rnn_input = sentence_of_emb.permute(1,0,2)
        h_0 = torch.zeros(self.sup, input_sentences.size(0), self.hidden_size).cuda()
        
        # out : [sentence_length, batch_size, 2 * hidden_size]
        out, h_n = self.rnn(rnn_input, h_0)
        
        # h_n : [batch_size, 4 * self.hidden_size]
        h_n = h_n.permute(1, 0, 2)
        h_n = h_n.contiguous().view( h_n.size(0), -1 )
        
        linear_out_1 = self.linear_block(h_n, self.fc1)
        
        logits = self.fc2(linear_out_1)
        
        loss = self.softmax(logits, y_batch)

        return loss, logits
    
    


class CNN(nn.Module):
    
    def __init__(self, emb_size, emb_dimension, output_size, pretrained_emb, dropout=0.0,
                       kernel_sz=3, stride=1, padding=(1,0), out_filters=256):
    
        super(CNN, self).__init__()
        
        # Loading pre-trained embeddings
        self.embeddings = nn.Embedding(emb_size, emb_dimension)
        self.embeddings.weight = nn.Parameter(pretrained_emb, requires_grad=False)
        
        # Convolutional Layer
        self.conv1 = nn.Conv2d(1, out_filters, (kernel_sz, emb_dimension), stride=stride, padding=padding)
        
        # Dropout Layer
        self.dropout = nn.Dropout(p=dropout)
        
        # Linear Layers
        self.fc1 = nn.Linear(out_filters, int(out_filters/2) )
        self.fc2 = nn.Linear(int(out_filters/2), output_size)
        
        # Cross-Entropy Loss
        self.softmax = nn.CrossEntropyLoss()
    
    
    # If sentences are shorter than the kernel size, it yields out of bounds err.
    def conv_block(self, x, conv_layer):
        
        # [batch_size, out_filters, H_new, 1]
        conv_out = conv_layer(x)

        # [batch_size, out_filters, H_new]
        conv_out_act = F.relu(conv_out.squeeze(3))
        
        # Max-overtime pooling: Take the maximum value for each of the feature maps
        # [batch_size, out_filters]
        conv_out = F.max_pool1d(conv_out_act, conv_out_act.size()[2]).squeeze(2)
        
        return conv_out
    
    
    def linear_block(self, x, linear_layer):
        linear_out = linear_layer(x)
        linear_out = F.relu(linear_out)
        linear_out = self.dropout(linear_out)
        return linear_out
    
        
    # neg_u should correspond to the index of the negative samples sampled
    def forward(self, input_sentences, y_target):
        
        # Get embeddings of each of the words in the sentences
        sentence_of_emb = self.embeddings(input_sentences)
        
        # Reformat to 2D Convolutional shape [batch_size, 1, sentence_length, emb_dim]
        sentence_of_emb = sentence_of_emb.unsqueeze(1)
        
        # Convolutional Block: [batch_size, out_filters]
        max_out = self.conv_block(sentence_of_emb, self.conv1)
        
        linear_out_1 = self.linear_block(max_out, self.fc1)
        
        logits = self.fc2(linear_out_1)
        
        loss = self.softmax(logits, y_target)
        
        
        return loss, logits