import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import sys


class CNN(nn.Module):
    
    def __init__(self, feature_maps, filter_size=5, padding=2, dropout=0.0):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, feature_maps[0], filter_size, padding=padding)
        self.conv2 = nn.Conv2d(feature_maps[0], feature_maps[1], filter_size, padding=padding)
        self.fc1 = nn.Linear(feature_maps[1] * 7 * 7, 128)
        self.dropout = nn.Dropout(0.0)
        self.fc2 = nn.Linear(128, 10)
        
        self.crossEntropy = nn.CrossEntropyLoss()
        
    
    def conv_block(self, x, conv_layer):
        x_conv = conv_layer(x)
        x_act = F.relu(x_conv)
        x_pool = F.max_pool2d(x_act, 2)
        return x_pool
        
    def forward(self, x):
        out = self.conv_block(x, self.conv1)     # [batch_size, 32, 14, 14]
        out = self.conv_block(out, self.conv2)   # [batch_size, 64,  7,  7]
        out = self.fc1(out.view(x.size(0), -1))
        out = F.relu(out)
        out = self.dropout(out)
        logits = self.fc2(out)
        
        return logits
    
    
    def loss(self, logits, y_batch):
        return self.crossEntropy(logits, y_batch)
        