import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torch.autograd import Variable
import numpy as np
import sys

"""
    u_embedding: Embedding for center word.
    v_embedding: Embedding for neighbor words.
"""


class SkipGramModel(nn.Module):
    
    def __init__(self, emb_size, emb_dimension):
        super(SkipGramModel, self).__init__()
        self.emb_size = emb_size
        self.emb_dimension = emb_dimension
        self.v_embeddings = nn.Embedding(emb_size, emb_dimension, sparse=True)
        self.u_embeddings = nn.Embedding(emb_size, emb_dimension, sparse=True)
        
        initrange = 0.5 / self.emb_dimension
        init.uniform_(self.v_embeddings.weight.data, -initrange, -initrange)
        init.constant_(self.u_embeddings.weight.data, 0)
        
    # neg_u should correspond to the index of the negative samples sampled
    def forward(self, pos_v, pos_u, neg_u):
        emb_v = self.v_embeddings(Variable(pos_v))
        emb_u = self.u_embeddings(Variable(pos_u))
        emb_neg_u = self.u_embeddings(Variable(neg_u))
        
        # Compute loss
        scores = torch.mul(emb_v, emb_u)
        scores = torch.sum(scores, dim=1)
        scores = F.logsigmoid(scores)
        
        neg_scores = torch.bmm(emb_neg_u, emb_v.unsqueeze(2)).squeeze()
        neg_scores = torch.sum(neg_scores, dim=1)
        neg_scores = F.logsigmoid(-neg_scores)

        return -1 * (torch.sum(scores) + torch.sum(neg_scores))
        
        
    def save_embedding(self, id2word, file_name):
        embedding = self.v_embeddings.weight.cpu().data.numpy()
        np.save("results/" + file_name, embedding)
#         with open(file_name, 'w') as f:
#             f.write('%d %d\n' % (len(id2word), self.emb_dimension))
#             for wid, w in id2word.items():
#                 e = ' '.join(map(lambda x: str(x), embedding[wid]))
#                 f.write('%s %s\n' % (w, e))