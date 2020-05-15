import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from nltk.metrics.spearman import spearman_correlation
from scipy.stats import spearmanr
import os
import sys

from data_reader import DataReader, Word2VecDataset
from model import SkipGramModel

os.environ["CUDA_VISIBLE_DEVICES"]="4"

# 0.025
class Word2VecTrainer:
    def __init__(self, inFile, outFile, prFile=None, emb_dimensions=100, batch_size=512, 
                 window_size=5, iterations=50, initial_lr=0.003):
        
        self.data = DataReader(inFile, txtFile=prFile)
        dataset = Word2VecDataset(self.data, window_size)
        self.dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, 
                                     num_workers=0, collate_fn=dataset.collate)
        
        self.output_file_name = outFile
        self.emb_size = len(self.data.word2id)
        self.batch_size = batch_size
        self.emb_dimensions = emb_dimensions
        self.iterations = iterations
        self.initial_lr = initial_lr
        self.skip_gram_model = SkipGramModel(self.emb_size, self.emb_dimensions)
        
        
        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device('cuda:0' if self.use_cuda else 'cpu')
        
        if self.use_cuda:
            self.skip_gram_model.cuda()
            
            
    def train(self):
        
        loss_history = []
        spear_history = []
        best_spearman = 0.0
        
        for itr in range(self.iterations):
            print("\nIteration: " + str(itr + 1))
            optimizer = optim.SGD(self.skip_gram_model.parameters(), lr=self.initial_lr)
            
            running_loss = 0.0
            for i, batch in enumerate(tqdm(self.dataloader)):
                
#                 print("V Vector:", batch[0])
#                 print("U Mat:", batch[1])
#                 print("Neg Sample:", batch[2])

                pos_v = batch[0].to(self.device)
                pos_u = batch[1].to(self.device)
                neg_u = batch[2].to(self.device)

                optimizer.zero_grad()
                loss = self.skip_gram_model.forward(pos_v, pos_u, neg_u)
                
                loss.backward()
    
                optimizer.step()
                
                running_loss = running_loss * 0.9 + loss.item() * 0.1
    

            print("Loss: " + str(running_loss))
            loss_history.append(running_loss)
            
            new_spearman = self.test(inFile="wordsim353/combined.csv")
            spear_history.append(new_spearman)
            
            if new_spearman > best_spearman:
                self.skip_gram_model.save_embedding(self.data.id2word, self.output_file_name)
                best_spearman = new_spearman
                
        return loss_history, spear_history
    
    
    def test(self, inFile, embFile="emb_art_10.npy"):
        
        self.cos_dict = dict()
        self.cos_dict_id = dict()
        
        
        # 1. Import wordsim353 and visualize it
        csv = pd.read_csv(inFile)
        csv = np.array(csv)

        idsim = dict()
        wordsim = dict()
        
        for (word_a, word_b, num) in csv:
            if word_a in self.data.word2id and word_b in self.data.word2id:
                idsim [(self.data.word2id[word_a], self.data.word2id[word_b])] = num
                wordsim[(word_a, word_b )] = num
        
        # 2. Load embeddings & normalize them
        if not self.skip_gram_model.v_embeddings:
            self.embeddings = np.load(embFile, allow_pickle=True)
        else:
            self.embeddings = self.skip_gram_model.v_embeddings.weight.cpu().data.numpy()

        
        # 3. Compute Cosine Similarities
        for (id_a,id_b), value in idsim.items():

            embeddings_a = self.embeddings[id_a].reshape(1,-1)
            embeddings_b = self.embeddings[id_b].reshape(1,-1)
            
            similarity = np.asscalar(cosine_similarity(embeddings_a, embeddings_b)[0])
            
            self.cos_dict[(self.data.id2word[id_a], self.data.id2word[id_b])] = similarity
            self.cos_dict_id[id_a, id_b] = similarity

            
        # Array form
        a = list([])
        b = list([])
        for (id_a,id_b), value in idsim.items():
            a.append(value)
            b.append(self.cos_dict_id[(id_a,id_b)])
        
        
        
        print("Spearman Coefficient:",spearman_correlation(self.cos_dict_id, idsim))
        spear = spearmanr(a,b)
        
        print(spear)
        
        return (spear[0])
        
    
if __name__ == '__main__':
    
    num_art = 10000
    emb_dimensions = 100
    window_size = 3
    iterations = 300
    
    
    print("===================== PRE-PROCESSING =======================")
    
    w2v = Word2VecTrainer(inFile="enwiki-latest-pages-articles.xml.bz2", 
                          outFile=("emb_art_" + str(num_art)+"__"+str(emb_dimensions)+".npy"), 
                          prFile=("wiki_"+str(num_art)+"_art.txt"), 
                          window_size=window_size, emb_dimensions=emb_dimensions, iterations=iterations)
    
    print("======================== TRAINING ==========================")
    loss_history, spear_history = w2v.train()
    np.save("results/loss_" + str(num_art)+"__"+str(emb_dimensions)+".npy", loss_history)
    np.save("results/spear_" + str(num_art)+"__"+str(emb_dimensions)+".npy", spear_history)
    
    
    print("========================= TESTING ==========================")
    
#     w2v.test(inFile="wordsim353/combined.csv", embFile=("emb_art_" + str(num_art))+".npy")
    