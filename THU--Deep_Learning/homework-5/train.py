import torch
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import os
import sys
import argparse
import math

from data_reader import DataReader, BatchGenerator
from models import CNN, RNN

os.environ["CUDA_VISIBLE_DEVICES"]="2"


class TextClassifier:
    def __init__(self, batch_size, iterations, initial_lr,
                       hidden_size, dropout, kernel_sz,
                       num_layers):
        
        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device('cuda:0' if self.use_cuda else 'cpu')
        
        self.data = DataReader()
        train_iter, val_iter, test_iter = self.data.init_dataset(batch_size, ('cuda:0' if self.use_cuda else 'cpu'))
        self.train_batch_loader = BatchGenerator(train_iter, 'text', 'label')
        self.val_batch_loader = BatchGenerator(val_iter, 'text', 'label')
        self.test_batch_loader = BatchGenerator(test_iter, 'text', 'label')

        # Store hyperparameters
        self.batch_size = batch_size
        self.iterations = iterations
        self.initial_lr = initial_lr
        
        # Create Model
        emb_size, emb_dim = self.data.TEXT.vocab.vectors.size()
#         padding = (math.floor(kernel_sz / 2), 0)
        
#         self.model = CNN(emb_size=emb_size, emb_dimension=emb_dim,
#                              output_size=len(self.data.LABEL.vocab),
#                              dropout=dropout, kernel_sz=kernel_sz, stride=1, padding=padding,
#                              out_filters=hidden_size, pretrained_emb=self.data.TEXT.vocab.vectors)
        
        self.model = RNN(emb_size=emb_size, emb_dimension=emb_dim,
                             pretrained_emb=self.data.TEXT.vocab.vectors,
                             output_size=len(self.data.LABEL.vocab),
                             num_layers=num_layers,
                             hidden_size=hidden_size,
                             dropout=dropout)
        
        
        if self.use_cuda:
            self.model.cuda()
            
            
    def train(self, min_stride=3):
        
        train_loss_hist = []
        val_loss_hist = []
        train_acc_hist = []
        val_acc_hist = []
        test_acc_hist = []
        
        best_score = 0.0
        
        loss = 0.0
        
        for itr in range(self.iterations):
            print("\nIteration: " + str(itr + 1))
            optimizer = optim.SGD(self.model.parameters(), lr=self.initial_lr)
            self.model.train()
            total_loss = 0.0
            total_acc = 0.0
            steps = 0
            
            data_iter = iter(self.train_batch_loader)

            for i in range(len(self.train_batch_loader)):
            
                ((x_batch,  x_len_batch), y_batch) = next(data_iter)

#                 if torch.min(x_len_batch) > min_stride:
                optimizer.zero_grad()

                loss, logits = self.model.forward(x_batch, y_batch)

                acc = torch.sum(torch.argmax(logits, dim=1) == y_batch)

                total_loss += loss.item()
                total_acc += acc.item()
                steps += 1

                loss.backward()

                optimizer.step()
                    
            
            train_loss_hist.append(total_loss/steps)
            train_acc_hist.append(total_acc / len(self.data.train_data) )
            
            val_loss, val_acc = self.eval_model(self.val_batch_loader, len(self.data.val_data))
            
            val_loss_hist.append(val_loss)
            val_acc_hist.append(val_acc)
            
            if val_acc > best_score:
                best_score = val_acc
                test_loss, test_acc = self.eval_model(self.test_batch_loader, len(self.data.test_data))
            
            print("Train: {Loss: " + str(total_loss/steps) + ", Acc: " + str(total_acc / len(self.data.train_data)) + " }" )
            print("Val: {Loss: " + str(val_loss) + ", Acc: " + str(val_acc) + " }" )
            
        
        test_acc_hist.append(test_acc)
            
        return train_loss_hist, train_acc_hist, val_loss_hist, val_acc_hist, test_acc_hist
    
    
    def eval_model(self, batch_loader, N, min_stride=3):
        self.model.eval()
        
        total_loss = 0.0
        total_acc = 0.0
        steps = 0
        
        batch_iterator = iter(batch_loader)
        
        with torch.no_grad():
            for i in range(len(batch_loader)):
            
                ((x_batch,  x_len_batch), y_batch) = next(batch_iterator)
                
                
#                 if torch.min(x_len_batch) > min_stride:
                loss, logits = self.model(x_batch, y_batch)

                acc = torch.sum(torch.argmax(logits, dim=1) == y_batch)

                total_loss += loss.item()
                total_acc += acc.item()
                
        return (total_loss/N), (total_acc/N)
        
# =======================================================================================
    
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='PyTorch SST Training. See code for default values.')
    
    # MODEL HYPERPARAMETERS
    parser.add_argument('--lr', default=0.01, metavar='lr', type=float, help='Learning rate')
    parser.add_argument('--itr', default=100, metavar='iter', type=int, help='Number of iterations')
    parser.add_argument('--hidden', default=256, metavar='hidden', type=int, help='Hidden features')
    parser.add_argument('--num_layers', default=2, type=int, help='Number of RNN layers')
    parser.add_argument('--batch_size', default=32, metavar='batch_size', type=int, help='Batch size')
    parser.add_argument('--dropout', '--dr', default=0.2, type=float, help='Dropout')
    parser.add_argument('--kernel_sz', '--ks', default=3, type=int, help='Kernel Size (default: 3)')
    
    
    args = parser.parse_args()
    
    
    iterations = args.itr
    n_hidden = args.num_layers
    hidden_size = args.hidden
    dr = args.dropout
    drop = (1 if dr != 0.0 else 0)
    
    
    print("===================== PRE-PROCESSING =======================")
    
    sentimentClassifier = TextClassifier(batch_size=args.batch_size, iterations=iterations,
                                         initial_lr=args.lr, hidden_size=hidden_size, dropout=dr,
                                         kernel_sz=args.kernel_sz, num_layers=args.num_layers)
    
    
    print("======================== TRAINING ==========================")
    
    train_loss_hist, train_acc_hist, val_loss_hist, val_acc_hist, test_acc = sentimentClassifier.train()
    
    np.save("results/train_loss_iter" + str(iterations)+"__h"+str(n_hidden)+"_"+str(hidden_size)+"__dr"+str(drop)+".npy", train_loss_hist)
    
    np.save("results/train_acc_iter" + str(iterations)+"__h"+str(n_hidden)+"_"+str(hidden_size)+"__dr"+str(drop)+".npy", train_acc_hist)
    
    np.save("results/val_loss_iter" + str(iterations)+"__h"+str(n_hidden)+"_"+str(hidden_size)+"__dr"+str(drop)+".npy", val_loss_hist)
    
    np.save("results/val_acc_iter" + str(iterations)+"__h"+str(n_hidden)+"_"+str(hidden_size)+"__dr"+str(drop)+".npy", val_acc_hist)
    
    np.save("results/test_acc_iter" + str(iterations)+"__h"+str(n_hidden)+"_"+str(hidden_size)+"__dr"+str(drop)+".npy", test_acc)
    
    
    print("========================= TESTING ==========================")
    
    print("Test ACC:", test_acc[0])
    
    