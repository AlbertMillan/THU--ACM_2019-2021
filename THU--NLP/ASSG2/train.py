import torch
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import argparse
import os
import sys

from data_reader import DataReader, BatchGenerator
from models import CNN

os.environ["CUDA_VISIBLE_DEVICES"]="4"


class TextClassifier:
    def __init__(self, paths, batch_size=6, iterations=50, initial_lr=0.003,
                       hidden_size=256, dropout=0.2, kernel_sz=3):
        
        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device('cuda:0' if self.use_cuda else 'cpu')
        
        self.data = DataReader(paths)
        self.data.set_training_data(batch_size, ('cuda:0' if self.use_cuda else 'cpu'))
        self.train_batch_loader = BatchGenerator(self.data.train_data, 'Sentence', 'Label')
        self.val_batch_loader = BatchGenerator(self.data.val_data, 'Sentence', 'Label')
        self.test_batch_loader = BatchGenerator(self.data.test_data, 'Sentence', 'Label')

        # Store hyperparameters
        self.batch_size = batch_size
        self.iterations = iterations
        self.initial_lr = initial_lr
        self.kernel_sz = kernel_sz
        
        # Create Model
        emb_size, emb_dim = self.data.TEXT.vocab.vectors.size()
        self.cnn_model = CNN(emb_size=emb_size, emb_dimension=emb_dim,
                             n_out=len(self.data.LABEL.vocab),
                             dropout=dropout, kernel_sz=kernel_sz, stride=1, padding=0,
                             out_filters=hidden_size, pretrained_emb=self.data.TEXT.vocab.vectors)
        
        
        if self.use_cuda:
            self.cnn_model.cuda()
            
            
    def train(self):
        
        train_loss_hist = []
        val_loss_hist = []
        train_acc_hist = []
        val_acc_hist = []
        test_acc_hist = []
        
        loss = 0.0
        
        best_model = 0.0
        
        for itr in range(self.iterations):
            print("\nIteration: " + str(itr + 1))
            optimizer = optim.SGD(self.cnn_model.parameters(), lr=self.initial_lr)
            self.cnn_model.train()
            total_loss = 0.0
            total_acc = 0.0
            steps = 0
            
            data_iter = iter(self.train_batch_loader)
    
            # For some reason using for loop on iterator (next) is missing the target variable (y)
            # Have to loop over the length and retrieve the batch_data inside the loop
            for i in range(len(self.train_batch_loader)):
            
                ((x_batch,  x_len_batch), y_batch) = next(data_iter)

#                 if torch.min(x_len_batch) > self.kernel_sz:
                optimizer.zero_grad()

                loss, logits = self.cnn_model.forward(x_batch, y_batch)

                acc = torch.sum(torch.argmax(logits, dim=1) == y_batch)

                total_loss += loss.item()
                total_acc += acc.item()
                steps += 1

                loss.backward()

                optimizer.step()
                    
            
            train_loss_hist.append(total_loss/steps)
            train_acc_hist.append(total_acc / len(self.data.trainds) )
            
            val_loss, val_acc = self.eval_model(self.val_batch_loader, len(self.data.valds))
            
            val_loss_hist.append(val_loss)
            val_acc_hist.append(val_acc)
            
            if best_model < val_acc:
                best_model = val_acc
                test_loss, test_acc = self.eval_model(self.test_batch_loader, len(self.data.testds) )
                
            
            print("Train: {Loss: " + str(total_loss/steps) + ", Acc: " + str(total_acc / len(self.data.trainds)) + " }" )
            print("Val: {Loss: " + str(val_loss) + ", Acc: " + str(val_acc) + " }" )
            
        
#         test_loss, test_acc = self.eval_model(self.test_batch_loader, len(self.data.testds) )
        
        test_acc_hist.append(test_acc)
            
        return train_loss_hist, train_acc_hist, val_loss_hist, val_acc_hist, test_acc
    
    
    def eval_model(self, batch_loader, N):
        self.cnn_model.eval()
        
        total_loss = 0.0
        total_acc = 0.0
        steps = 0
        
        batch_iter = iter(batch_loader)
        
        with torch.no_grad():
            for i in range(len(batch_loader)):
            
                ((x_batch,  x_len_batch), y_batch) = next(batch_iter)
                
                loss, logits = self.cnn_model(x_batch, y_batch)

                acc = torch.sum(torch.argmax(logits, dim=1) == y_batch)

                total_loss += loss.item()
                total_acc += acc.item()
                steps += 1
                
        return (total_loss/steps), (total_acc/N)
        
# =======================================================================================
    
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='PyTorch SST Training. See code for default values.')
    
    # MODEL HYPERPARAMETERS
    parser.add_argument('--lr', default=0.003, metavar='lr', type=float, help='Learning rate')
    parser.add_argument('--itr', default=120, metavar='iter', type=int, help='Number of iterations')
    parser.add_argument('--hidden', default=256, metavar='iter', type=int, help='Hidden features')
    parser.add_argument('--batch_size', default=32, metavar='batch_size', type=int, help='Batch size')
    parser.add_argument('--momentum', '--m', default=0.9, type=float, help='Momentum')
    parser.add_argument('--dropout', '--dr', default=0.2, type=float, help='Dropout')
    parser.add_argument('--kernel_sz', '--k', default=3, type=int, help='Kernel Size')
    
    
    args = parser.parse_args()
    
    
    iterations = args.itr
    n_hidden = 1
    hidden_size = args.hidden
    dr = args.dropout
    drop = (1 if dr != 0.0 else 0)
    folder = 'trees'
    paths = {
        'train': os.path.join(folder,'train.txt'),
        'val': os.path.join(folder,'dev.txt'),
        'test': os.path.join(folder,'test.txt')
    }
    
    
    print("===================== PRE-PROCESSING =======================")
    
    sentimentClassifier = TextClassifier(paths=paths, batch_size=args.batch_size, iterations=iterations, 
                                         dropout=dr, hidden_size=hidden_size, kernel_sz=args.kernel_sz)
    
    print("======================== TRAINING ==========================")
    
    train_loss_hist, train_acc_hist, val_loss_hist, val_acc_hist, test_acc = sentimentClassifier.train()
    
    np.save("results/train_loss_iter" + str(iterations)+"__h"+str(n_hidden)+"_"+str(hidden_size)+"__dr"+str(drop)+"__k"+str(args.kernel_sz)+".npy", train_loss_hist)
    
    np.save("results/train_acc_iter" + str(iterations)+"__h"+str(n_hidden)+"_"+str(hidden_size)+"__dr"+str(drop)+"__k"+str(args.kernel_sz)+".npy", train_acc_hist)
    
    np.save("results/val_loss_iter" + str(iterations)+"__h"+str(n_hidden)+"_"+str(hidden_size)+"__dr"+str(drop)+"__k"+str(args.kernel_sz)+".npy", val_loss_hist)
    
    np.save("results/val_acc_iter" + str(iterations)+"__h"+str(n_hidden)+"_"+str(hidden_size)+"__dr"+str(drop)+"__k"+str(args.kernel_sz)+".npy", val_acc_hist)
    
    np.save("results/test_acc_iter" + str(iterations)+"__h"+str(n_hidden)+"_"+str(hidden_size)+"__dr"+str(drop)+"__k"+str(args.kernel_sz)+".npy", test_acc)
    
    
    
    print("========================= TESTING ==========================")
    
    print("Test ACC:", test_acc)
    
    