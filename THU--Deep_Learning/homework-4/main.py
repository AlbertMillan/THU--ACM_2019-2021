import numpy as np
import torchvision
from torchvision import transforms
import torch
from torch import optim

import argparse
import time
import shutil
import sys, os

from model import CNN


class Classifier:
    
    def __init__(self, ds_path, lr, iterations, batch_size, hidden_layers_out, 
                 print_freq, save_dir, momentum, dropout):
        
        self.train_data = torchvision.datasets.MNIST(ds_path, train=True, 
                                                       transform=transforms.ToTensor(), 
                                                       download=True)
        
        self.test_data = torchvision.datasets.MNIST(ds_path, train=False, 
                                                      transform=transforms.ToTensor(), 
                                                      download=True)
        
        self.train_loader = torch.utils.data.DataLoader(self.train_data, batch_size=batch_size, shuffle=True)
        self.test_loader = torch.utils.data.DataLoader(self.test_data, batch_size=batch_size)
        
        self.save_dir = save_dir
        self.is_momentum = (momentum != 0.0)
        
        # Set Model Hyperparameters
        self.learning_rate = lr
        self.iterations = iterations
        self.print_freq = print_freq
        self.model = CNN(hidden_layers_out, dropout=dropout)
        
        self.cuda = torch.cuda.is_available()
        
        if self.cuda:
            self.model = self.model.cuda()
            
    
    def train(self, momentum, nesterov, weight_decay):
        
        train_loss_hist = []
        train_acc_hist = []
        test_loss_hist = []
        test_acc_hist = []
        
        best_pred = 0.0
        
        end = time.time()
        
        for itr in range(self.iterations):
            
            self.model.train()
            
            if self.is_momentum:
                optimizer = optim.SGD(self.model.parameters(), lr=self.learning_rate, 
                                      momentum=momentum, nesterov=nesterov, 
                                      weight_decay=weight_decay)
            else:
                optimizer = optim.SGD(self.model.parameters(), lr=self.learning_rate)
            
            losses = AverageMeter()
            batch_time = AverageMeter()
            top1 = AverageMeter()
            
            for i, (x_batch, y_batch) in enumerate(self.train_loader):
                # Compute output for example
                logits = self.model(x_batch)
                loss = self.model.loss(logits, y_batch)

                # Update Mean loss for current iteration
                losses.update(loss.item(), x_batch.size(0))
                prec1 = self.accuracy(logits.data, y_batch, k=1)
                top1.update(prec1.item(), x_batch.size(0))

                # compute gradient and do SGD step
                loss.backward()
                optimizer.step()

                # Set grads to zero for new iter
                optimizer.zero_grad()
                
                batch_time.update(time.time() - end)
                end = time.time()
                
                if i % self.print_freq == 0:
                    print('Epoch: [{0}][{1}/{2}]\t'
                          'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                          'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                          'Acc {top1.val:.3f} ({top1.avg:.3f})'.format(
                              itr, i, len(self.train_loader), batch_time=batch_time,
                              loss=losses, top1=top1))
                    
            # evaluate on validation set
            test_loss, test_prec1 = self.test(self.test_loader)
            
            train_loss_hist.append(losses.avg)
            train_acc_hist.append(top1.avg)
            test_loss_hist.append(test_loss)
            test_acc_hist.append(test_prec1)
            
            # Store best model
            is_best = best_pred < test_prec1
            if is_best:
                best_pred = test_prec1
                self.save_checkpoint(is_best, (itr+1), self.model.state_dict(), self.save_dir)
                
        return (train_loss_hist, train_acc_hist, test_loss_hist, test_acc_hist)
        
        
    def test(self, batch_loader):
        self.model.eval()
        
        losses = AverageMeter()
        batch_time = AverageMeter()
        top1 = AverageMeter()
        
        end = time.time()
        
        for i, (x_batch,y_batch) in enumerate(batch_loader):

            with torch.no_grad():
                logits = self.model(x_batch)
                loss = self.model.loss(logits, y_batch)

            # Update Mean loss for current iteration
            losses.update(loss.item(), x_batch.size(0))
            prec1 = self.accuracy(logits.data, y_batch, k=1)
            top1.update(prec1.item(), x_batch.size(0))

            batch_time.update(time.time() - end)
            end = time.time()
            
            if i % self.print_freq == 0:
                print('Epoch: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc {top1.val:.3f} ({top1.avg:.3f})'.format(
                          i, len(batch_loader), batch_time=batch_time,
                          loss=losses, top1=top1))

        print(' * Acc {top1.avg:.3f}'.format(top1=top1))
        return (losses.avg, top1.avg)
    
    
    def accuracy(self, output, y, k=1):
        """Computes the precision@k for the specified values of k"""
        # Rehape to [N, 1]
        target = y.view(-1, 1)

        _, pred = torch.topk(output, k, dim=1, largest=True, sorted=True)
        correct = torch.eq(pred, target)

        return torch.sum(correct).float() / y.size(0)
    
    
    def save_checkpoint(self, is_best, epoch, state, save_dir, base_name="chkpt_plain"):
        """Saves checkpoint to disk"""
        directory = save_dir
        filename = base_name + ".pth.tar"
        if not os.path.exists(directory):
            os.makedirs(directory)
        filename = directory + filename
        torch.save(state, filename)
        if is_best:
            shutil.copyfile(filename, directory + base_name + '__model_best.pth.tar')

    
class AverageMeter():
    
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        
        
        
        
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='PyTorch MNIST Training. See code for default values.')

    
    # STORAGE LOCATION VARIABLES
    parser.add_argument('--ds_path', default='datasets/', metavar='Path', type=str, help='Dataset path')
    parser.add_argument('--save_dir', '--sd', default='model_chkpt/new/', type=str, help='Path to Model')
#     parser.add_argument('--save_name', '--mn', default='chkpt_plain.pth.tar', type=str, help='File Name')
    
    
    # MODEL HYPERPARAMETERS
    parser.add_argument('--lr', default=0.001, metavar='lr', type=float, help='Learning rate')
    parser.add_argument('--itr', default=20, metavar='iter', type=int, help='Number of iterations')
    parser.add_argument('--batch_size', default=128, metavar='batch_size', type=int, help='Batch size')
    parser.add_argument('--momentum', '--m', default=0.9, type=float, help='Momentum')
    parser.add_argument('--nesterov', default=True, type=bool, help='nesterov momentum')
    parser.add_argument('--weight_decay', '--wd', default=5e-4, type=float, help='weight decay (default: 5e-4)')
    parser.add_argument('--print_freq', '-p', default=10, type=int, help='print frequency (default: 10)')
    
    parser.add_argument('--h1', default=32, type=int, help='Feature maps produced by conv layer 1')
    parser.add_argument('--h2', default=64, type=int, help='Feature maps produced by conv layer 2')
    parser.add_argument('--dropout', '--dr', default=0.0, type=float, help='Dropout rate (deafault: 0.0)')
    
    args = parser.parse_args()
    
    hidden_layers_out = [args.h1, args.h2]
    is_dropout = (1 if args.dropout == 0.0 else 0)
    
    
    classifier = Classifier(args.ds_path, args.lr, args.itr, args.batch_size, 
                            hidden_layers_out, args.print_freq, args.save_dir, args.momentum,
                            args.dropout)
    
    print("======================= TRAINING =======================")
    
    (train_loss_hist, train_acc_hist, test_loss_hist, test_acc_hist) = classifier.train(args.momentum, args.nesterov, args.weight_decay)
    
    np.save("results/train_loss__"+str(hidden_layers_out[0])+"_"+
                                   str(hidden_layers_out[1])+"__dr"+
                                   str(is_dropout)+".npy",
                                        train_loss_hist)
    
    np.save("results/train_acc__"+str(hidden_layers_out[0])+"_"+
                                  str(hidden_layers_out[1])+"__dr"+
                                  str(is_dropout)+".npy", 
                                        train_acc_hist)
    
    np.save("results/test_loss__"+str(hidden_layers_out[0])+"_"+
                                  str(hidden_layers_out[1])+"__dr"+
                                  str(is_dropout)+".npy",
                                        test_loss_hist)
    
    np.save("results/test_acc__"+str(hidden_layers_out[0])+"_"+
                                 str(hidden_layers_out[1])+"__dr"+
                                 str(is_dropout)+".npy",
                                        test_acc_hist)