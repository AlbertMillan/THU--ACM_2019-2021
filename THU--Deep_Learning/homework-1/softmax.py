import numpy as np
import sys


class Softmax():
    
    def __init__(self, D, C, learning_rate=1e-3):
        self.W = 0.001 * np.random.randn(D, C)
        self.lr = learning_rate

    def get_weights(self):
        return self.W

    def predict(self, X):
        scores = np.dot(X, self.W)
        pred = np.argmax(scores, axis=1)
        return pred

    def update_weights(self, grad):
        self.W += -self.lr * grad


    def vectorized_loss(self, X, y, reg):
        """
        W   - Weights (D x C) : (784 x 10)
        X   - Training images (N x D) : (120 : 784)
        y   - Training labels (N, ) : (120, )
        reg - Regularizing constant
        """

        dW = np.zeros_like(self.W)
        num_train = X.shape[0]

        # Compute loss
        xW = np.dot(X, self.W)     # (200, 2)
        xW_exp = np.exp(xW)

        # Retrieve s_y & compute s
        s_y = xW_exp[np.arange(num_train), y]
        s_sum = np.sum(xW_exp, axis=1)

        loss = np.sum( - np.log( s_y / s_sum ) )

        # Gradient Computation
        dW = np.dot(X.T, (xW_exp / s_sum[:,None]) )

        mask = np.zeros_like(xW_exp)
        mask[np.arange(num_train), y] = -1
        mask = np.dot(X.T, mask)
        dW += mask

        # Average over all images
        loss /= num_train
        dW /= num_train

        loss += reg * np.sum(self.W**2)
        dW += reg * 2 * self.W

        return loss, dW
