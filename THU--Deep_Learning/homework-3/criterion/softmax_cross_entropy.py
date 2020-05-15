""" Softmax Cross-Entropy Loss Layer """

import numpy as np

# a small number to prevent dividing by zero, maybe useful for you
EPS = 1e-11

class SoftmaxCrossEntropyLossLayer():
	def __init__(self):
		self.acc = 0.
		self.loss = np.zeros(1, dtype='f')

	def forward(self, logit, gt):
		"""
	      Inputs: (minibatch)
	      - logit: forward results from the last FCLayer, shape(batch_size, 10)
	      - gt: the ground truth label, shape(batch_size, 10)
	    """

		############################################################################
	    # TODO: Put your code here
		# Calculate the average accuracy and loss over the minibatch, and
		# store in self.accu and self.loss respectively.
		# Only return the self.loss, self.accu will be used in solver.py.

		# Pre-processing
		N, C = logit.shape
		self.N = N
        
        
		# Store values
		self.gt = gt
        
		# Loss
		self.e_s = np.exp(logit)
		e_y = np.sum(self.e_s * gt, axis=1)
		self.e_sum = np.sum(self.e_s, axis=1)
		total_loss = - np.sum( np.log(e_y / self.e_sum) )
		self.loss = total_loss / N
        
        
		# Accuracy
		pred = np.zeros_like(gt)
		pred[np.arange(N), np.argmax(logit, axis=1)] = 1
		self.acc = float( np.sum( np.logical_and(pred, gt) ) / N )

	    ############################################################################

		return self.loss


	def backward(self):

		############################################################################
	    # TODO: Put your code here
		# Calculate and return the gradient (have the same shape as logit)
		dscores = self.e_s / self.e_sum[:,None]
		dscores -= self.gt

		return dscores / self.N

	    ############################################################################
