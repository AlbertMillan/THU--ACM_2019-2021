""" Euclidean Loss Layer """

import numpy as np

class EuclideanLossLayer():
	def __init__(self):
		self.acc = 0.
		self.loss = 0.

	def forward(self, logit, gt):
		"""
	      Inputs: (minibatch)
	      - logit: forward results from the last FCLayer, shape(batch_size, 10)
	      - gt: the ground truth label, shape(batch_size, 10)
	    """

		############################################################################
	    # TODO: Put your code here
		# Calculate the average accuracy and loss over the minibatch, and
		# store in self.acc and self.loss respectively.
		# Only return the self.loss, self.acc will be used in solver.py.
		N, C = logit.shape
        
		# Loss
		self.diff = (logit - gt) / N
		self.loss = np.linalg.norm(self.diff)
        
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
		return (self.diff / self.loss)

	    ############################################################################
