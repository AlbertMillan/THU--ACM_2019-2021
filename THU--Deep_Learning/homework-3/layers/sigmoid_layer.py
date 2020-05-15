""" Sigmoid Layer """

import numpy as np

class SigmoidLayer():
	def __init__(self):
		"""
		Applies the element-wise function: f(x) = 1/(1+exp(-x))
		"""
		self.trainable = False

	def forward(self, Input, **kwargs):

		############################################################################
	    # TODO: Put your code here
		# Apply Sigmoid activation function to Input, and return results.
		self.input = Input
		return 1 / (1+np.exp(-Input))

	    ############################################################################

	def backward(self, delta):

		############################################################################
	    # TODO: Put your code here
		# Calculate the gradient using the later layer's gradient: delta

		sig_z = 1 / (1+np.exp(-self.input))
		dsig_z = sig_z * (1 - sig_z)

		return dsig_z * delta.T
    
	    ############################################################################
