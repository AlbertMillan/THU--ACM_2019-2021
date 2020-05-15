""" Sigmoid Layer """

import numpy as np

class SigmoidLayer():
	def __init__(self):
		"""
		Applies the element-wise function: f(x) = 1/(1+exp(-x))
		"""
		self.trainable = False

	def forward(self, Input):

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
# 		print(">>> SIGMOID BACKWARD")
# 		print("Sigmoid Delta:", delta.shape)

		sig_z = 1 / (1+np.exp(-self.input))
		dsig_z = sig_z * (1 - sig_z)
# 		print("Sig Input:", self.input.shape)
# 		print("Sig dSig:", dsig_z.shape)
# 		print("Sig Output:", (np.dot(dsig_z, delta)).shape )
		return dsig_z * delta.T


	    ############################################################################
