""" Dropout Layer """

import numpy as np

class DropoutLayer():
	def __init__(self):
		self.trainable = False

	def forward(self, Input, is_training=True):

		############################################################################
	    # TODO: Put your code here
		# Assume dropout of 0.5. Would be more elegant if passed as a a parameter
		p = 0.25
		mask = None
        
		if is_training:
		    mask = (np.random.rand(*Input.shape) > p) / p
		    out = Input * mask
		else:
		    out = Input
            
		self.cache = (p, is_training, mask)
        
		return out       
        
	    ############################################################################

	def backward(self, delta):

		############################################################################
	    # TODO: Put your code here

		(p, is_training, mask) = self.cache
        
		if is_training:
		    dout = delta * mask
		else:
		    dout = delta
            
		return dout
	    ############################################################################
