# -*- encoding: utf-8 -*-

import numpy as np
import sys

class MaxPoolingLayer():
	def __init__(self, kernel_size, pad):
		'''
		This class performs max pooling operation on the input.
		Args:
			kernel_size: The height/width of the pooling kernel.
			pad: The width of the pad zone.
		'''

		self.kernel_size = kernel_size
		self.pad = pad
		self.trainable = False

	def forward(self, Input, **kwargs):
		'''
		This method performs max pooling operation on the input.
		Args:
			Input: The input need to be pooled.
		Return:
			The tensor after being pooled.
		'''
		############################################################################
	    # TODO: Put your code here
		# Apply convolution operation to Input, and return results.
		# Tips: you can use np.pad() to deal with padding.
		self.Input = Input
		x_padded = np.pad(Input, ((0,), (0,), (self.pad,), (self.pad,)), mode='constant', constant_values=0)
        
		(N, C, H, W) = Input.shape
		ph, pw, s = self.kernel_size, self.kernel_size, 2
    
		H_new = int(1 + (H - ph) / s)
		W_new = int(1 + (W - pw) / s)
    
		out = np.zeros((N,C,H_new,W_new))

		same_size = ph == pw == s
		tiles = (H % ph == 0) and (W % pw == 0)
        
		# Fast Pooling
		if same_size and tiles:
		    x_reshaped = Input.reshape(N, C, H // ph, ph, W // pw, pw)
		    out = x_reshaped.max(axis=3).max(axis=4)
		    self.params = ('reshape', x_reshaped, out)
        
		# Slow Pooling. Does not get executed in the unless the kernel does not fit the input.
		else:
		    self.params = ('naive', None, None)
		    # Performed on a per-channel/depth basis
		    for i in range(N):
		        for j in range(H_new):
		            for k in range(W_new):
		                # 1-by-1 depth element case
		                out[i,:,j,k] = np.amax(Input[i,:,(j*s):(j*s+ph),(k*s):(k*s+pw)].reshape(C, ph*pw), axis=1)
        
		return out

	    ############################################################################

	def backward(self, delta):
		'''
		Args:
			delta: Local sensitivity, shape-(batch_size, filters, output_height, output_width)
		Return:
			delta of previous layer
		'''
		############################################################################
	    # TODO: Put your code here
		# Calculate and return the new delta.

		(method, x_reshaped, out) = self.params
        
		(N, C, H, W) = self.Input.shape
		ph, pw, s = self.kernel_size, self.kernel_size, 2

		H_new = int(1 + (H - ph) / s)
		W_new = int(1 + (W - pw) / s)

		dx = np.zeros_like(self.Input)
        
		if method == 'reshape':
		    dx_reshaped = np.zeros_like(x_reshaped)
		    out_newaxis = out[:, :, :, np.newaxis, :, np.newaxis]
		    mask = (x_reshaped == out_newaxis)
		    delta_newaxis = delta[:, :, :, np.newaxis, :, np.newaxis]
		    delta_broadcast, _ = np.broadcast_arrays(delta_newaxis, dx_reshaped)
		    dx_reshaped[mask] = delta_broadcast[mask]
		    dx_reshaped /= np.sum(mask, axis=(3, 5), keepdims=True)
		    dx = dx_reshaped.reshape(self.Input.shape)
            
		else:
		    for i in range(N):
		        for c in range(C): 
		            for j in range(H_new):
		                for k in range(W_new):
		                    window = self.Input[i,c,(j*s):(j*s+ph),(k*s):(k*s+pw)]
		                    m = np.max(window)

		                    dx[i,c,(j*s):(j*s+ph),(k*s):(k*s+pw)] = (window == m) * delta[i,c,j,k]
        
		return dx

	    ############################################################################
