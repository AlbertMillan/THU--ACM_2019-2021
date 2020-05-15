# -*- encoding: utf-8 -*-

import numpy as np
import sys

# if you implement ConvLayer by convolve function, you will use the following code.
from scipy.signal import fftconvolve as convolve

class ConvLayer():
	"""
	2D convolutional layer.
	This layer creates a convolution kernel that is convolved with the layer
	input to produce a tensor of outputs.
	Arguments:
		inputs: Integer, the channels number of input.
		filters: Integer, the number of filters in the convolution.
		kernel_size: Integer, specifying the height and width of the 2D convolution window (height==width in this case).
		pad: Integer, the size of padding area.
		trainable: Boolean, whether this layer is trainable.
	"""
	def __init__(self, inputs,
	             filters,
	             kernel_size,
	             pad,
	             trainable=True):
		self.inputs = inputs
		self.filters = filters
		self.kernel_size = kernel_size
		self.pad = pad
		assert pad < kernel_size, "pad should be less than kernel_size"
		self.trainable = trainable

		self.XavierInit()

		self.grad_W = np.zeros_like(self.W)
		self.grad_b = np.zeros_like(self.b)

	def XavierInit(self):
		raw_std = (2 / (self.inputs + self.filters))**0.5
		init_std = raw_std * (2**0.5)

		self.W = np.random.normal(0, init_std, (self.filters, self.inputs, self.kernel_size, self.kernel_size))
		self.b = np.random.normal(0, init_std, (self.filters,))

	def forward(self, Input, **kwargs):
		'''
		forward method: perform convolution operation on the input.
		Agrs:
			Input: A batch of images, shape-(batch_size, channels, height, width)
		'''
		############################################################################
	    # TODO: Put your code here
		# Apply convolution operation to Input, and return results.
		# Tips: you can use np.pad() to deal with padding.
        
		self.Input = Input
		x_padded = np.pad(Input, ((0,), (0,), (self.pad,), (self.pad,)), mode='constant', constant_values=0)
		pad = self.pad
        
		(N, C, H, W) = Input.shape
		(F, _, HH, WW) = self.W.shape
		s = 1

		# Img2col indices
		assert (H + 2 * self.pad - HH) % s == 0 
		assert (W + 2 * self.pad - WW) % s == 0
		H_new = int(1 + (H + 2 * pad - HH) / s)
		W_new = int(1 + (W + 2 * pad - WW) / s)

		i0 = np.repeat(np.arange(HH), WW)
		i0 = np.tile(i0, C)
		i1 = s * np.repeat(np.arange(H_new), W_new)
		j0 = np.tile(np.arange(WW), HH * C)
		j1 = s * np.tile(np.arange(W_new), H_new)
		i = i0.reshape(-1, 1) + i1.reshape(1, -1)
		j = j0.reshape(-1, 1) + j1.reshape(1, -1)
    
		k = np.repeat(np.arange(C), HH * WW).reshape(-1, 1)
        
		cols = x_padded[:, k, i, j]
		self.X_cols = cols.transpose(1, 2, 0).reshape(HH * WW * C, -1)
        
        
		W_cols = self.W.reshape(F, -1)

		
		out = W_cols @ self.X_cols + self.b[:,np.newaxis]
		out = out.reshape(F, H_new, W_new, N)
		out = out.transpose(3, 0, 1, 2)

		self.params = (k, i, j, s)
        
		return out
	    ############################################################################


	def backward(self, delta):
		'''
		backward method: perform back-propagation operation on weights and biases.
		Args:
			delta: Local sensitivity, shape-(batch_size, filters, output_height, output_width)
		Return:
			delta of previous layer
		'''
		############################################################################
	    # TODO: Put your code here
		# Calculate self.grad_W, self.grad_b, and return the new delta.
        
		(N, C, H, W) = self.Input.shape
		(F, _, HH, WW) = self.W.shape
		(k, i, j, s) = self.params
        
		self.grad_b = np.sum(delta, axis=(0,2,3))
        
		delta_reshaped = delta.transpose(1, 2, 3, 0).reshape(F, -1)
		dW = delta_reshaped @ self.X_cols.T
		self.grad_W = dW.reshape(self.W.shape)
        
		W_reshape = self.W.reshape(F, -1)
		dX_cols = W_reshape.T @ delta_reshaped
        
		# col2im
		H_padded, W_padded = H + 2 * self.pad, W + 2 * self.pad
		dx_padded = np.zeros((N, C, H_padded, W_padded), dtype=dX_cols.dtype)
        
		cols_reshaped = dX_cols.reshape(C * HH * WW, -1, N).transpose(2, 0, 1)
		np.add.at(dx_padded, (slice(None), k, i, j), cols_reshaped)
        
		if self.pad != 0:
		    dx_padded = dx_padded[:,:, self.pad:-self.pad, self.pad:-self.pad]
        
		return dx_padded
	    ############################################################################
