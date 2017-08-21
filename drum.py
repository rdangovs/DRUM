from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import variable_scope as vs

from tensorflow.python.ops.rnn_cell_impl import RNNCell
from tensorflow.python.ops.rnn_cell_impl import _linear

sigmoid = math_ops.sigmoid 
tanh = math_ops.tanh
matm = math_ops.matmul
mul = math_ops.multiply 
relu = nn_ops.relu
sign = math_ops.sign

def rotation_operator(x, y, eps = 1e-12): 
	"""Rotation between two tensors: U(x,y) is unitary and takes x to y. 
	
	Args: 
		x: a tensor from where we want to start 
		y: a tensor at which we want to finish 
		eps: the cutoff for the normalizations (avoiding division by zero)
	Returns: 
		a tensor, which is the unitary rotation operator U(x,y)
	"""
	
	size_batch = tf.shape(x)[0]
	hidden_size = tf.shape(y)[1]

	#construct the 2x2 rotation
	u = tf.nn.l2_normalize(x, 1, epsilon = eps)
	costh = tf.reduce_sum(u * tf.nn.l2_normalize(y, 1, epsilon = eps), 1)
	sinth = tf.sqrt(1 - costh ** 2)
	step1 = tf.reshape(costh, [size_batch, 1])
	step2 = tf.reshape(sinth, [size_batch, 1])
	Rth = tf.reshape(tf.concat([step1, -step2, step2, step1], axis = 1), [size_batch, 2, 2])

	#get v and concatenate u and v 
	v = tf.nn.l2_normalize(y - tf.reshape(tf.reduce_sum(u * y, 1), [size_batch,1]) * u, 1, epsilon = eps)
	step3 = tf.concat([tf.reshape(u, [size_batch, 1, hidden_size]),
					  tf.reshape(v, [size_batch, 1, hidden_size])], 
					  axis = 1)
	
	#do the batch matmul 
	step4 = tf.reshape(u, [size_batch, hidden_size, 1])
	step5 = tf.reshape(v, [size_batch, hidden_size, 1])
	
	return (tf.eye(hidden_size, batch_shape = [size_batch]) - 
		   tf.matmul(step4, tf.transpose(step4, [0,2,1])) - 
		   tf.matmul(step5, tf.transpose(step5, [0,2,1])) + 
		   tf.matmul(tf.matmul(tf.transpose(step3, [0,2,1]), Rth), step3))

def rotation_components(x, y, eps = 1e-12): 
	"""Components for the operator U(x,y)
	   Together with `rotate` achieves best memory complexity: O(N_batch * N_hidden)

	Args: 
		x: a tensor from where we want to start 
		y: a tensor at which we want to finish 
		eps: the cutoff for the normalizations (avoiding division by zero)
	Returns: 
		Four components: u, v, [u,v] and R(theta)
	"""
	
	size_batch = tf.shape(x)[0]
	hidden_size = tf.shape(x)[1]

	#construct the 2x2 rotation
	u = tf.nn.l2_normalize(x, 1, epsilon = eps)
	costh = tf.reduce_sum(u * tf.nn.l2_normalize(y, 1, epsilon = eps), 1)
	sinth = tf.sqrt(1 - costh ** 2)
	step1 = tf.reshape(costh, [size_batch, 1])
	step2 = tf.reshape(sinth, [size_batch, 1])
	Rth = tf.reshape(tf.concat([step1, -step2, step2, step1], axis = 1), [size_batch, 2, 2])

	#get v and concatenate u and v 
	v = tf.nn.l2_normalize(y - tf.reshape(tf.reduce_sum(u * y, 1), [size_batch,1]) * u, 1, epsilon = eps)
	step3 = tf.concat([tf.reshape(u, [size_batch, 1, hidden_size]),
					  tf.reshape(v, [size_batch, 1, hidden_size])], 
					  axis = 1)
	
	#do the batch matmul 
	step4 = tf.reshape(u, [size_batch, hidden_size, 1])
	step5 = tf.reshape(v, [size_batch, hidden_size, 1])
	return step4, step5, step3, Rth 

def rotate(v1, v2, v):
	"""Rotates v with a unitary rotation U(v1,v2)

	Args: 
		v: a tensor, which is the vector we want to rotate
		== to define U(v1,v2) == 
		v1: a tensor from where we want to start 
		v2: a tensor at which we want to finish 
		
	Returns: 
		A tensor: the vector U(v1,v2)[v]
	"""
	size_batch = tf.shape(v1)[0]
	hidden_size = tf.shape(v1)[1]

	U  = rotation_components(v1, v2)
	h = tf.reshape(v, [size_batch, hidden_size, 1])

	return	(v + tf.reshape(	
							- tf.matmul(U[0], tf.matmul(tf.transpose(U[0], [0,2,1]), h))
							- tf.matmul(U[1], tf.matmul(tf.transpose(U[1], [0,2,1]), h)) 
							+ tf.matmul(tf.transpose(U[2], [0,2,1]), tf.matmul(U[3], tf.matmul(U[2], h))),
							[size_batch, hidden_size]
						))

# # test the rotations
"""
sess = tf.Session()

d = tf.constant([[1.21,1.23,3.2,0.0,1.7,0.0]],shape=[1,6])
a = tf.constant([[4.2,5.22,7.0,2.0,3.3,4.0]],shape=[1,6])
b = tf.constant([[0.7,10.0,2.3,6.5,0.0,0.5]],shape=[1,6])
c = rotate(a, b, d)
e = rotation_operator(a, b)

print(sess.run(d))
print(sess.run(e))
print(sess.run(tf.matmul(e,tf.reshape(d,[1,6,1]))))
print(sess.run(c))

input()
"""


class DRUMCell(RNNCell):
	"""De-noising Rotational Unit of Memory"""

	def __init__(self,
				 hidden_size,
			     activation = None,
    			 reuse = None,
    			 kernel_initializer = None,
    		     bias_initializer = None, 
    		     normalization = None, 
    		     eps = 1e-12
    		     ):
		super(DRUMCell, self).__init__(_reuse = reuse)
		self._hidden_size = hidden_size
		self._activation = activation or relu
		self._normalization = normalization 
		self._kernel_initializer = kernel_initializer
		self._bias_initializer = bias_initializer
		self._eps = eps 

	@property
	def state_size(self):
		return self._hidden_size
	
	@property
	def output_size(self):
		return self._hidden_size

	def call(self, inputs, state):
		"""De-noising Rotational Unit of Memory (DRUM)"""
		with vs.variable_scope("gates"): 
			bias_ones = self._bias_initializer
			if self._bias_initializer is None:
				dtype = [a.dtype for a in [inputs, state]][0]
				bias_ones = init_ops.constant_initializer(1.0, dtype = dtype)
				value = sigmoid(
				_linear([inputs, state], 2 * self._hidden_size, True, bias_ones,
						self._kernel_initializer))
			r, u = array_ops.split(value = value, num_or_size_splits = 2, axis = 1)
		with vs.variable_scope("candidate"):
			x_mixed = _linear(inputs, self._hidden_size, True, self._bias_initializer, self._kernel_initializer)
			state_new = rotate(x_mixed, r, state)
			c = self._activation(x_mixed + state_new)
		new_h = u * state + (1 - u) * c
		if self._normalization != None: 
			new_h = tf.nn.l2_normalize(new_h, 1, epsilon = self._eps) * self._normalization 
		return new_h, new_h
