from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np 
import aux 
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

#taken from supercell_DRUM
def orthogonal(shape):
	shape = int(shape[0]), int(shape[1])
	flat_shape = (shape[0], np.prod(shape[1:]))
	a = np.random.normal(0.0, 1.0, flat_shape)
	u, _, v = np.linalg.svd(a, full_matrices = False)
	q = u if u.shape == flat_shape else v
	return q.reshape(shape)

#taken from supercell_DRUM
def drum_ortho_initializer(scale=1.0):
	def _initializer(shape, dtype = tf.float32, partition_info = None):
		size_x = shape[0]
		size_h = shape[1] // 2 # assumes a drum.
		t = np.zeros(shape)
		t[:, :size_h] = orthogonal([size_x, size_h]) * scale
		t[:, size_h:size_h * 2] = orthogonal([size_x, size_h]) * scale
		return tf.constant(t, dtype)
	return _initializer

def layer_norm_all(h, 
					 batch_size, 
					 base, 
					 num_units, 
					 scope = "layer_norm", 
					 reuse = False, 
					 gamma_start = 1.0, 
					 epsilon = 1e-3, 
					 use_bias = True):
	# Layer Norm (faster version, but not using defun)
	#
	# Performas layer norm on multiple base at once (ie, i, g, j, o for lstm)
	#
	# Reshapes h in to perform layer norm in parallel
	if batch_size == None: 
		batch_size = tf.shape(h)[0]
		h_reshape = tf.reshape(h, [batch_size, base, num_units])
		mean = tf.reduce_mean(h_reshape, [2], keep_dims = True)
		var = tf.reduce_mean(tf.square(h_reshape - mean), [2], keep_dims = True)
		epsilon = tf.constant(epsilon)
		rstd = tf.rsqrt(var + epsilon)
		h_reshape = (h_reshape - mean) * rstd
		# reshape back to original
		h = tf.reshape(h_reshape, [batch_size, base * num_units])
	with tf.variable_scope(scope):
		if reuse == True:
			tf.get_variable_scope().reuse_variables()
			gamma = tf.get_variable('ln_gamma', [base * num_units], initializer = tf.constant_initializer(gamma_start))
		if use_bias:
			beta = tf.get_variable('ln_beta', [base * num_units], initializer = tf.constant_initializer(0.0))
		if use_bias:
			return gamma * h + beta
		return gamma * h

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

	U	= rotation_components(v1, v2)
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
	"""
	De-noising Rotational Unit of Memory
	TODO: add info about the model 
	"""

	def __init__(self,
				 num_units,
				 activation = None,
				 T_norm = None, 
				 eps = 1e-12,
				 use_zoneout = False, 
				 zoneout_keep_h = 0.9,
				 is_training = False, 
				 use_layer_norm = False
				 ):
		self.num_units = num_units
		self.activation = activation or relu 
		self.T_norm = T_norm
		self.eps = eps 
		self.use_zoneout	= use_zoneout
		self.zoneout_keep_h = zoneout_keep_h
		self.is_training = is_training
		self.use_layer_norm = use_layer_norm 
		

		# self, num_units, f_bias=1.0, use_zoneout=False, zoneout_keep_h = 0.9, zoneout_keep_c = 0.5, is_training = False
		# ):
		"""Initialize the Layer Norm LSTM cell.
		Args:
			num_units: int, The number of units in the LSTM cell.
			forget_bias: float, The bias added to forget gates (default 1.0).
			use_recurrent_dropout: float, Whether to use Recurrent Dropout (default False)
			dropout_keep_prob: float, dropout keep probability (default 0.90)
		"""
		# self.num_units = num_units
		# self.f_bias = f_bias

		# self.use_zoneout	= use_zoneout
		# self.zoneout_keep_h = zoneout_keep_h
		# self.zoneout_keep_c = zoneout_keep_c

		# self.is_training = is_training


	@property
	def state_size(self):
		return self._hidden_size
	
	@property
	def output_size(self):
		return self._hidden_size

	def call(self, inputs, state):
		"""De-noising Rotational Unit of Memory (DRUM)"""
		with vs.variable_scope("gates"): 
			batch_size = inputs.get_shape().as_list()[0]
			x_size = inputs.get_shape().as_list()[1]
			w_init = None #uniform initializer 
			h_init = drum_ortho_initializer(1.0)
			W_xh = tf.get_variable('W_xh', [x_size, 2 * self._hidden_size], initializer = w_init)
			W_hh = tf.get_variable('W_hh', [self._hidden_size, 2 * self._hidden_size], initializer = h_init)
			bias = tf.get_variable('bias_drum', [2 * self._hidden_size], initializer = tf.constant_initializer(0.0))
			xh = tf.matmul(inputs, W_xh)
			hh = tf.matmul(state, W_hh)
			ux, rx = tf.split(xh, 2, 1)
			uh, rh = tf.split(hh, 2, 1)
			ub, rb = tf.split(bias, 2, 0) #broadcasting the bias 
			u = sigmoid(ux + uh + ub)
			r = sigmoid(rx + rh + rb)
			if self._use_layer_norm: 
				concat = tf.concat([u, r], 1)
				concat = layer_norm_all(concat, batch_size, 2, self._hidden_size, 'ln_all')
				u, r = tf.split(concat, 2, 1)
		with vs.variable_scope("candidate"):
			W_xh_mixed = tf.get_variable('W_xh_mixed', [x_size, self._hidden_size], initializer = w_init)
			x_mixed = tf.matmul(inputs, W_xh_mixed)
			state_new = rotate(x_mixed, r, state)
			c = self._activation(x_mixed + state_new)
			if self._dropout_keep_prob != None: 
				tf.nn.dropout(c, self._dropout_keep_prob)
		new_h = u * state + (1 - u) * c
		if self._normalization != None: 
			new_h = tf.nn.l2_normalize(new_h, 1, epsilon = self._eps) * self._normalization 
		return new_h, new_h