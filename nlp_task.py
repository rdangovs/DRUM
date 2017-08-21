
from __future__ import absolute_import
from __future__ import division
from __future__	import print_function

import numpy as np
import argparse, os
import tensorflow as tf
from tensorflow.python.ops import init_ops
from tensorflow.contrib.rnn import BasicLSTMCell, BasicRNNCell, GRUCell, LSTMStateTuple
from EURNN import EURNNCell
from GORU import GORUCell
from drum import DRUMCell
from rotational_models import GRRUCell

from ptb_iterator import *
import re

word_level = False
notstates = False

def file_data(stage, n_batch, n_data, T, n_epochs, vocab_to_idx,readIntegers=True):

	if not word_level:
		if stage == 'train':
			file_name = 'data/ptb.char.train.txt'
		elif stage == 'valid':
			file_name = 'data/ptb.char.valid.txt'	
		elif stage == 'test':
			file_name = 'data/ptb.char.test.txt'
	else:
		if stage == 'train':
			file_name = 'data/ptb.train_words_nodigits_new.txt'
		elif stage == 'valid':
			file_name = 'data/ptb.valid_words_nodigits_new.txt'	
		elif stage == 'test':
			file_name = 'data/ptb.test_words_nodigits_new.txt'

	with open(file_name,'r') as f:
		raw_data = f.read()
		print("Data length: " , len(raw_data))

	if word_level:
		if readIntegers:
			raw_data = [int(line.rstrip('\n')) for line in open(file_name)]
	
	raw_data = raw_data.replace('\n', '')
	raw_data = raw_data.replace(' ', '')


	if vocab_to_idx == None:
		vocab = set(raw_data)
		vocab_size = len(vocab)
		print("Vocab size: ", vocab_size)

		my_dict = {}
		idx_to_vocab= {}
		vocab_to_idx = {}
		for index, item in enumerate(vocab):
			idx_to_vocab[index] = item
			vocab_to_idx[item] = index
	data = [vocab_to_idx[c] for c in raw_data][:n_data]
	print("Total data length: " , len(data))

	# print(data[0:1000])

	#Deal with batch size >.<

	#Numsteps is your sequence length. In this case the earlier formula. 
	def gen_epochs(n, numsteps, n_batch):
		for i in range(n):
			yield ptb_iterator(data, n_batch, numsteps)
	
	print("Sequence length: ", T)
	myepochs = gen_epochs(n_epochs, T, n_batch)
	print(myepochs)

	return myepochs, vocab_to_idx

# file_data('train', 20, 10000000, 50, 20)

def main(model, T, n_epochs, n_batch, n_hidden, capacity, comp, FFT, learning_rate, decay, nb_v, norm,
	 dynamic_norm):
	# --- Set data params ----------------
	#Create Data
	max_len_data = 1000000000
	epoch_train, vocab_to_idx = file_data('train', n_batch, max_len_data, T, n_epochs, None)
	n_input = len(vocab_to_idx)
	epoch_val, _ = file_data('valid', nb_v, max_len_data, T, 10000, vocab_to_idx)
	epoch_test, _ = file_data('test', nb_v, max_len_data, T, 1, vocab_to_idx)
	n_output = n_input


	# --- Create graph and compute gradients ----------------------
	x = tf.placeholder("int32", [None, T])
	y = tf.placeholder("int64", [None, T])
	if model == "LSTM":
		i_s = LSTMStateTuple(tf.placeholder("float", [None, n_hidden]), tf.placeholder("float", [None, n_hidden]))
	else:
		i_s = tf.placeholder("float", [None, n_hidden])
	
	n_embed = 30
	if (word_level):
		embed_init_val = np.sqrt(6.)/np.sqrt(n_input)
		embed = tf.get_variable('Embedding', [n_input, n_embed] ,initializer = init_ops.random_normal_initializer(-embed_init_val, embed_init_val), dtype=tf.float32)
		input_data = tf.nn.embedding_lookup(embed, x)
		n_input = n_embed

	else:	
		input_data = tf.one_hot(x, n_input, dtype=tf.float32)


	# Input to hidden layer
	cell = None
	h = None
	#h_b = None
	if model == "DRUM":
		if norm != None: 
			cell = DRUMCell(n_hidden, normalization = norm)
		else: 
			cell = DRUMCell(n_hidden)
		hidden_out, states = tf.nn.dynamic_rnn(cell, input_data, dtype=tf.float32, 
							initial_state = i_s)
	if model == "GRRU":
		cell = GRRUCell(n_hidden)
		if h == None:
			h = cell.zero_state(n_batch,tf.float32)
		hidden_out, states = tf.nn.dynamic_rnn(cell, input_data, dtype=tf.float32)
	if model == "LSTM":
		cell = BasicLSTMCell(n_hidden, state_is_tuple=True, forget_bias=1)
		if h == None:
			h = cell.zero_state(n_batch,tf.float32)
		hidden_out, states = tf.nn.dynamic_rnn(cell, input_data, dtype=tf.float32)
	elif model == "GRU":
		cell = GRUCell(n_hidden)
		if h == None:
			h = cell.zero_state(n_batch,tf.float32)
		hidden_out, states = tf.nn.dynamic_rnn(cell, input_data, dtype=tf.float32)
	elif model == "RNN":
		cell = BasicRNNCell(n_hidden)
		if h == None:
			h = cell.zero_state(n_batch,tf.float32)
		hidden_out, states = tf.nn.dynamic_rnn(cell, input_data, dtype=tf.float32)
	elif model == "EURNN":
		cell = EURNNCell(n_hidden, capacity, FFT, comp)
		if h == None:
			h = cell.zero_state(n_batch,tf.float32)
		if comp:
			hidden_out_comp, states = tf.nn.dynamic_rnn(cell, input_data, dtype=tf.complex64)
			hidden_out = tf.real(hidden_out_comp)
		else:
			hidden_out, states = tf.nn.dynamic_rnn(cell, input_data, dtype=tf.float32)
	elif model == "GORU":
		cell = GORUCell(n_hidden, capacity, FFT, comp)
		if h == None:
			h = cell.zero_state(n_batch,tf.float32)
		if comp:
			hidden_out_comp, states = tf.nn.dynamic_rnn(cell, input_data, dtype=tf.complex64)
			hidden_out = tf.real(hidden_out_comp)
		else:
			hidden_out, states = tf.nn.dynamic_rnn(cell, input_data, dtype=tf.float32)
	


	# Hidden Layer to Output
	V_init_val = np.sqrt(6.)/np.sqrt(n_output + n_input)

	V_weights = tf.get_variable("V_weights", shape = [n_hidden, n_output], \
			dtype=tf.float32, initializer=tf.random_uniform_initializer(-V_init_val, V_init_val))
	V_bias = tf.get_variable("V_bias", shape=[n_output], \
			dtype=tf.float32, initializer=tf.constant_initializer(0.01))

	hidden_out_list = tf.unstack(hidden_out, axis=1)
	temp_out = tf.stack([tf.matmul(i, V_weights) for i in hidden_out_list])
	output_data = tf.nn.bias_add(tf.transpose(temp_out, [1,0,2]), V_bias) 


	# define evaluate process
	cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=output_data, labels=y))
	correct_pred = tf.equal(tf.argmax(output_data, 2), y)
	accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


	# --- Initialization ----------------------
	optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate, decay=decay).minimize(cost)
	init = tf.global_variables_initializer()

	for i in tf.global_variables():
		print(i.name)

	# --- save result ----------------------
	filename = "./output/character/T=" + str(T) + "/normalized_" + model  + "_N=" + str(n_hidden) + "nb_v" + str(nb_v) + "_norm=" + str(norm) + "_numEpochs=" + str(n_epochs)#"_lambda=" + str(learning_rate) + "_beta=" + str(decay)
		
	if model == "EURNN"  or model == "GORU":
		print(model)
		if FFT:
			filename += "_FFT"
		else:
			filename = filename + "_L=" + str(capacity)

	filename = filename + ".txt"
	if not os.path.exists(os.path.dirname(filename)):
		try:
			os.makedirs(os.path.dirname(filename))
		except OSError as exc: # Guard against race condition
			if exc.errno != errno.EEXIST:
				raise
	f = open(filename, 'w')
	f.write("########\n\n")
	f.write("## \tModel: %s with N=%d"%(model, n_hidden))
	if model == "EURNN" or model == "GORU":
		if FFT:
			f.write(" FFT")
		else:
			f.write(" L=%d"%(capacity))
	f.write("\n\n")
	f.write("########\n\n")


	

	def do_validation():
		j = 0
		val_losses = []
		val_max = 0
		val_norm_max = 0
		for val in epoch_val:
			j +=1 
			if j >= 2:
				break
			print("Running validation...")
			if model == "LSTM":
				val_state = LSTMStateTuple(np.zeros((nb_v, n_hidden), dtype = np.float), np.zeros((1, n_hidden), dtype = np.float))
			else:
				val_state = np.zeros((nb_v, n_hidden), dtype = np.float)
			for stepb, (X_val,Y_val) in enumerate(val):
				val_batch_x = X_val
				val_batch_y = Y_val
				val_dict = {x:val_batch_x,y:val_batch_y,i_s:val_state}
				if notstates:
					val_acc,val_loss = sess.run([accuracy,cost],feed_dict=val_dict)
				else:
					val_acc, val_loss, val_state = sess.run([accuracy, cost,states],feed_dict=val_dict)
					val_max = max(val_max, np.max(val_state))
					val_norm_max = max(val_norm_max, np.sqrt(np.sum([i**2 for i in val_state[0]])))
				val_losses.append(val_loss)
		print("Validations:", )
		validation_losses.append(sum(val_losses)/len(val_losses))
		print("Validation Loss= " + \
				  "{:.6f}".format(validation_losses[-1]))
		
		f.write("Step: %d\t Loss: %f\t Max. val. of state: %f\t Max. norm of state: %f\n"%(t, validation_losses[-1],val_max,val_norm_max))
		f.flush()

	# saver = tf.train.Saver()

	step = 0
	with tf.Session(config=tf.ConfigProto(log_device_placement=False, allow_soft_placement=False)) as sess:
		print("Session Created")

		# if loadFrom != "":
		# 	new_saver = tf.train.import_meta_graph(loadFrom+'.meta')
		# 	new_saver.restore(sess, tf.train.latest_checkpoint('./'))
		# 	print("Session loaded from: " , loadFrom)
		# else:
		# 	#summary_writer = tf.train.SummaryWriter('/tmp/logdir', sess.graph)
		# 	sess.run(init)
		

		steps = []
		losses = []
		accs = []
		validation_losses = []

		sess.run(init)
		if model == "LSTM":
			training_state = LSTMStateTuple(np.zeros((n_batch, n_hidden), dtype = np.float), np.zeros((n_batch, n_hidden), dtype = np.float))
		else:
			training_state = np.zeros((n_batch, n_hidden), dtype = np.float)
		i = 0
		t = 0
		for epoch in epoch_train:
			print("Epoch: " , i)

			for step, (X,Y) in enumerate(epoch):
				batch_x = X
				batch_y = Y
				myfeed_dict={x: batch_x, y: batch_y, i_s: training_state}
				if notstates:
					_, acc, loss = sess.run([optimizer, accuracy, cost], feed_dict = myfeed_dict)
				else:
					empty,acc,loss,training_state = sess.run([optimizer, accuracy, cost, states], feed_dict = myfeed_dict)
				print(np.max(training_state))
				print(np.sqrt(np.sum([i**2 for i in training_state[0]])))

				print("Iter " + str(step) + ", Minibatch Loss= " + \
					  "{:.6f}".format(loss) + ", Training Accuracy= " + \
				  	  "{:.5f}".format(acc))
				
				steps.append(t)
				losses.append(loss)
				accs.append(acc)
				t += 1


				if step % 1000 == 999:
					do_validation()
					# saver.save(sess,savename)
					#Now I need to take an epoch and go through it. I will average the losses at the end
						# f2.write("%d\t%f\t%f\n"%(step, loss, acc))
					# f.flush()
					# f2.flush()
				# mystates = sess.run(states, feed_dict=myfeed_dict)
				# print ("States",training_state)

			i += 1

		print("Optimization Finished!")
		
		


		j = 0
		test_losses = []
		for test in epoch_test:
			j +=1 
			if j >= 2:
				break
			print("Running validation...")
			if model == "LSTM":
				test_state = LSTMStateTuple(np.zeros((nb_v, n_hidden), dtype = np.float), np.zeros((nb_v, n_hidden), dtype = np.float))
			else:
				test_state = np.zeros((nb_v, n_hidden), dtype = np.float)
			for stepb, (X_test,Y_test) in enumerate(test):
				test_batch_x = X_test
				test_batch_y = Y_test
				test_dict = {x:test_batch_x,y:test_batch_y,i_s:test_state}
				test_acc, test_loss, test_state = sess.run([accuracy, cost,states],feed_dict=test_dict)
				test_losses.append(test_loss)
		print("test:", )
		test_losses.append(sum(test_losses)/len(test_losses))
		print("test Loss= " + \
				  "{:.6f}".format(test_losses[-1]))
		f.write("Test result: %d (step) \t%f (loss)\n"%(t, test_losses[-1]))

		


if __name__=="__main__":
	parser = argparse.ArgumentParser(
		description="Copying Memory Problem")

	parser.add_argument("model", default='LSTM', help='Model name: LSTM, EURNN, GRU, GORU')
	parser.add_argument('-T', type=int, default=50, help='T-gram')
	parser.add_argument("--n_epochs", '-E', type=int, default=20, help='num epochs')
	parser.add_argument('--n_batch', '-B', type=int, default=32, help='batch size')
	parser.add_argument('--n_hidden', '-H', type=int, default=512, help='hidden layer size')
	parser.add_argument('--capacity', '-L', type=int, default=2, help='Tunable style capacity, only for EURNN, default value is 2')
	parser.add_argument('--comp', '-C', type=str, default="False", help='Complex domain or Real domain. Default is False: real domain')
	parser.add_argument('--FFT', '-F', type=str, default="False", help='FFT style, default is False')
	parser.add_argument('--learning_rate', '-R', default=0.001, type=float)
	parser.add_argument('--decay', '-D', default=0.9, type=float)
	parser.add_argument('--nb_v', '-nbv', default=32, type=int)
	parser.add_argument('--norm', '-norm', default=None, type=float)
	parser.add_argument('--dynamic_norm', '-d_norm', default=None, type=str, help = 'type of norm dynamics: none, growth, decay')
	# parser.add_argument("--model_save_to", type=str, default="my-model", help='Name to save the file to')
	# parser.add_argument("--model_load_from", type=str, default="", help='Name to load the model from')
	# parser.add_argument("--num_layers", type=int, default=1, help='Int: Number of layers (1)')



	args = parser.parse_args()
	dict = vars(args)

	for i in dict:
		if (dict[i]=="False"):
			dict[i] = False
		elif dict[i]=="True":
			dict[i] = True
		
	kwargs = {	
				'model': dict['model'],
				'T':dict['T'],
				'n_epochs': dict['n_epochs'],
			  	'n_batch': dict['n_batch'],
			  	'n_hidden': dict['n_hidden'],
			  	'capacity': dict['capacity'],
			  	'comp': dict['comp'],
			  	'FFT': dict['FFT'],
			  	'learning_rate': dict['learning_rate'],
			  	'decay': dict['decay'],
				'nb_v': dict['nb_v'], 
				'norm': dict['norm'],
				'dynamic_norm': dict['dynamic_norm']
			}
	print(kwargs)
	main(**kwargs)
