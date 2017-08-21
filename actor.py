import numpy as np
import tensorflow as tf
import math
from batch_norm import batch_wrapper

class Actor():
	def __init__(self, args, sess, state_dim, action_dim):
		print('Initializing actor network')
		self.args = args
		self.sess = sess
		self.state_dim = state_dim
		self.action_dim = action_dim

		self.states = tf.placeholder(tf.float32, [None, self.state_dim])

		with tf.variable_scope('Actor'):
			# Initialized from unifom distributions [-1/root(f), 1/root(f)] where f is fan-in
			weight_layer1 = tf.Variable(tf.random_uniform([self.state_dim, self.args.layer1], -1/math.sqrt(self.state_dim), 1/math.sqrt(self.state_dim)), name='Weight1') 		
			bias_layer1 = tf.Variable(tf.random_uniform([self.args.layer1], -1e-3, 1e-3), name='Bias1')
			weight_layer2 = tf.Variable(tf.random_uniform([self.args.layer1, self.args.layer2], -1/math.sqrt(self.args.layer1), 1/math.sqrt(self.args.layer1)), name='Weight2')
			bias_layer2 = tf.Variable(tf.random_uniform([self.args.layer2], -1e-3, 1e-3), name='Bias2')
			# Final layer variables are initialized from uniform distribution [-0.003, 0.003] 
			# To ensure initial output to be zero
			weight_layer3 = tf.Variable(tf.random_uniform([self.args.layer2, self.action_dim], -3e-3, 3e-3), name='Weight3')
			bias_layer3 = tf.Variable(tf.random_uniform([self.action_dim], -1e-3, 1e-3), name='Bias3')

#		tr_vrbs = tf.trainable_variables()
		# Bias1, Bias2, Bias3, Weight1, Weigh2, Weight3
#		variable_list = sorted(tr_vrbs, key=lambda x: x.op.name)
		variable_list = [weight_layer1, bias_layer1, weight_layer2, bias_layer2, weight_layer3, bias_layer3]		
		for i in variable_list:
			print(i.op.name)

		if self.args.bn:
			self.is_training = tf.placeholder(tf.bool)

			self.layer1_bn = batch_wrapper(tf.matmul(self.states, weight_layer1) + bias_layer1, self.is_training, name='BN1')
			layer1_out = tf.nn.relu(self.layer1_bn.do_bn)
			self.layer2_bn = batch_wrapper(tf.matmul(layer1_out, weight_layer2) + bias_layer2, self.is_training, name='BN2')
			layer2_out = tf.nn.relu(self.layer2_bn.do_bn)
			
			self.layer3_bn = batch_Wrapper(tf.matmul(layer2_out, weight_layer3) + bias_layer3, self.is_training, name='BN3')
			self.layer3_out = tf.tanh(self.layer3_bn.do_bn)

			self.target_states, self.target_layer3_out, self.target_soft_update, self.target_is_training = \
				self.create_target_network(variable_list)

		else:
			layer1_out = tf.nn.relu(tf.matmul(self.states, weight_layer1) + bias_layer1)
			layer2_out = tf.nn.relu(tf.matmul(layer1_out, weight_layer2) + bias_layer2)
			# Use tanh layer to bound the actions
			self.layer3_out = tf.tanh(tf.matmul(layer2_out, weight_layer3) + bias_layer3)
			self.target_states, self.target_layer3_out, self.target_soft_update = self.create_target_network(variable_list)

			

		# Get 'q' action gradient computed in critic network
		self.q_action_gradient = tf.placeholder(tf.float32, [None, self.action_dim])
		# self.layer3_out : [None, self.action_dim] 
		'''
			tf.gradients(ys, xs, grad_ys=None)
			grad_ys : Optional. A tensor or list of tensors the same size as 'ys' and
					holding the initial gradients for each y in 'ys'
			Simply, holding gradient of composition function	
			Returns list of derivatives for each x in xs
		'''
		# Since q_action_gradient will have negative direction to minimize value estimate loss
		self.gradients = tf.gradients(self.layer3_out, variable_list, -self.q_action_gradient)
		self.optimizer = tf.train.AdamOptimizer(self.args.actor_lr).apply_gradients(zip(self.gradients, variable_list)) 

		self.sess.run(tf.global_variables_initializer())

		# Initialize exponential moving average to initialize target network
		'''
			ExponentialMovingAverage.apply() method has to be called to create
			shadow variables and add ops to maintain moving average
			The shadow variables are initialized with the same initial values as the trainable variable
		'''
		self.update_target()


	# Using target network by soft target update
	# Use target network to calculate target value
	'''
		Rather than directly copying the weights, create a copy of network and update target network by
		slowly tracking using exponential moving average
		This method greatly improves the stability of learning
	'''
	def create_target_network(self, variable_list):
		
		states = tf.placeholder(tf.float32, [None, self.state_dim])
		ema = tf.train.ExponentialMovingAverage(decay=1-self.args.tau)
		# Maintains moving averages of variables through Shadow variables(update shadow variables)
		soft_update = ema.apply(variable_list)
		# Soft updated values of variable
		# returns variable holding the average
		target_variable = [ema.average(i) for i in variable_list]
		
		if self.args.bn:
			is_training = tf.placeholder(tf.bool)
			
			self.target_layer1_bn = batch_wrapper(tf.matmul(states, target_variable[0]) + target_variable[1], is_training, tau=self.args.tau, target=self.layer1_bn, name='Target_BN1')
			layer1_out = tf.nn.relu(layer1_bn)
			self.target_layer2_bn = batch_wrapper(tf.matmul(layer1_out, target_variable[2]) + target_variable[3], is_training, tau=selef.args.tau, target=self.layer2_bn,  name='Target_BN2')
			layer2_out = tf.nn.relu(layer2_bn)
			self.layer3_layer3_bn = batch_wrapper(tf.matmul(layer2_out, target_variable[4]) + target_variable[5], is_training, tau=self.args.tau, target=self.layer3_bn, name='Traget_BN3')
			layer3_out = tf.tanh(layer3_bn)

			return states, layer3_out, soft_update, is_training

		else:
			layer1_out = tf.nn.relu(tf.matmul(states, target_variable[0]) + target_variable[1])
			layer2_out = tf.nn.relu(tf.matmul(layer1_out, target_variable[2]) + target_variable[3])
			layer3_out = tf.tanh(tf.matmul(layer2_out, target_variable[4]) + target_variable[5])
			
			return states, layer3_out, soft_update
		
	def update_target(self):
		if self.args.bn:
			self.sess.run([self.target_soft_update, self.target_layer1_bn.update, self.target_layer2_bn.update, self.target_layer3_bn.update])
		else:
			self.sess.run(self.target_soft_update)
#		print('Update target actor network')


	
