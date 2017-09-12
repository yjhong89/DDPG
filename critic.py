import numpy as np
import tensorflow as tf
import math
from batch_norm import batch_wrapper

'''
	Almost same with Actor.py
	Differs from using L2 weight decay and action is included as inputs
'''

class Critic():
	def __init__(self, args, sess, state_dim, action_dim):
		print('Initializing critic network')
		self.args = args
		self.sess = sess
		self.state_dim = state_dim
		self.action_dim = action_dim 

		self.states = tf.placeholder(tf.float32, [None, self.state_dim])
		self.actions = tf.placeholder(tf.float32, [None, self.action_dim])
		self.rewards = tf.placeholder(tf.float32, [None])
		self.done = tf.placeholder(tf.float32, [None])

		with tf.variable_scope('Critic'):
			# Initialized from uniform distributions [-1/root(f), 1/root(f)] where f is fan-in
			weight1 = tf.Variable(tf.random_uniform((self.state_dim, self.args.layer1), -1/math.sqrt(self.state_dim), 1/math.sqrt(self.state_dim)), name='Weigh1')
			bias1 = tf.Variable(tf.random_uniform([self.args.layer1], -1e-3, 1e-3), name='Bias1')
			# Action included
			weight2 = tf.Variable(tf.random_uniform((self.args.layer1, self.args.layer2), -1/math.sqrt(self.args.layer1+self.action_dim), 1/math.sqrt(self.args.layer1+self.action_dim)), name='Weight2')
			weight2_action = tf.Variable(tf.random_uniform((self.action_dim, self.args.layer2), -1/math.sqrt(self.args.layer1+self.action_dim), 1/math.sqrt(self.args.layer1+self.action_dim)), name='Weight2_A')
			bias2 = tf.Variable(tf.random_uniform([self.args.layer2], -1e-3, 1e-3), name='Bias2')
			# Output : (1,) shape representing Q value
			weight3 = tf.Variable(tf.random_uniform((self.args.layer2, 1), -3e-3, 3e-3), name='Weight3')
			bias3 = tf.Variable(tf.random_uniform([1], -1e-3, 1e-3), name='Bias3')

		variable_list = [weight1, bias1, weight2, weight2_action, bias2, weight3, bias3]
		for i in variable_list:
			print(i.op.name)

		if self.args.bn:
			self.is_training = tf.placeholder(tf.bool)
			
			self.layer1_bn = batch_wrapper(tf.matmul(self.states, weight1) + bias1, self.is_training, name='Critic_BN1')
			layer1_out = tf.nn.relu(self.layer1_bn.do_bn)
			# Use batch normalization on all layers of the Q network prior to the action input
			layer2_out = tf.nn.relu(tf.matmul(layer1_out, weight2) + tf.matmul(self.actions, weight2_action) + bias2)
			self.layer3_out = tf.matmul(layer2_out, weight3) + bias3
			self.target_states, self.target_actions, self.target_layer3_out, self.target_soft_update, self.target_is_training = \
				self.create_target_network(variable_list)

		else:
			layer1_out = tf.nn.relu(tf.matmul(self.states, weight1) + bias1)
			layer2_out = tf.nn.relu(tf.matmul(layer1_out, weight2) + tf.matmul(self.actions, weight2_action) + bias2)
			# Outputs q value, do not use nonlinearty
			self.layer3_out = tf.matmul(layer2_out, weight3) + bias3
			self.target_states, self.target_actions, self.target_layer3_out, self.target_soft_update = self.create_target_network(variable_list)

		self.target_q = tf.placeholder(tf.float32, [None])
		# Need to match shape with 'self.layer3_out'
		self.target = tf.expand_dims(self.rewards + tf.multiply(1-self.done, self.args.gamma*self.target_q), 1)
		# Include l2 regularization term
		self.l2_decay = 0
		for i in variable_list:
			# tf.nn.l2_loss returns 0-D tensor
			# sum(input**2)/2
			self.l2_decay += tf.nn.l2_loss(i)
		self.l2_decay *= self.args.regularize_decay
		self.cost = tf.reduce_mean(tf.pow(self.target - self.layer3_out, 2)) + self.l2_decay
		self.optimizer = tf.train.AdamOptimizer(self.args.critic_lr).minimize(self.cost)
		# To feed actor, get gradient with respect to action input
		# Will be [batch size, num actions]
		self.gradients = tf.gradients(self.layer3_out, self.actions)

		self.sess.run(tf.global_variables_initializer())
		self.update_target()
		

	def create_target_network(self, variable_list):
		print('Creating critic target network')
		states = tf.placeholder(tf.float32, [None, self.state_dim])
		actions = tf.placeholder(tf.float32, [None, self.action_dim])
		ema = tf.train.ExponentialMovingAverage(decay=1-self.args.tau)
		soft_update = ema.apply(variable_list)
		target_variable = [ema.average(i) for i in variable_list]

		if self.args.bn:
			is_training = tf.placeholder(tf.bool)
			self.target_layer1_bn = batch_wrapper(tf.matmul(states, target_variable[0]) + target_variable[1], is_training, tau=self.args.tau, target=self.layer1_bn, name='Critic_Target_BN1')
			layer1_out = tf.nn.relu(self.layer1_bn.do_bn)
			layer2_out = tf.nn.relu(tf.matmul(layer1_out, target_variable[2]) + tf.matmul(actions, target_variable[3]) + target_variable[4])
			layer3_out = tf.matmul(layer2_out, target_variable[5]) + target_variable[6]
			return states, actions, layer3_out, soft_update, is_training

		else:
			layer1_out = tf.nn.relu(tf.matmul(states, target_variable[0]) + target_variable[1])
			layer2_out = tf.nn.relu(tf.matmul(layer1_out, target_variable[2]) + tf.matmul(actions, target_variable[3]) + target_variable[4])
			layer3_out = tf.matmul(layer2_out, target_variable[5]) + target_variable[6]
			return states, actions, layer3_out, soft_update


	def update_target(self):
		if self.args.bn:
			self.sess.run([self.target_soft_update, self.target_layer1_bn.update])
		else:
			self.sess.run(self.target_soft_update)
		#print('Update target critic network')
	

