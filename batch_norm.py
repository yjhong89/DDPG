import tensorflow as tf


class batch_wrapper():
	def __init__(self, inputs, is_training, tau=None, target=None, name=None, decay=0.99):
		self.tau = tau
		with tf.variable_scope(name or 'bn'):
			self.scale = tf.get_variable('Scale', [inputs.get_shape()[-1]], initializer=tf.constant_initializer(1))
			self.beta = tf.get_variable('Beta', [inputs.get_shape()[-1]], initializer=tf.constant_initializer(0))
			# Directly assing to population mean and variance by exponential moving average
			self.pop_mean = tf.get_variable('Pop_mean', [inputs.get_shape()[-1]], trainable=False, initializer=tf.constant_initializer(1))
			self.pop_var = tf.get_variable('Pop_var', [inputs.get_shape()[-1]], trainable=False, initializer=tf.constant_initializer(0))
			# Those variables must be declared in __init__ for when train=False in the first time
			self.batch_mean, self.batch_var = tf.nn.moments(inputs, [0])
			self.train_mean = tf.assign(self.pop_mean, self.pop_mean*decay + self.batch_mean*(1-decay))
			self.train_var = tf.assign(self.pop_var, self.pop_var*decay + self.batch_var*(1-decay))

		def training():
			with tf.control_dependencies([self.train_mean, self.train_var]):
				# Use batch mean and var when training
				return tf.nn.batch_normalization(inputs, self.batch_mean, self.batch_var, self.beta, self.scale, variance_epsilon=1e-6)

		def test():
			# Use population mean and variance when testing
			return tf.nn.batch_normalization(inputs, self.train_mean, self.train_var, self.beta, self.scale, variance_epsilon=1e-6)

		# Soft update on batch noramalization variables
		if target is not None:
			self.update_target(target)

		self.do_bn = tf.cond(is_training, training, test)


	def update_target(self, target):
		if self.tau is not None:
			update_scale = self.scale.assign(self.scale*(1-self.tau)+target.scale*self.tau)
			update_beta = self.beta.assign(self.beta*(1-self.tau)+target.beta*self.tau)
			# tf.group returns an operation that executes all its input
			self.update = tf.group(update_scale, update_beta)
		else:
			raise Exception('Need tau')

	
	
