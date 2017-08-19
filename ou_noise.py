import numpy as np

'''
	Reference : https://github.com/rllab
'''


class OU_noise:
	def __init__(self, action_dimension, mu=0, theta=0.15, sigma=0.2):
		self.action_dim = action_dimension
		self.mu = mu
		self.theta = theta
		self.sigma = sigma
		self.result = np.ones(self.action_dim) * self.mu
		self.reset()

	def reset(self):
		self.result = np.ones(self.action_dim) * self.mu
		
	def noise(self):
		x = self.result
		dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
		self.result = x + dx
		#print(self.result)
		return self.result
