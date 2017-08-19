import numpy as np
import tensorflow as tf
import random
from collections import deque

class Replay_Buffer:
	def __init__(self, size, batch_size):
		self.size = size
		self.batch_size = batch_size
		self.pointer = 0
		self.buffer = deque()

	def insert(self, state, action, reward, next_state, done):
		experience = (state, action, reward, next_state, done)

		# Not full yet
		if self.pointer < self.size:
			# deque.append : add to right side
			self.buffer.append(experience)
			self.pointer += 1
		# When buffer is full
		else:
			# deque.popleft : take leftmost value and remove it from queue
			deque.popleft()
			deque.append(experience)
		print('Added experience')


	def get_batches(self):
		# Get random batches from queue as type 'list'
		batches = random.sample(self.buffer, self.batch_size)
		return batches


	@property
	def get_size(self):
		if self.pointer < self.size:
			return self.pointer
		else:
			return self.size

 
