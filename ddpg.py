import tensorflow as tf
import numpy as np
from actor import Actor
from critic import Critic
from replay_buffer import Replay_Buffer
from ou_noise import OU_noise
import utils

class DDPG:
	def __init__(self, args, sess, env):
		self.args = args
		self.sess = sess
		self.env = env

		# Get number of states
		self.state_dim = self.env.observation_space.shape[0]
		# Get number of actions
		self.action_dim = self.env.action_space.shape[0]
		# Get maximum steps per episode
		self.step_per_episode = self.env.timestep_limit

		self.actor_network = Actor(self.args, self.sess, self.state_dim, self.action_dim)
		self.critic_network = Critic(self.args, self.sess, self.state_dim, self.action_dim)

		# Initalize replay buffer
		self.replay_buffer = Replay_Buffer()
		self.exploration = OU_noise()

		
	def train(self):
		# Count training step
		self.steps = 0
		# Receive initial observation
		print('Reset game')
		self.reset_statistics()
	
		utils.initialize_log()
		
		for episode in xrange(self.args.num_episodes):
			print('%d episode starts' % episode+1) 
			# Initial observation
			observation = self.env.reset()
			for step in xrange(self.step_per_episode):
				# Select action according to the current policy and exploration noise
				action = self.get_action()
				current_observation = observation
				observation, reward, done, _ = self.env.step(action)
			
				self.replay_buffer.insert()

				if self.replay_buffer.get_size() > self.args.training_start:
					self.steps += 1
					print('%d training steps' % self.steps)
					batch_s, batch_act, batch_rwd, batch_done, batch_next_s = self.replay_buffer.get_batches()
					if self.args.bn:
						# Get next action from actor target network
						batch_next_action = self.sess.run(self.actor_network.target_layer3_out, feed_dict={self.actor_network.target_states:batch_next_s, self.target_is_training:False})
						# Get target q
						batch_target_q = self.sess.run(self.critic_network.target_layer3_out, feed_dict={self.target_states:batch_next_s, self.target_actions:batch_next_action, self.target_is_training:True}
						# Set target and update critic by minimizing the loss
						feed_dict = {self.critic_network.rewards:batch_rwd, self.critic_network.target_q:batch_target_q, self.critic_network.is_training:True}
						cost_, action_gradient, _ = self.sess.run([self.critic_network.cost, self.critic_network.gradients, self.critic_network.optimizer])
						# Update the actor policy gradient using the sampled policy gradient
						feed_dict = {self.actor_network.states:current_observation, self.actor_network.q_action_gradient:action_gradient, self.actor_is_training:True}
						self.sess.run(self.actor_network.optimizer, feed_dict=feed_dict)
					else:
						batch_next_action = self.sess.run(self.action_network.target_layer3_out, feed_dict={self.actor_network.target_states:batch_next_s})
						batch_target_q = self.sess.run(self.critic_network.target_layer3_out, feed_dict={self.target_states:batch_next_s, self.target_actions:batch_next_action)

						# Set target(y) and update critic by minimizing the loss
						feed_dict = {self.critic_network.rewards:batch_rwd, self.critic_network.target_q:batch_target_q}
						cost_, action_gradient, _ = self.sess.run([self.critic_network.cost, self.critic_network.gradients, self.critic_network.optimizer])
						# Update the actor policy gradient using the sampled policy gradient
						feed_dict = {self.actor_network.states:current_observation, self.actor_network.q_action_gradient:action_gradient}
						self.sess.run(self.actor_network.optimizer, feed_dict=feed_dict)


				self.cost_per_episode += cost_
				self.reward_per_episode += reward
				# Update target network
				self.actor_network.update_target()
				self.critic_network.update_target()

				if done:
					print('%d episode end' % episode+1)
					self.exploration.reset()
					self.total_reward += self.reward_per_episode
					utils.write_logs(episode+1, self.reward__per_episode, self.cost_per_episode    )
					self.cost_per_episode = 0
					self.reward_per_episode = 0
					break


	def get_action(self):


	def initialize_statistics(self):
		self.reward_per_episode = 0
		self.total_reward = 0
		self.cost_per_episode = 0



	@property
	def model_dir(self):



	def save(self, global_step):
	


	def load(self):		


