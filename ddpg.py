import tensorflow as tf
import numpy as np
from actor import Actor
from critic import Critic
from replay_buffer import Replay_Buffer
from ou_noise import OU_noise
import utils
import os, time

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
		self.step_per_episode = self.env.spec.timestep_limit
		print('Number of actions : %d, Number of states : %d, Number of steps per episode : %d' % (self.action_dim, self.state_dim, self.step_per_episode))

		self.actor_network = Actor(self.args, self.sess, self.state_dim, self.action_dim)
		self.critic_network = Critic(self.args, self.sess, self.state_dim, self.action_dim)

		# Initalize replay buffer
		self.replay_buffer = Replay_Buffer(size=self.args.replay_size, batch_size=self.args.batch_size)
		self.exploration = OU_noise(self.action_dim)

		self.saver = tf.train.Saver()

		
	def train(self):
		# Count training step
		self.steps = 0
		start_time = time.time()
		# Receive initial observation
		self.initialize_statistics()
	
		utils.initialize_log()

		if self.args.monitor:
			self.env.monitor.start(self.args.env_name, force=True)
		
		for episode in xrange(self.args.num_episodes):
			print('%d episode starts' % (episode+1)) 
			# Initial observation
			observation = self.env.reset()
			for step in xrange(self.step_per_episode):
				# Select action according to the current policy and exploration noise
				action = self.get_action(observation)
				current_observation = observation
				observation, reward, done, _ = self.env.step(action)
			
				self.replay_buffer.insert(state=current_observation, action=action, reward=reward, next_state=observation, done=done)

				if self.replay_buffer.get_size > self.args.training_start:
					self.steps += 1
					print('%d training steps' % self.steps)
					# batch : [state, reward, next_state, done]
					batch = self.replay_buffer.get_batches()
					batch_s = np.asarray([batches[0] for batches in batch])
					batch_act = np.asarray([batches[1] for batches in batch])
					batch_rwd = np.asarray([batches[2] for batches in batch])
					batch_next_s = np.asarray([batches[3] for batches in batch])
					batch_done = np.asarray([batches[4] for batches in batch])

					if self.args.bn:
						# Get next action from actor target network
						batch_next_action = self.sess.run(self.actor_network.target_layer3_out, feed_dict={self.actor_network.target_states:batch_next_s, self.target_is_training:False})
						# Get target q
						batch_target_q = self.sess.run(self.critic_network.target_layer3_out, feed_dict={self.critic_network.target_states:batch_next_s, self.critic_network.target_actions:batch_next_action, self.target_is_training:False})

						# Set target and update critic by minimizing the loss
						feed_dict = {self.critic_network.states:batch_s, self.critic_network.actions:batch_act, self.critic_network.rewards:batch_rwd, self.critic_network.done:batch_done, self.critic_network.target_q:np.squeeze(batch_target_q), self.critic_network.is_training:True}
						cost_, action_gradient, _ = self.sess.run([self.critic_network.cost, self.critic_network.gradients, self.critic_network.optimizer], feed_dict=feed_dict)

						# Update the actor policy gradient using the sampled policy gradient
						action_batch_q_gradient = self.sess.run(self.actor_network.layer3_out, feed_dict={self.actor_network.states:batch_s, self.actor_network.is_training:False})[0]
						action_gradient = self.sess.run(self.critic_network.gradients, feed_dict={self.critic_network.states:batch_s, self.critic_network.actions:action_batch_q_gradient, self.critic_network.is_training:False})
						feed_dict = {self.actor_network.states:batch_s, self.actor_network.q_action_gradient:action_gradient, self.actor_is_training:True}
						self.sess.run(self.actor_network.optimizer, feed_dict=feed_dict)

					else:
						batch_next_action = self.sess.run(self.actor_network.target_layer3_out, feed_dict={self.actor_network.target_states:batch_next_s})
						batch_target_q = self.sess.run(self.critic_network.target_layer3_out, feed_dict={self.critic_network.target_states:batch_next_s, self.critic_network.target_actions:batch_next_action})

						# Set target(y) and update critic by minimizing the loss
						feed_dict = {self.critic_network.states:batch_s, self.critic_network.actions:batch_act, self.critic_network.rewards:batch_rwd, self.critic_network.done:batch_done, self.critic_network.target_q:np.squeeze(batch_target_q)}
						cost_, _ = self.sess.run([self.critic_network.cost, self.critic_network.optimizer], feed_dict=feed_dict)

						# Update the actor policy gradient using the sampled policy gradient
						action_batch_q_gradient = self.sess.run(self.actor_network.layer3_out, feed_dict={self.actor_network.states:batch_s})
						# Need to index [0] since output [1,64,1](tf.gradient returns 'list')
						action_gradient = self.sess.run(self.critic_network.gradients, feed_dict={self.critic_network.states:batch_s, self.critic_network.actions:action_batch_q_gradient})[0] 
						feed_dict = {self.actor_network.states:batch_s, self.actor_network.q_action_gradient:action_gradient}
						self.sess.run(self.actor_network.optimizer, feed_dict=feed_dict)


					self.cost_per_episode += cost_
					self.reward_per_episode += reward
					# Update target network
					self.actor_network.update_target()
					self.critic_network.update_target()

				if done:
					print('%d episode end' % (episode+1))
					self.exploration.reset()
					if self.replay_buffer.get_size > self.args.training_start:
						utils.write_log(self.steps, self.reward_per_episode, episode+1, start_time, mode='train', total_loss=self.cost_per_episode)
					# Initialize log variable
					self.initialize_statistics()
					break
						

		self.env.monitor.close()

	# Get actions according to the current policy and exploration noise
	# mu + noise
	def get_action(self, obs):
		if self.args.bn:
			current_policy_action = self.sess.run(self.actor_network.layer3_out, feed_dict={self.actor_network.states:np.asarray([obs]), self.actor_network.is_training:False})[0]
		else:
			current_policy_action = self.sess.run(self.actor_network.layer3_out, feed_dict={self.actor_network.states:np.asarray([obs])})[0]
		return current_policy_action + self.exploration.noise()

	def pure_actor_action(self, obs):
		if self.args.bn:
			current_policy_action = self.sess.run(self.actor_network.layer3_out, feed_dict={self.actor_network.states:np.asarray([obs]), self.actor_network.is_training:False})[0]
		else:
			current_policy_action = self.sess.run(self.actor_network.layer3_out, feed_dict={self.actor_network.states:np.asarray([obs])})[0]
		return current_policy_action
		

	def eval(self):
		self.eval_step = 0 
		start_time = time.time()

		self.initialize_statistics()
		utils.initialize_log()

		self.env.monitor.start(self.args.env_name, force=True)

		for episode in xrange(self.args.test_episodes):
			print('%d episode starts' % episode+1) 
			# Initial observation
			observation = self.env.reset()
			for step in xrange(self.step_per_episode):
				self.eval_step += 1
				# Select action according to the current policy and exploration noise
				action = self.pure_actor_action(observation)
				current_observation = observation
				observation, reward, done, _ = self.env.step(action)
				self.reward_per_episode += reward
				if done:
					utils.write_logs(self.eval_step, self.reward_per_episode, episode+1, start_time, mode='eval')
					self.initialize_statistics()
					break				

		self.env.monitor.close()


	def initialize_statistics(self):
		self.reward_per_episode = 0
		self.cost_per_episode = 0


	@property
	def model_dir(self):
		return '{}batch'.format(self.args.batch_size)


	def save(self, global_step):
		model_name = 'DDPG'
		checkpoint_dir = os.path.join(self.args.checkpoint_dir, self.model_dir)
		if not os.path.exists(checkpoint_dir):
			os.mkdir(checkpoint_dir)
		self.saver.save(self.sess, os.path.join(checkpoint_dir, model_name), global_step=global_step)
		print('Checkpoint saved at %d steps' % global_step)	


	def load(self):		
		checkpoint_dir = os.path.join(self.args.checkpoint_dir, self.model_dir)
		ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
		if ckpt and ckpt.model_checkpoint_path:
			ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
			self.saver.restore(self.sess, ckpt.model_checkpoint_path)
			print('Checkpoint loaded')
			return True

		else:
			print('Checkpoint load failed')
			return False



