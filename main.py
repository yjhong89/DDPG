import tensorflow as tf
import numpy as np
import gym
import os
from ddpg import DDPG
import argparse

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--replay_size', type=int, default=1000000)
	parser.add_argument('--train_start', type=int, default=10000)
	parser.add_argument('--batch_size', type=int, default=64)
	parser.add_argument('--gamma', type=float, default=0.99)
	parser.add_argument('--layer1', type=int, default=400)
	parser.add_argument('--layer2', type=int, default=300)
	parser.add_argument('--actor_lr', type=float, default=1e-4)
	parser.add_argument('--critic_lr', type=float, default=1e-3)
	parser.add_argument('--tau', type=float, default=1e-3)
	parser.add_argument('--regularize_decay', type=float, default=1e-2)
	parser.add_argument('--bn', type=str2bool, default='n')
	parser.add_argument('--num_episodes', type=int, default=100000)
	parser.add_argument('--env_name', type=str, default='InvertedPendulum-v1')
	parser.add_argument('--checkpoint_dir', type=str, default='./checkpoint')
	parser.add_argument('--log_dir', type=str, default='./logs')
	parser.add_argument('--training', type=str2bool, default='y')
	parser.add_argument('--test_episodes', type=int, default=1e2)
	parser.add_argument('--monitor', type=str2bool, default='n')
	parser.add_argument('--training_start', type=int, default=5000)
	parser.add_argument('--save_interval', type=int, default=5000)
	
	args = parser.parse_args()
	print(args)
	if not os.path.exists(args.checkpoint_dir):
		os.mkdir(args.checkpoint_dir)
	if not os.path.exists(args.log_dir):
		os.mkdir(args.log_dir)

	env = gym.make(args.env_name)
		

	run_config = tf.ConfigProto()
	run_config.log_device_placement = False
	run_config.gpu_options.allow_growth = True

	with tf.Session(config=run_config) as sess:
		ddpg = DDPG(args, sess, env)
		if args.training:
			ddpg.train()
		else:
			ddpg.eval()	


def str2bool(v):
	if v.lower() in ('yes', 'y', '1', 'true', 't'):
		return True
	elif v.lower() in ('no', 'n', '0', 'false', 'f'):
		return False
	else:
		raise ValueError('Neither true/false')


if __name__ == "__main__":
	main()
