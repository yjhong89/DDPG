import numpy as np
import tensorflow as tf
import os, time

LOG_DIR = './logs'
TRAIN = 'train.csv'
EVAL = 'eval.csv'

def initialize_log():
	try:
		train_log = open(os.path.join(LOG_DIR, TRAIN), 'a')
	except:
		print('Initialize log..')
		train_log = open(os.path.join(LOG_DIR, TRAIN), 'w')
		train_log.write('Step\t'+',episode.rwd\t'+',episode.cost\t'+',time\n')
	try:
		eval_log = open(os.path.join(LOG_DIR, EVAL), 'a')
	except:
		eval_log = open(os.path.join(LOG_DIR, EVAL), 'w')
		eval_log.write('Step\t'+',episode.rwd\t+'+',time\n')

	return train_log, eval_log


def write_log(steps, total_rwd, num_episode, start_time, mode, total_loss=0):
	train_log, eval_log = initialize_log()

	if mode == 'train':
		print('At Training step %d, %d-th episode => total.rwd : %3.4f, total.cost : %3.4f' % \
		(steps, num_episode, total_rwd, total_loss))
		train_log.write(str(steps)+'\t,' + str(total_rwd)+'\t,' + str(total_loss)+'\t,' \
		 + str(time.time() - start_time) + '\n')
		train_log.flush()
	elif mode == 'eval':
		print('At Evaluation step %d, %d-th episode => total.Q : %3.4f, total.rwd : %3.4f' % \
		(steps, num_episode, total_rwd, total_loss))
		eval_logs.write(str(steps)+'\t,' + str(total_rwd)+'\t,' + str(time.time() - start_time) + '\n')
		eval_log.flush()

