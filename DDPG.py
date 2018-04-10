"""Generic base class for reinforcement learning agents."""

import gym
#import matplotlib
#import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
#from tensorflow.contrib.layers import fully_connected, batch_norm, relu
import random

from util import PReplayBuffer, ReplayBuffer, OUNoise

class Actor:
	def __init__(self, state_input, action_range, action_dim, scope):
		self.state_input = state_input
		self.action_range = action_range
		self.action_dim = action_dim
		
		with tf.variable_scope("actor_"+scope):
			self.out = self.build_model()
	
	def build_model(self):
		hidden = tf.layers.dense(self.state_input, 8, activation = tf.nn.relu, name = 'dense')
		hidden_2 = tf.layers.dense(hidden, 8, activation = tf.nn.relu, name = 'dense_1')
		hidden_3 = tf.layers.dense(hidden_2, 8, activation = tf.nn.relu, name = 'dense_2')
		actions_unscaled = tf.layers.dense(hidden_3, self.action_dim, name = 'dense_3')
		actions = (tf.nn.sigmoid(actions_unscaled) - 0.5)*self.action_range
		return actions

	
class Critic:
	def __init__(self, a_in, s_in, scope, reuse=False):
		
		self.action_in = a_in
		self.state_in = s_in
		
		with tf.variable_scope("critic_"+scope):
			self.q = self.build_model(self.state_in, self.action_in, reuse=reuse)
			#if not actions==None:
			#	self.q_inf = self.build_model(self.state_in, self.actions, reuse=True)
			#else:
			#	self.q_inf = self.q_det
	
	def build_model(self, s_in, a_in, reuse):
		inputs = tf.concat([s_in, a_in], axis=1)
		hidden = tf.layers.dense(inputs, 8, activation = tf.nn.relu, name = 'dense', reuse = reuse)
		hidden_2 = tf.layers.dense(hidden, 8, activation = tf.nn.relu, name = 'dense_1', reuse = reuse)
		hidden_3 = tf.layers.dense(hidden_2, 8, activation = tf.nn.relu, name = 'dense_2', reuse = reuse)
		q_logits = tf.layers.dense(hidden_3, 1, name = 'dense_3', reuse = reuse)
        
		return q_logits


class DDPG(BaseAgent):
	"""Generic base class for reinforcement agents."""
	
	def __init__(self, task):
         
		self.lr_actor = 5e-3				# learning rate for the actor
		self.lr_critic = 1e-3			# learning rate for the critic
		#self.lr_decay = 1				# learning rate decay (per episode)
		self.l2_reg_actor = 1e-7			# L2 regularization factor for the actor
		self.l2_reg_critic = 1e-7		# L2 regularization factor for the critic
         
		#self.num_episodes = 2000		# number of episodes
		#self.max_steps_ep = 10000	# default max number of steps per episode (unless env has a lower hardcoded limit)
		
		self.batch_size = 1024
		self.memory = ReplayBuffer(int(1e5))
		#self.episodes_nr = 10000
		self.gamma = 0.999
		self.tau = 2e-2
        
        self.task = task
		assert(task.action_space.high == -task.action_space.low)
		self.action_range = task.action_space.high[0]
        
		self.action_dim = np.prod(np.array(task.action_space.shape))
		self.state_dim = np.prod(np.array(task.observation_space.shape))
        
		#self.noise = OUNoise(self.action_dim)
		self.action_range = task.action_space.high - task.action_space.low
		
		#self.initial_noise_scale = 0.1	# scale of the exploration noise process (1.0 is the range of each action dimension)
		#self.noise_decay = 1 #0.99		# decay rate (per episode) of the scale of the exploration noise process
		#self.exploration_mu = 0.0	# mu parameter for the exploration noise process: dXt = theta*(mu-Xt)*dt + sigma*dWt
		#self.exploration_theta = 0.15 # theta parameter for the exploration noise process: dXt = theta*(mu-Xt)*dt + sigma*dWt
		#self.exploration_sigma = 0.2	# sigma parameter for the exploration noise process: dXt = theta*(mu-Xt	)*dt + sigma*dWt

		self.noise = OUNoise(self.action_dim)
		
		tf.reset_default_graph()
		
		self.state_ph = tf.placeholder(dtype=tf.float32, shape=[None,self.state_dim])
		self.action_ph = tf.placeholder(dtype=tf.float32, shape=[None,self.action_dim])
		self.reward_ph = tf.placeholder(dtype=tf.float32, shape=[None])
		self.next_state_ph = tf.placeholder(dtype=tf.float32, shape=[None,self.state_dim])
		self.is_not_terminal_ph = tf.placeholder(dtype=tf.float32, shape=[None]) # indicators (go into target computation)
		
		# episode counter
		#self.episodes = tf.Variable(0.0, trainable=False, name='episodes')
		#self.episode_inc_op = episodes.assign_add(1)

        
		
		self.actions = Actor(self.state_ph, self.action_range, self.action_dim, "local").out
		self.target_actions = tf.stop_gradient(Actor(self.next_state_ph, self.action_range, self.action_dim, "target").out)
		
		self.q_det = Critic(self.action_ph, self.state_ph, "local", reuse=False).q
		self.q_inf = Critic(self.actions, self.state_ph, "local", reuse=True).q
		
		self.target_critic = tf.stop_gradient(Critic(self.target_actions, self.next_state_ph, "target").q)
		 
		self.actor_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='actor_local')
		self.slow_target_actor_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='actor_target')
		self.critic_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='critic_local')
		self.slow_target_critic_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='critic_target')
		
		self.update_targets_ops = []
		for i, self.slow_target_actor_var in enumerate(self.slow_target_actor_vars):
			self.update_slow_target_actor_op = self.slow_target_actor_var.assign(self.tau*self.actor_vars[i]+(1-self.tau)*self.slow_target_actor_var)
			self.update_targets_ops.append(self.update_slow_target_actor_op)

		for i, self.slow_target_var in enumerate(self.slow_target_critic_vars):
			self.update_slow_target_critic_op = self.slow_target_var.assign(self.tau*self.critic_vars[i]+(1-self.tau)*self.slow_target_var)
			self.update_targets_ops.append(self.update_slow_target_critic_op)

		self.update_slow_targets_op = tf.group(*self.update_targets_ops, name='update_slow_targets')
		
		self.targets = tf.expand_dims(self.reward_ph, 1) + tf.expand_dims(self.is_not_terminal_ph, 1) * self.gamma * self.target_critic
		
		self.td_errors = self.targets - self.q_det
		
		self.critic_loss = tf.reduce_mean(tf.square(self.td_errors))
		for var in self.critic_vars:
			if not 'bias' in var.name:
				self.critic_loss += self.l2_reg_critic * 0.5 * tf.nn.l2_loss(var)

		# critic optimizer
		self.critic_train_op = tf.train.AdamOptimizer(self.lr_critic*self.lr_decay**self.episodes).minimize(self.critic_loss)

		# actor loss function (mean Q-values under current policy with regularization)
		self.actor_loss = -1*tf.reduce_mean(self.q_inf)
		for var in self.actor_vars:
			if not 'bias' in var.name:
				self.actor_loss += self.l2_reg_actor * 0.5 * tf.nn.l2_loss(var)

		# actor optimizer
		# the gradient of the mean Q-values wrt actor params is the deterministic policy gradient (keeping critic params fixed)
		self.actor_train_op = tf.train.AdamOptimizer(self.lr_actor).minimize(self.actor_loss, var_list=self.actor_vars)

		# initialize session
		self.sess = tf.Session()	
		self.sess.run(tf.global_variables_initializer())
		
		self.total_steps = 0
		
	
	def reset_state(self, state):
		self.observation = state
		self.prev_action = None
		self.total_reward = 0
		self.steps_in_ep = 0
			
	
	def step(self, next_observation, reward, done):
		"""Process state, reward, done flag, and return an action.
		
		Params
		======
		- state: current state vector as NumPy array, compatible with task's state space
		- reward: last reward received
		- done: whether this episode is complete
		
		Returns
		=======
		- action: desired action vector as NumPy array, compatible with task's action space
		"""
		
		# choose action based on deterministic policy
		action_for_state, = sess.run(self.actions, feed_dict = {self.state_ph: self.observation[None]})

		if self.prev_action == None:
			self.prev_action = action_for_state
			return action_for_state

		# add temporally-correlated exploration noise to action (using an Ornstein-Uhlenbeck process)
		# print(action_for_state)
		#noise_process = self.exploration_theta*(self.exploration_mu - noise_process) + self.exploration_sigma*np.random.randn(self.action_dim)
		# print(noise_scale*noise_process)
		action_for_state += self.noise.sample() #noise_process #*noise_scale

		# take step
		#next_observation, reward, done, _info = self.env.step(action_for_state)
		#if ep%1 == 0: self.env.render()
		self.total_reward += reward

		self.memory.add_to_memory((self.observation, action_for_state, reward, next_observation, 0.0 if done else 1.0))

		# update network weights to fit a minibatch of experience
		if self.memory.len() >= self.batch_size:

			# grab N (s,a,r,s') tuples from replay memory
			minibatch = self.memory.sample_from_memory(self.batch_size)

			# update the critic and actor params using mean-square value error and deterministic policy gradient, respectively
			_, _ = sess.run([self.critic_train_op, self.actor_train_op], 
						feed_dict = {
							self.state_ph: np.asarray([elem[0] for elem in minibatch]),
							self.action_ph: np.asarray([elem[1] for elem in minibatch]),
							self.reward_ph: np.asarray([elem[2] for elem in minibatch]),
							self.next_state_ph: np.asarray([elem[3] for elem in minibatch]),
							self.is_not_terminal_ph: np.asarray([elem[4] for elem in minibatch])})

					# update slow actor and critic targets towards current actor and critic
			_ = sess.run(self.update_slow_targets_op)

		self.observation = next_observation
		self.total_steps += 1
		self.steps_in_ep += 1
	
		#if done: 
			# Increment episode counter
			_ = sess.run(self.episode_inc_op)
		#	break
		
		self.prev_action = action_for_state
		
		return action_for_state