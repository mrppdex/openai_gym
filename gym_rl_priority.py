import gym
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers import fully_connected, batch_norm, relu
import random

from util import PReplayBuffer, ReplayBuffer, OUNoise

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_integer('episodes_number', 100, "Number of episodes")


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
	
class Agent:
	def __init__(self, gamma=0.999, buffer_size=1e5, batch_size=1024,
                 episodes_nr=50000, tau=2e-2, gym_name='MountainCarContinuous-v0'):
         
		self.lr_actor = 5e-3				# learning rate for the actor
		self.lr_critic = 1e-3			# learning rate for the critic
		self.lr_decay = 1				# learning rate decay (per episode)
		self.l2_reg_actor = 1e-7			# L2 regularization factor for the actor
		self.l2_reg_critic = 1e-7		# L2 regularization factor for the critic
         
		self.num_episodes = episodes_nr		# number of episodes
		self.max_steps_ep = 10000	# default max number of steps per episode (unless env has a lower hardcoded limit)
		self.train_every = 1			# number of steps to run the policy (and collect experience) before updating network weights
		self.replay_memory_capacity = buffer_size	# capacity of experience replay memory
		
		self.batch_size = batch_size
		self.memory = ReplayBuffer(int(buffer_size))
		self.episodes_nr = episodes_nr
		self.gamma = gamma
		self.tau = tau
        
		self.env = gym.make(gym_name)
		assert(self.env.action_space.high == -self.env.action_space.low)
		self.action_range = self.env.action_space.high[0]
        
		self.action_dim = np.prod(np.array(self.env.action_space.shape))
		self.state_dim = np.prod(np.array(self.env.observation_space.shape))
        
		#self.noise = OUNoise(self.action_dim)
		self.action_range = self.env.action_space.high - self.env.action_space.low
		
		self.initial_noise_scale = 0.1	# scale of the exploration noise process (1.0 is the range of each action dimension)
		self.noise_decay = 1 #0.99		# decay rate (per episode) of the scale of the exploration noise process
		self.exploration_mu = 0.0	# mu parameter for the exploration noise process: dXt = theta*(mu-Xt)*dt + sigma*dWt
		self.exploration_theta = 0.15 # theta parameter for the exploration noise process: dXt = theta*(mu-Xt)*dt + sigma*dWt
		self.exploration_sigma = 0.2	# sigma parameter for the exploration noise process: dXt = theta*(mu-Xt	)*dt + sigma*dWt

		self.noise = OUNoise(self.action_dim)
		
	def run(self):
		
		tf.reset_default_graph()
		
		state_ph = tf.placeholder(dtype=tf.float32, shape=[None,self.state_dim])
		action_ph = tf.placeholder(dtype=tf.float32, shape=[None,self.action_dim])
		reward_ph = tf.placeholder(dtype=tf.float32, shape=[None])
		next_state_ph = tf.placeholder(dtype=tf.float32, shape=[None,self.state_dim])
		is_not_terminal_ph = tf.placeholder(dtype=tf.float32, shape=[None]) # indicators (go into target computation)
		
		# episode counter
		episodes = tf.Variable(0.0, trainable=False, name='episodes')
		episode_inc_op = episodes.assign_add(1)

        
		
		actions = Actor(state_ph, self.action_range, self.action_dim, "local").out
		target_actions = tf.stop_gradient(Actor(next_state_ph, self.action_range, self.action_dim, "target").out)
		
		q_det = Critic(action_ph, state_ph, "local", reuse=False).q
		q_inf = Critic(actions, state_ph, "local", reuse=True).q
		
		target_critic = tf.stop_gradient(Critic(target_actions, next_state_ph, "target").q)
		 
		actor_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='actor_local')
		slow_target_actor_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='actor_target')
		critic_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='critic_local')
		slow_target_critic_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='critic_target')
		
		update_targets_ops = []
		for i, slow_target_actor_var in enumerate(slow_target_actor_vars):
			update_slow_target_actor_op = slow_target_actor_var.assign(self.tau*actor_vars[i]+(1-self.tau)*slow_target_actor_var)
			update_targets_ops.append(update_slow_target_actor_op)

		for i, slow_target_var in enumerate(slow_target_critic_vars):
			update_slow_target_critic_op = slow_target_var.assign(self.tau*critic_vars[i]+(1-self.tau)*slow_target_var)
			update_targets_ops.append(update_slow_target_critic_op)

		update_slow_targets_op = tf.group(*update_targets_ops, name='update_slow_targets')
		
		targets = tf.expand_dims(reward_ph, 1) + tf.expand_dims(is_not_terminal_ph, 1) * self.gamma * target_critic
		
		td_errors = targets - q_det
		
		critic_loss = tf.reduce_mean(tf.square(td_errors))
		for var in critic_vars:
			if not 'bias' in var.name:
				critic_loss += self.l2_reg_critic * 0.5 * tf.nn.l2_loss(var)

		# critic optimizer
		critic_train_op = tf.train.AdamOptimizer(self.lr_critic*self.lr_decay**episodes).minimize(critic_loss)

		# actor loss function (mean Q-values under current policy with regularization)
		actor_loss = -1*tf.reduce_mean(q_inf)
		for var in actor_vars:
			if not 'bias' in var.name:
				actor_loss += self.l2_reg_actor * 0.5 * tf.nn.l2_loss(var)

		# actor optimizer
		# the gradient of the mean Q-values wrt actor params is the deterministic policy gradient (keeping critic params fixed)
		actor_train_op = tf.train.AdamOptimizer(self.lr_actor*self.lr_decay**episodes).minimize(actor_loss, var_list=actor_vars)

		# initialize session
		sess = tf.Session()	
		sess.run(tf.global_variables_initializer())
		
		
		total_steps = 0
		for ep in range(self.num_episodes):

			total_reward = 0
			steps_in_ep = 0
			
			#noise_process = np.zeros(self.action_dim)
			#noise_scale = (self.initial_noise_scale * self.noise_decay**ep) * self.action_range


			# Initial state
			observation = self.env.reset()
			if ep%1 == 0: self.env.render()
	

			for t in range(self.max_steps_ep):

				# choose action based on deterministic policy
				action_for_state, = sess.run(actions, feed_dict = {state_ph: observation[None]})

				# add temporally-correlated exploration noise to action (using an Ornstein-Uhlenbeck process)
				# print(action_for_state)
				#noise_process = self.exploration_theta*(self.exploration_mu - noise_process) + self.exploration_sigma*np.random.randn(self.action_dim)
				# print(noise_scale*noise_process)
				action_for_state += self.noise.sample() #noise_process #*noise_scale

				# take step
				next_observation, reward, done, _info = self.env.step(action_for_state)
				if ep%1 == 0: self.env.render()
				total_reward += reward

				self.memory.add_to_memory((observation, action_for_state, reward, next_observation, 0.0 if done else 1.0))

				# update network weights to fit a minibatch of experience
				if total_steps%self.train_every == 0 and self.memory.len() >= self.batch_size:

					# grab N (s,a,r,s') tuples from replay memory
					minibatch = self.memory.sample_from_memory(self.batch_size)

					# update the critic and actor params using mean-square value error and deterministic policy gradient, respectively
					_, _ = sess.run([critic_train_op, actor_train_op], 
						feed_dict = {
							state_ph: np.asarray([elem[0] for elem in minibatch]),
							action_ph: np.asarray([elem[1] for elem in minibatch]),
							reward_ph: np.asarray([elem[2] for elem in minibatch]),
							next_state_ph: np.asarray([elem[3] for elem in minibatch]),
							is_not_terminal_ph: np.asarray([elem[4] for elem in minibatch])})

					# update slow actor and critic targets towards current actor and critic
					_ = sess.run(update_slow_targets_op)

				observation = next_observation
				total_steps += 1
				steps_in_ep += 1
	
				if done: 
					# Increment episode counter
					_ = sess.run(episode_inc_op)
					break
		
			print('Episode %2i, Reward: %7.3f, Steps: %i'%(ep,total_reward,steps_in_ep))

		env.close()

def main(_):
    #print(FLAGS.episodes_number)
    #input("press enter")
    a = Agent()#episodes_nr=FLAGS.episodes_number)
    a.run()

if __name__ == '__main__':
    a = Agent()#episodes_nr=FLAGS.episodes_number)
    a.run()
    #tf.app.run()
		
		
  