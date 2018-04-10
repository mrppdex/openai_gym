from collections import deque, namedtuple
import numpy as np
import random

Experience = namedtuple('Experience',['state', 'action', 'reward', 'next_state', 'done'])

class SumTree:
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.write = 0
        self.n_entries = 0
        
    def _propagate(self, idx, change):
        parent = (idx - 1) >> 1
        self.tree[parent] += change
        
        if parent != 0:
            self._propagate(parent, change)
            
    def _retrieve(self, idx, s):
        #print("_retrieve(self, {}, {})".format(idx, s))
        left = 2 * idx + 1
        right = left + 1
        #print("left={}, right={} ".format(left, right))
        
        if left >= len(self.tree):
            #print("cond, idx=", idx, "type=", type(idx))
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])
            
    def total(self):
        return self.tree[0]
    
    def add(self, p, data):
        idx = self.capacity + self.write - 1
        
        self.data[self.write] = data
        self.update(idx, p)
        
        self.write += 1
        self.write %= self.capacity
        
        self.n_entries = min(self.n_entries + 1, self.capacity)
        
    def update(self, idx, p):
        change = p - self.tree[idx]
        
        self.tree[idx] = p
        self._propagate(idx, change)
        
    def get(self, s):
        idx = self._retrieve(0, s)
        #print("idx={}, self.capacity={} ".format(idx, self.capacity))
        dataIdx = idx - self.capacity + 1
        
        return (idx, self.tree[idx], self.data[dataIdx])
        
class PReplayBuffer:
    e = 0.01
    a = 0.6
    beta = 0.4
    beta_inc = 0.001
    
    def __init__(self, capacity):
        self.tree = SumTree(capacity)
        self.capacity = capacity
        
    def _get_priority(self, error):
        return (error + self.e) ** self.a
    
    def add(self, error, sample):
        p = self._get_priority(error)
        self.tree.add(p, sample)
        
    def sample(self, n):
        """Returns batch, idxs, w_i"""
        batch = []
        idxs = []
        segment = self.tree.total() / n
        priorities = []
        
        self.beta = np.min([1., self.beta + self.beta_inc])
        
        for i in range(n):
            a = segment * i
            b = segment * (i + 1)
            
            s = random.uniform(a, b)
            (idx, p, data) = self.tree.get(s)
            priorities.append(p)
            batch.append(data)
            idxs.append(idx)
            
        sampling_probs = priorities / self.tree.total()
        #is_weight = np.power(self.tree.n_entries * sampling_probs, -self.beta)
        #is_weight /= is_weight.max()
        is_weight = None
        
        return batch, idxs, is_weight
    
    def update(self, idx, error):
        p = self._get_priority(error)
        self.tree.update(idx, p)
    
    
class ReplayBuffer:
	def __init__(self, size):
		self.replay_memory = deque(maxlen=size)			# used for O(1) popleft() operation

	def add_to_memory(self, experience):
		self.replay_memory.append(experience)

	def sample_from_memory(self, minibatch_size):
		return random.sample(self.replay_memory, minibatch_size)
		
	def len(self):
		return len(self.replay_memory)
        
class OUNoise:
    """Ornstein-Uhlenback process."""
    def __init__(self, size, mu=None, theta=0.15, sigma=0.3):
        """Initialize parameters and noise process."""
        self.size = size
        self.mu = mu if mu is not None else np.zeros(self.size)
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.size) * self.mu
        self.reset()
        
    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = self.mu
        
    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return self.state