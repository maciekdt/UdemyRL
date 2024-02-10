import random
import numpy as np
from q_value_model import QValueModel


class Policy:
    def __init__(self, q_value, train_mode=True, epsilon=None, action_space=[0, 1]):
        self.epsilon = epsilon
        self.q_value: QValueModel = q_value
        self.action_space: list = action_space
        self.train_mode = train_mode
        
    def get_action(self, state):
        if self.train_mode and random.random() < self.epsilon:
            return random.choice(self.action_space)
        else:
            max_Q = float('-inf')
            max_action = None
            for action in self.action_space:
                Q = self.q_value.predict_Q(state=state, action=action)
                if Q > max_Q:
                    max_Q = Q
                    max_action = action 
            return max_action
        