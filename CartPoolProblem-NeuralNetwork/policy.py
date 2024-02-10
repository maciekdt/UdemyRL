import random
import numpy as np
from q_value_model import QValueModel


class Policy:
    def __init__(self, q_value, train_mode=True, init_epsilon=.8, min_epsilon=.01, delta_epsilon=.000_05):
        self.current_epsilon = init_epsilon
        self.min_epsilon = min_epsilon
        self.delta_epsilon = delta_epsilon
        self.q_value: QValueModel = q_value
        self.action_space = [0,1]
        self.train_mode = train_mode
        
    def get_action(self, state):
        if self.train_mode and random.random() < self.current_epsilon:
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
        
    def decrease_epsilon(self):
        if self.current_epsilon <= self.min_epsilon:
            self.current_epsilon = self.min_epsilon
        else:
            self.current_epsilon -= self.delta_epsilon
        