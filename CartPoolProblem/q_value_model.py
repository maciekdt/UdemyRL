import numpy as np
from scipy.spatial.distance import cdist

class QValueModel:
    input_vector_size = 5
    
    def __init__(self, alpha_rate, gamma, rbf_centers_size, action_space):
        self.alpha_rate = alpha_rate
        self.gamma = gamma
        self.w = np.random.uniform(low=0, high=1, size=rbf_centers_size)
        self.action_space = action_space
        
        self.rbf_centers = np.random.uniform(low=-2, high=2, size=(rbf_centers_size, self.input_vector_size))
        self.rbf_beta = .1
        self.normalizator = MinMaxScaler()
        
        #print("w:", self.w)
        #print("rbf centers::", self.rbf_centers)
        
        
    def update(self, state, action, reward, next_state):
        max_Q = float('-inf')
        for a in self.action_space:
            Q = self.w.T @ self.rbf_feature_transform(next_state, a)
            if Q > max_Q:
                max_Q = Q
        G = reward + self.gamma * max_Q
        rbf_action_state = self.rbf_feature_transform(state, action)
        loss = G - (self.w.T @ rbf_action_state)
        self.w = self.w + self.alpha_rate * loss * rbf_action_state
    
    
    def predict_Q(self, state, action):
        return self.w.T @ self.rbf_feature_transform(state, action)
        
        
    def rbf_feature_transform(self, state, action):
        action_state = np.append(state, action)
        action_state = self.normalizator.transform(action_state)
        #print("normalized_action_state:", action_state)
        action_state = np.expand_dims(action_state, axis=0)
        distances = cdist(action_state, self.rbf_centers, 'euclidean')
        transformed_features = np.exp(-self.rbf_beta * distances**2)
        transformed_features = np.squeeze(transformed_features)
        #print("transformed_features:", transformed_features)
        return transformed_features
    

class MinMaxScaler:
    def __init__(self, max_velocity=5):
        self.min_values = np.array([-4.8, -max_velocity, -0.418, -max_velocity, 0])
        self.max_values = np.array([4.8, max_velocity, 0.418, max_velocity, 1])
        
    def transform(self, X):
        return (X - self.min_values) / (self.max_values - self.min_values)