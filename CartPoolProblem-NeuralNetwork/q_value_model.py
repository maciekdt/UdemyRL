import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


class RegresionNNModel(nn.Module):
    def __init__(self):
        super(RegresionNNModel, self).__init__()
        self.ln1 = nn.Linear(in_features=5, out_features=24)
        self.tanh1 = nn.Tanh()
        self.ln2 = nn.Linear(in_features=24, out_features=48)
        self.tanh2 = nn.Tanh()
        self.ln3 = nn.Linear(in_features=48, out_features=1)
        

    def forward(self, x):
        x = self.tanh1(self.ln1(x))
        x = self.tanh2(self.ln2(x))
        x = self.ln3(x)
        return x
    

class QValueModel:
    def __init__(self, alpha_rate, gamma):
        self.alpha_rate = alpha_rate
        self.gamma = gamma
        self.action_space = [0,1]
        
        nn_model = RegresionNNModel()
        self.nn_model = nn_model
        self.optimizer = optim.Adam(nn_model.parameters(), lr=alpha_rate)
        self.criterion = nn.MSELoss()
        self.normalizator = MinMaxScaler()
        
        
    def update(self, state, action, reward, next_state):
        max_Q = float('-inf')
        for a in self.action_space:
            Q = self.predict_Q(next_state, a)
            if Q > max_Q: max_Q = Q
                
        G = reward + self.gamma * max_Q
        G = torch.tensor([[G]], dtype=torch.float32)

        self.optimizer.zero_grad()
        Q = self.nn_model(self.feature_transform(state, action))
        loss = self.criterion(Q, G)
        
        loss.backward()
        self.optimizer.step()
        return loss.item()
        
    
    def predict_Q(self, state, action):
        action_state = self.feature_transform(state, action)
        Q = self.nn_model(action_state).item()
        return Q
        
        
    def feature_transform(self, state, action):
        action_state = np.append(state, action)
        action_state = self.normalizator.transform(action_state)
        action_state = action_state.astype(np.float32)
        action_state = torch.from_numpy(action_state)
        action_state = action_state.unsqueeze(0)
        return action_state


class MinMaxScaler:
    def __init__(self, max_velocity=5.):
        self.min_values = np.array([-4.8, -max_velocity, -0.418, -max_velocity, 0])
        self.max_values = np.array([4.8, max_velocity, 0.418, max_velocity, 1])
        
    def transform(self, X):
        return (X - self.min_values) / (self.max_values - self.min_values)

    
    
"""    
q = QValueModel(alpha_rate=.1, gamma=.9)
state = np.array([3.3, -3.2, 0.04, 2.3])
action = 0
print("Pred Q-value:", q.predict_Q(state=state, action=action))

reward = 50
next_state = np.array([2.3, -1.2, 0.12, 1.3])
print("Update Loss:", q.update(state=state, action=action, reward=reward, next_state=state))
print("Update Loss:", q.update(state=state, action=action, reward=reward, next_state=state))

print("Updated Pred Q-value:", q.predict_Q(state=state, action=action))
"""