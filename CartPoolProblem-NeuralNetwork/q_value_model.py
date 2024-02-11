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
 
    
    def get_G(self, reward:float, next_state:list):
        with torch.no_grad():
            max_Q = float('-inf')
            for a in self.action_space:
                Q = self.nn_model(self.feature_transform(next_state, a)).item()
                if Q > max_Q: max_Q = Q
            return reward + self.gamma * max_Q
            
        
    
    def batch_update(self, batch_state_action:list, batch_G:list):
        batch_state_action = torch.tensor(batch_state_action, dtype=torch.float32)
        batch_G = torch.tensor(batch_G, dtype=torch.float32)
        self.optimizer.zero_grad()
        pred_Q = self.nn_model(batch_state_action)
        loss = self.criterion(pred_Q, batch_G)
        loss.backward()
        self.optimizer.step()
        return loss.item()
    
    def predict_Q(self, state:list, action:float):
        with torch.no_grad():
            action_state = self.feature_transform(state, action)
            Q = self.nn_model(action_state).item()
            return Q
            
        
    def feature_transform(self, state:list, action:float):
        action_state = state.copy()
        action_state.append(action)
        action_state = torch.tensor(action_state, dtype=torch.float32)
        action_state = action_state.unsqueeze(0)
        return action_state

def test():
    q = QValueModel(alpha_rate=.1, gamma=.99)
    s1 = [1,2,3,4]
    s2 = [4,3,2,1]
    a = 1

    print("Q-value for s1:", q.predict_Q(state=s1, action=a))
    print("Q-value for s2:", q.predict_Q(state=s1, action=a))
    print("G for s1 (Q-value + r=10):", q.get_G(reward=10., next_state=s1))
    print("G for s2 (Q-value + r=-5):", q.get_G(reward=-5., next_state=s2))

    batch_s = [[[1,2,3,4,1], [4,3,2,1,1]]]
    batch_G = [[[100], [-100]]]

    for i in range(1000):
        q.batch_update(batch_state_action=batch_s, batch_G=batch_G)


    print("Q-value for s1:", q.predict_Q(state=s1, action=a))
    print("Q-value for s2:", q.predict_Q(state=s2, action=a))

