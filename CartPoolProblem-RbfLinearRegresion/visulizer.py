import gym
import pickle
from q_value_model import QValueModel
from policy import Policy

class Visualizer:
    def __init__(self):
        self.env = gym.make("CartPole-v1", render_mode='human')
        
    
    def start(self, cache_file_path):
        with open(cache_file_path, 'rb') as file:
            q_value_model: QValueModel = pickle.load(file)
            print("Q-value model loaded from memory")
            policy = Policy(epsilon=None, q_value=q_value_model, train_mode=False)
            
            s = self.env.reset()[0]
            done = False
            cum_r = 0
            while not done:
                a = policy.get_action(s)
                s, r, terminated, truncated, _ = self.env.step(a)
                cum_r += r
                if terminated or truncated:
                    done = True
            return cum_r
                
vis = Visualizer()
cum_r = vis.start("q_cache.pkl")
print("Cumulative reward:", cum_r)