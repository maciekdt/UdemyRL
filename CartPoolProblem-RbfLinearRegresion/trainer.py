import gym
import pickle
import copy
from policy import Policy
from q_value_model import QValueModel


class Trainer:
    def __init__(self, alpha_rate=.1, gamma=.99, rbf_centers_size=100, epsilon=.1):
        self.env = gym.make("CartPole-v1")
        self.alpha_rate = alpha_rate
        self.gamma = gamma
        self.rbf_centers_size = rbf_centers_size
        self.epsilon = epsilon
        
        
    def train(self, episodes_number=100_000, interval=1000, cache_file_path=None):
        action_space = [0, 1]
        q_value_model = QValueModel(alpha_rate=self.alpha_rate, gamma=self.gamma, rbf_centers_size=self.rbf_centers_size, action_space=action_space)
        policy = Policy(epsilon=self.epsilon, q_value=q_value_model, action_space=action_space, train_mode=True)

        avg_rewards_history = []
        reward_sum = 0
        max_reward_q_model:QValueModel = q_value_model
        max_reward = float('-inf')
        
        for i in range(episodes_number):
            reward = self.one_episode_train(policy=policy, q_value_model=q_value_model)
            reward_sum += reward
            if i % interval == 0 and i != 0:
                avg_reward = reward_sum/interval
                avg_rewards_history.append(avg_reward)
                if avg_reward > max_reward:
                    max_reward = avg_reward
                    max_reward_q_model = copy.copy(q_value_model)
                print(f'Episode [{i}/{episodes_number}] Avarage cumulative reward = {avg_reward}')
                reward_sum = 0
        if cache_file_path != None:
            with open(cache_file_path, 'wb') as file:
                pickle.dump(max_reward_q_model, file)
            print(f"Saved best Q-value model with avarage reward for {interval} episodes: {max_reward}")
        return avg_rewards_history
        
    
    def one_episode_train(self, policy, q_value_model):
        state = self.env.reset()[0]
        done = False
        cum_reward = 0

        while not done:
            a = policy.get_action(state)
            s_next, r, terminated, truncated, info = self.env.step(a)
            q_value_model.update(state=state, action=a, reward=r, next_state=s_next)
            cum_reward += r
            if terminated or truncated:
                done = True
        return cum_reward
            
            
        