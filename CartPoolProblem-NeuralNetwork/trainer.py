import gym
import pickle
import copy
from policy import Policy
from q_value_model import QValueModel


class Trainer:
    def __init__(self, q_value_model, policy):
        self.env = gym.make("CartPole-v1")
        self.q_value_model: QValueModel = q_value_model
        self.policy: Policy = policy
        self.action_space = [0,1]
        
        
    def train(self, batch_size=100, episodes_number=100_000, interval=1000, cache_file_path=None):

        # For reward statistic
        rewards_history = []
        reward_sum_for_episode = 0
        reward_sum_for_interval = 0
        
        # For loss statistic
        loss_sum = 0
        updates_number = 0
        #max_reward_q_model:QValueModel = self.q_value_model
        #max_reward = float('-inf')
        
        current_state = self.env.reset()[0].tolist()
        batch_action_state = []
        batch_G = []
        
        episode = 0
        
        while episode < episodes_number:
            next_state, reward, is_episode_end = self.one_step_train(
                state=current_state,
                batch_action_state=batch_action_state,
                batch_G=batch_G)
            
            if not is_episode_end:
                current_state = next_state
                reward_sum_for_episode += reward
            else:
                current_state = self.env.reset()[0].tolist()
                episode += 1
                self.policy.decrease_epsilon()
                
                # For statistic
                reward_sum_for_interval += reward_sum_for_episode
                reward_sum_for_episode = 0
                if episode % interval == 0 and episode != 0:
                    rewards_history.append(reward_sum_for_interval/interval)
                    reward_sum_for_interval = 0
                    print(f'\nEpisode[{episode}/{episodes_number}]')
                    print(f'Avarage return reward = {rewards_history[len(rewards_history)-1]}')
                    print(f'Epsilon = {format(self.policy.current_epsilon, ".3f")}')
                    print(f'Avarage loss: {format(loss_sum/updates_number, ".6f")}')
                    loss_sum = 0
                    updates_number = 0
                
            #Updating model and reseting batches
            if(len(batch_action_state) >= batch_size):
                batch_loss = self.q_value_model.batch_update(
                    batch_state_action=[batch_action_state],
                    batch_G=[batch_G])
                updates_number += 1
                batch_action_state = []
                batch_G = []
                
                # For loss statistic
                loss_sum += batch_loss
                updates_number += 1
                
        return rewards_history

    
    def one_step_train(self, state, batch_action_state, batch_G):
        action = self.policy.get_action(state)
        next_state, reward, terminated, truncated, info = self.env.step(action)
        
        next_state = next_state.tolist()
        G = self.q_value_model.get_G(reward=reward, next_state=next_state)
        action_state = state.copy()
        action_state.append(action) 
        
        batch_action_state.append(action_state)
        batch_G.append([G])
        
        # Add terminated states to batch with G = 0
        if terminated or truncated:
            for a in self.action_space:
                terminated_action_state = next_state.copy()
                terminated_action_state.append(a)
                batch_action_state.append(terminated_action_state)
                batch_G.append([0.])
            
        return next_state, reward, terminated or truncated
            
    def save_model(self, q_model, file_path):
        with open(file_path, 'wb') as file:
            pickle.dump(q_model, file)
            

q_value_model = QValueModel(alpha_rate=.001, gamma=.95)
policy = Policy(train_mode=True, init_epsilon=1., min_epsilon=.01, delta_epsilon=.00001, q_value=q_value_model)
trainer = Trainer(policy=policy, q_value_model=q_value_model)
trainer.train(episodes_number=100_000, batch_size=1000)
        