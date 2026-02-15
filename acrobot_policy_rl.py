import gymnasium as gym
import torch
from torch import nn, optim
import numpy as np
from collections import deque, namedtuple
import matplotlib.pyplot as plt


class PolicyNN(nn.Module):
    def __init__(self, num_obs, num_act, hyper):
        super(PolicyNN, self).__init__()

        policy_net_list = [ nn.Linear(num_obs, hyper["net_breadth"]),\
                            nn.LeakyReLU(0.2) ]
        for _ in range(hyper["net_depth"]-1):
            policy_net_list.extend([nn.Linear(hyper["net_breadth"], hyper["net_breadth"]),\
                                    nn.LeakyReLU(0.2)])
        policy_net_list.extend([nn.Linear(hyper["net_breadth"], num_act),\
                                nn.Softmax(dim=1)])

        self.policy_net = nn.Sequential(*policy_net_list)
        self.hyper = hyper
        self.policy_net.to(hyper["device"])

    def fwd_policy(self, model_input):
        model_input = torch.tensor(model_input, dtype=torch.float32).view(-1, num_obs).to(self.hyper["device"])
        model_output = self.policy_net(model_input)
        return model_output
    
    def select_action(self, state):
        model_output = self.fwd_policy(state)
        action = torch.multinomial(model_output, 1)
        return action

    def trainer(self, acrobot_env, num_episodes):

        duration = np.zeros(num_episodes)
        for idx in range(num_episodes):

            transition_tuple = namedtuple("memory", ("state", "action", "new_state", "reward"))
            state, _ = acrobot_env.reset(seed=seed+idx)
            memory, trunc, term = [], False, False

            while not (trunc or term):
                action = self.select_action(state)
                new_state, reward, term, trunc, _ = acrobot_env.step(action)
                memory.append( transition_tuple(state, action, new_state, reward) )

                state = new_state
                duration[idx] += 1

            disc_rewards, curr_reward = [], 0 
            for transition in reversed(memory):
                new_reward = curr_reward + self.hyper["gamma"]*transition.reward
                disc_rewards.append(new_reward)
                curr_reward = new_reward
            
            disc_rewards = disc_rewards[::-1]

            loss = 0
            for transition, disc_reward in zip(memory, disc_rewards):
                probs = self.fwd_policy(transition.state).squeeze()
                loss += -disc_reward*torch.log(probs[transition.action])
                
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        return duration

            

seed = 12121995
torch.manual_seed(seed)

device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
lr, num_episodes = 1e-4, 1000

hyper = {"net_breadth": 256, "net_depth": 3, "batch_size": 256,\
         "device": device, "gamma": 0.90}

acrobot_env = gym.make("Acrobot-v1")
num_obs = acrobot_env.observation_space.shape[0]
num_act = acrobot_env.action_space.n


# Instantiate the network and train
acrobot_pnn = PolicyNN(num_obs, num_act, hyper)
optimizer = optim.Adam(acrobot_pnn.parameters(), lr=lr)
duration = acrobot_pnn.trainer(acrobot_env, num_episodes)

rolling_avg = np.cumsum(duration)
roll_window = 25
rolling_avg = (rolling_avg[roll_window:] - rolling_avg[:-roll_window])/roll_window

# Plot the number of time steps it takes the Acrobot to meet the "winning" criterion
fig, ax = plt.subplots(figsize=(5,5), ncols=1, nrows=1)
ax.plot(duration, color="blue")
ax.plot(range(roll_window,num_episodes), rolling_avg, color="red")
ax.set_xlabel("Learning Episodes")
ax.set_ylabel("Time to Success")
ax.set_title("PolicyNN Learning of Acrobot")
ax.legend(["Obs. Duration", f"{roll_window}-Rolling Avg."])
fig.savefig("acrobot_rl_pnn.pdf")


