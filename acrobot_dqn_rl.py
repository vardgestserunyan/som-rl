import torch 
from torch import nn, optim
import numpy as np
from collections import deque, namedtuple
import gymnasium as gym
import random
import logging
import matplotlib.pyplot as plt


class DeepQN(nn.Module):
    def __init__(self, dim_state, dim_actions, hyper):
        super(DeepQN, self).__init__()

        # Create layers for the policy network
        policy_net_list = [nn.Linear(dim_state, hyper["net_breadth"]),\
                           nn.LeakyReLU(0.2)]
        for _ in range(hyper["net_depth"]):
            policy_net_list.extend( [nn.Linear(hyper["net_breadth"], hyper["net_breadth"]),\
                                     nn.LeakyReLU(0.2)])
        policy_net_list.extend([nn.Linear(hyper["net_breadth"], dim_actions)])

        # Create layers for the target network
        target_net_list = [nn.Linear(dim_state, hyper["net_breadth"]),\
                           nn.LeakyReLU(0.2)]
        for _ in range(hyper["net_depth"]):
            target_net_list.extend( [nn.Linear(hyper["net_breadth"], hyper["net_breadth"]),\
                                     nn.LeakyReLU(0.2)])
        target_net_list.extend([nn.Linear(hyper["net_breadth"], dim_actions)])

        # Create attributes for the networks and save the hyperparameters
        self.hyper = hyper
        self.policy_net = nn.Sequential(*policy_net_list)
        self.target_net = nn.Sequential(*target_net_list)

        self.policy_net.to(self.hyper["device"])
        self.target_net.to(self.hyper["device"])
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def fwd_policy(self, model_input):
        # Fwd pass through the policy network
        policy_out = self.policy_net(model_input)
        return policy_out
    
    def fwd_target(self, model_input):
        # Fwd pass thorugh the target network
        target_out = self.target_net(model_input)
        return target_out
    
    def select_action(self, state, epsilon):
        # Select the next action given the current state
        state = torch.tensor(state, dtype=torch.float32).to(self.hyper["device"])
        if torch.rand(1) > epsilon:
            # Action based on exploitation
            next_action = self.policy_net(state).to("cpu").argmax().numpy()
        else:
            # Action based on exploration
            next_action = acrobot_env.action_space.sample()
        return next_action
    
    def optim_run(self, optimizer, memory):
        # Skip optimization if not enough traversal have happened 
        if len(memory) < self.hyper["batch_size"]:
            return None
        
        # Get items from the batch 
        batch = random.sample(memory, k=self.hyper["batch_size"])
        states_batch = torch.tensor(np.array([item.state for item in batch]), dtype=torch.float32).to(self.hyper["device"])
        actions_batch = torch.tensor([item.actions.item() for item in batch], dtype=torch.int32).to(self.hyper["device"])
        newstate_batch = torch.tensor(np.array([item.new_state for item in batch if not item.new_state is None]), dtype=torch.float32).to(self.hyper["device"])
        rewards_batch = torch.tensor([item.rewards for item in batch], dtype=torch.float32).to(self.hyper["device"])

        # Find the Q's for the current policy
        q_curr = self.policy_net(states_batch).gather(1, actions_batch.view(-1,1))

        # Find the Q's for the target, with terminal states having a Q of 0 (no next state to speak of)
        term_mask = torch.tensor([True if not item.new_state is None else False for item in batch], dtype=torch.bool).to(self.hyper["device"])
        next_state_q = torch.zeros_like(q_curr)
        with torch.no_grad():
            next_state_q[term_mask] = self.target_net(newstate_batch).max(dim=1).values.view(-1,1)
            target = rewards_batch.view(-1,1) + self.hyper["gamma"]*next_state_q
        
        # Compute the loss and backpropagate 
        loss = self.loss_fcn(q_curr, target)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        logger.info(f"Loss: {loss}")


    def trainer(self, acrobot_env, seed, optimizer, num_episodes, logger=None):

        # Put the models in training mode
        self.policy_net.train()
        self.target_net.train()
        
        # Create objects for tracking the history
        transition = namedtuple("transition", ("state", "actions", "new_state", "rewards"))
        memory = deque([], maxlen=self.hyper["memo_maxlen"])
        duration = np.zeros(num_episodes)
        logger.info("Started Deep Q Learning")

        # Start the episode-based training
        for idx in range(num_episodes):

            logger.info(f"Started Episode {idx}")
            
            # Initialize the environment, set the epsilon (explore/exploit balance)
            state, _ = acrobot_env.reset(seed=seed+idx)
            epsilon = max(self.hyper["eps_start"]*(self.hyper["eps_decay"]**idx), self.hyper["eps_end"])

            # Initialize loop variables
            trunc, term, t = False, False, 0
            while not (trunc or term): 
                # Pick an action and perform, adding to the seen states
                action= self.select_action(state, epsilon)
                new_state, reward, term, trunc, _ = acrobot_env.step(action)
                new_state = new_state if reward != 0 else None
                memory.append(transition(state, action, new_state, reward))

                # Perform optimization, updating the target network too
                self.optim_run(optimizer, memory)
                target_state_dict = self.target_net.state_dict()
                for key, value in self.policy_net.state_dict().items():
                    target_state_dict[key] = self.hyper["tau"]*target_state_dict[key] + (1-self.hyper["tau"])*value
                self.target_net.load_state_dict(target_state_dict)

                # Prepare for the next iteration
                t += 1
                state = new_state
                logger.info(f"Timepoint {t} done, reward: {reward}")

            duration[idx] = t

            logger.info(f"___________Done with Episode {idx}____________")
        
        return duration

    def loss_fcn(self, q_curr, target):
        # MSE loss between target and policy
        loss = nn.functional.mse_loss(q_curr, target)
        return loss
    

# Initialize the logging
logger = logging.getLogger(__name__)
logging.basicConfig(filename="acrobot_logger_dqn.log", level=logging.INFO)


# Define random seed
seed, num_episodes, lr = 12121995, 1000, 3e-4
torch.manual_seed(seed)
random.seed(seed)

# Define the Acrobot environment form Gymnasium
acrobot_env = gym.make("Acrobot-v1")
dim_state = acrobot_env.observation_space.shape[0]
dim_actions = int(acrobot_env.action_space.n)

# Set the hyperparameters
hyper = {"net_breadth": 256, "net_depth": 3, "batch_size": 256,\
         "eps_start": 0.95, "eps_decay": 0.995, "eps_end": 0.01,\
         "memo_maxlen": 5000, "device": "mps", "tau": 0.90,\
         "gamma": 0.95}

# Instantiate the network and train
acrobot_dqn = DeepQN(dim_state, dim_actions, hyper)
optimizer = optim.Adam(acrobot_dqn.parameters(), lr=lr)
duration = acrobot_dqn.trainer(acrobot_env, seed, optimizer, num_episodes, logger)

rolling_avg = np.cumsum(duration)
roll_window = 25
rolling_avg = (rolling_avg[roll_window:] - rolling_avg[:-roll_window])/roll_window

# Plot the number of time steps it takes the Acrobot to meet the "winning" criterion
fig, ax = plt.subplots(figsize=(5,5), ncols=1, nrows=1)
ax.plot(duration, color="blue")
ax.plot(range(roll_window,num_episodes), rolling_avg, color="red")
ax.set_xlabel("Learning Episodes")
ax.set_ylabel("Time to Success")
ax.set_title("DeepQN Learning of Acrobot")
ax.legend(["Obs. Duration", f"{roll_window}-Rolling Avg."])
fig.savefig("acrobot_rl_dqn.pdf")