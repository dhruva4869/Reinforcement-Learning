import gymnasium as gym # used in for loading some of the environments that we will test our agents on
import random
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
import time

# Parameters
LR = 1e-3
GAMMA = 0.99 # Discount factor in RL
# At the start the agent will not take anything, so we need to keep the epsilon really high since every step there is epsilon probability that the agent will take a random action
# Then we need to decay the epsilon over time so that the agent will start taking more and more greedy actions -> based off what it has already learnt, argmax Q =  PI function
EPS_START = 1.0
EPS_END = 0.01
EPS_DECAY = 0.995
BATCH_SIZE = 64
MEMORY_SIZE = 50000
TARGET_UPDATE = 10
EPISODES = 700
MAX_STEPS_PER_EPISODE = 500

# class for Q network
class DQN(nn.Module):
    """
    Simple Deep Q-Learning model
    """
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.fcc = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, action_dim),
        )

    def forward(self, x):
        return self.fcc(x)


class ReplayBuffer:
    """
    Replay buffer used to store the experiences of the agent
    transition type will be of the format = (state, action, reward, next_state, done)
    """
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def push(self, transition):
        self.memory.append(transition)

    def sample(self, size):
        return random.sample(self.memory, size)

    def __len__(self):
        return len(self.memory)


env = gym.make("CartPole-v1")
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n
# state, info = env.reset()

# print(state_dim)
# print(action_dim)
# labels = ["cart_position", "cart_velocity", "pole_angle", "pole_angular_velocity"]
# for i, label in enumerate(labels):
#     print(f"{label}: {state[i]:.2f}")


# Creating a q_net and a target_net for optimizing and loss checking
# then we can do optimizer.zero_grad() -> removes prev gradient loss.backward() -> backpropagtes and calculates the new gradients optimizer.step() -> updates the values
q_net = DQN(state_dim, action_dim)
target_net = DQN(state_dim, action_dim)
optimizer = optim.Adam(q_net.parameters(), lr=LR)
target_net.load_state_dict(q_net.state_dict())
memory = ReplayBuffer(MEMORY_SIZE)


epsilon = EPS_START
print("Starting Optimized Monte-Carlo training...")

# We'll store completed episodes for sampling
episode_buffer = []

for episode in range(EPISODES):
    state, _ = env.reset()
    episode_memory = []
    total_reward = 0

    for step in range(MAX_STEPS_PER_EPISODE):
        if random.random() < epsilon:
            # Explore
            action = env.action_space.sample()
        else:
            # Exploit learned Q-values
            with torch.no_grad():
                action = q_net(torch.FloatTensor(state)).argmax().item()

        next_state, reward, done, truncated, _ = env.step(action)
        total_reward += reward
        episode_memory.append((state, action, reward))
        state = next_state

        if done or truncated:
            break

    G = 0
    returns = []
    for (_, _, r) in reversed(episode_memory):
        G = r + GAMMA * G
        returns.insert(0, G)

    # Mean 
    returns = torch.FloatTensor(returns)
    returns = (returns - returns.mean()) / (returns.std() + 1e-8)

    # Store full episode in buffer
    states, actions, _ = zip(*episode_memory)
    episode_buffer.extend(list(zip(states, actions, returns.numpy())))

    # --- Mini-batch Monte Carlo training ---
    if len(episode_buffer) >= BATCH_SIZE:
        batch = random.sample(episode_buffer, BATCH_SIZE)
        states, actions, returns = zip(*batch)

        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions).unsqueeze(1)
        returns = torch.FloatTensor(returns)

        # Predicted Q-values
        q_values = q_net(states).gather(1, actions).squeeze()

        # Monte Carlo target = normalized discounted return
        loss = nn.MSELoss()(q_values, returns)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    epsilon = max(EPS_END, epsilon * EPS_DECAY)

    if episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(q_net.state_dict())
        print(f"[MC] Episode {episode+1}, Reward: {total_reward:.1f}, Epsilon: {epsilon:.3f}")

env.close()
print("Training complete .... 🐠")

env = gym.make("CartPole-v1", render_mode="human")  # enable GUI
state, _ = env.reset()
done = False
total_reward = 0
run_time = time.time() + 20

while not done:
    env.render()
    with torch.no_grad():
        action = q_net(torch.FloatTensor(state)).argmax().item()
    state, reward, done, _, _ = env.step(action)
    total_reward += reward

env.close()
print(f"Total Reward Achieved by Agent: {total_reward}. 🐠")