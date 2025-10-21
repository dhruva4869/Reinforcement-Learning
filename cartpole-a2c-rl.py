import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import numpy as np

LR = 1e-3
GAMMA = 0.99
EPS_START = 1.0
EPS_END = 0.01
EPS_DECAY = 0.995
BATCH_SIZE = 64
EPISODES = 700
MAX_STEPS_PER_EPISODE = 500
ENTROPY_BETA = 0.001  # encourages exploration


class ActorCritic(nn.Module):
    """
    A2C network with shared base and separate actor + critic heads
    """
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.fcc = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU()
        )

        self.actor = nn.Sequential(
            nn.Linear(256, action_dim)
        )

        self.critic = nn.Linear(256, 1)

    def forward(self, state):
        x = self.fcc(state)
        policy_logits = self.actor(x)
        value = self.critic(x)
        return policy_logits, value


env = gym.make("CartPole-v1")
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

model = ActorCritic(state_dim, action_dim)
optimizer = optim.Adam(model.parameters(), lr=LR)

epsilon = EPS_START
print("Starting A2C training...")

for episode in range(EPISODES):
    state, _ = env.reset()
    total_reward = 0

    log_probs = []
    values = []
    rewards = []
    entropies = []

    for step in range(MAX_STEPS_PER_EPISODE):
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        logits, value = model(state_tensor) # logits for actor and value for critic
        # from the actor we get logits -> this will be used in logprobs and entropy both
        # from the critic we get value -> this will be used in value loss and to find advantage
        # advantage will be used in policy loss
        # mse loss is easily value and return comparison
        probs = F.softmax(logits, dim=-1) # actor policy
        dist = torch.distributions.Categorical(probs)

        if random.random() < epsilon:
            action = env.action_space.sample()
        else:
            action = dist.sample().item()

        next_state, reward, done, truncated, _ = env.step(action)
        total_reward += reward

        log_prob = torch.log(probs[0, action] + 1e-8) # log probability of the action
        entropy = -(probs * torch.log(probs + 1e-8)).sum() # entropy of the actor policy

        log_probs.append(log_prob)
        values.append(value.squeeze()) # value for critic
        rewards.append(reward)
        entropies.append(entropy)

        state = next_state
        if done or truncated:
            break

    G = 0
    returns = []
    for r in reversed(rewards):
        G = r + GAMMA * G
        returns.insert(0, G)
    returns = torch.FloatTensor(returns)

    values = torch.stack(values)
    log_probs = torch.stack(log_probs)
    entropies = torch.stack(entropies)

    # Mean 
    returns = (returns - returns.mean()) / (returns.std() + 1e-8)

    advantages = returns - values.detach() # advantage = return - value

    policy_loss = -(log_probs * advantages).mean()
    value_loss = F.mse_loss(values, returns)
    entropy_loss = -ENTROPY_BETA * entropies.mean()

    loss = policy_loss + value_loss + entropy_loss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    epsilon = max(EPS_END, epsilon * EPS_DECAY)

    if episode % 10 == 0:
        print(f"[A2C] Episode {episode+1}, Reward: {total_reward:.1f}, Epsilon: {epsilon:.3f}")

env.close()
print("A2C Training complete 🎯")

env = gym.make("CartPole-v1", render_mode="human")
state, _ = env.reset()
done = False
total_reward = 0

while not done:
    with torch.no_grad():
        logits, _ = model(torch.FloatTensor(state).unsqueeze(0))
        probs = F.softmax(logits, dim=-1)
        action = probs.argmax().item()

    state, reward, done, _, _ = env.step(action)
    total_reward += reward

env.close()
print(f"Total Reward Achieved by A2C Agent: {total_reward} 🧠")
