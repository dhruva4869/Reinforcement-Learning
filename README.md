# CartPole Deep Q-Network (DQN) Reinforcement Learning

A simple implementation of Deep Q-Network (DQN) reinforcement learning algorithm to solve the CartPole-v1 environment from OpenAI Gymnasium.

## Overview

This project implements a Deep Q-Network (DQN) agent that learns to balance a pole on a cart by taking appropriate actions (left or right). The agent uses experience replay and target network techniques to stabilize learning.

## Environment

- **Environment**: CartPole-v1 from Gymnasium
- **State Space**: 4 continuous values (cart position, cart velocity, pole angle, pole angular velocity)
- **Action Space**: 2 discrete actions (push left, push right)
- **Goal**: Balance the pole for as long as possible (maximum 500 steps)

## Algorithm Details

### Deep Q-Network (DQN)
- **Architecture**: 4-layer fully connected neural network
  - Input: 4 (state dimensions)
  - Hidden layers: 128 → 256 → 128 neurons
  - Output: 2 (action dimensions)
  - Activation: ReLU and GELU

### Key Features
- **Experience Replay**: Stores and samples past experiences to break correlation
- **Target Network**: Separate target network updated periodically for stable learning
- **Epsilon-Greedy Exploration**: Balances exploration vs exploitation
- **Double DQN**: Uses target network for more stable Q-value estimation

## Hyperparameters

```python
LR = 1e-3                    # Learning rate
GAMMA = 0.99                 # Discount factor
EPS_START = 1.0              # Initial exploration rate
EPS_END = 0.01               # Final exploration rate
EPS_DECAY = 0.995            # Exploration decay rate
BATCH_SIZE = 64              # Training batch size
MEMORY_SIZE = 50000          # Replay buffer size
TARGET_UPDATE = 10           # Target network update frequency
EPISODES = 700               # Training episodes
MAX_STEPS_PER_EPISODE = 500  # Maximum steps per episode
```

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd "Reinforcement Learning"
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Training the Agent
```bash
python cartpole-simple-rl.py
```

The script will:
1. Train the DQN agent for 700 episodes
2. Print progress every 10 episodes
3. Show the trained agent playing the game with visualization

### What the Script Does

1. **Training Phase**:
   - Creates DQN and target networks
   - Runs episodes with epsilon-greedy exploration
   - Stores experiences in replay buffer
   - Trains the network using experience replay
   - Updates target network periodically

2. **Evaluation Phase**:
   - Loads the trained model
   - Runs the agent with greedy policy (no exploration)
   - Renders the environment to visualize performance

## Key Components

### DQN Class
- Neural network architecture for Q-value approximation
- Forward pass returns Q-values for all actions

### ReplayBuffer Class
- Stores agent experiences (state, action, reward, next_state, done)
- Samples random batches for training
- Helps break correlation between consecutive experiences

### Training Loop
- Epsilon-greedy action selection
- Experience storage
- Batch training with target network
- Periodic target network updates

## Expected Results

- The agent should learn to balance the pole for the full 500 steps
- Training typically shows improving performance over episodes
- Final evaluation should demonstrate stable pole balancing

## File Structure

```
├── cartpole-simple-rl.py    # Main training and evaluation script
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

## Dependencies

- `gymnasium[classic-control]`: OpenAI Gymnasium environments
- `torch`: PyTorch for neural networks
- `numpy`: Numerical computations

## Learning Process

1. **Initial Phase**: High exploration (ε=1.0), random actions
2. **Learning Phase**: Gradual decrease in exploration, more exploitation
3. **Convergence**: Low exploration (ε=0.01), mostly greedy actions

The agent learns through trial and error, gradually improving its policy to maximize cumulative reward.

## Tips for Improvement

- Adjust hyperparameters for different learning rates
- Experiment with network architecture
- Try different exploration strategies
- Implement additional DQN variants (Double DQN, Dueling DQN)
- Add reward shaping techniques

## Troubleshooting

- Ensure all dependencies are installed correctly
- Check that PyTorch is compatible with your system
- Monitor training progress through printed episode rewards
- Adjust hyperparameters if learning is too slow or unstable
