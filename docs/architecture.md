# 🏗️ Machine Learning Architecture

## The V-M-C Framework

The project implements a **Vision-Memory-Controller (V-M-C)** architecture, inspired by the World Models research paper. This architecture consists of three interconnected neural network components that work together to enable intelligent navigation.

---

## V - Vision (Variational Autoencoder - VAE)

### Purpose

The Vision component compresses the maze state into a compact latent representation, allowing the agent to work with essential information rather than the full maze grid.

### How It Works

- The VAE uses convolutional neural networks (CNNs) to encode the maze grid into an 8-dimensional latent vector
- This latent vector captures the essential information about the agent's current state
- The decoder reconstructs a 5×5 visualization showing how the Vision component "sees" the environment
- This compression allows the agent to work with a compact representation rather than the full maze grid

### Technical Details

- **Encoder Architecture:** Convolutional layers that compress maze state → latent vector (8 dimensions)
- **Decoder Architecture:** Convolutional layers that reconstruct latent vector → visual representation
- **Latent Space:** 8-dimensional vector representing compressed state information
- **Input:** Full maze grid (15×15 cells)
- **Output:** Latent vector (8 dims) + reconstruction visualization (5×5)

### Implementation

The VAE is implemented using TensorFlow.js with:
- Convolutional layers for spatial feature extraction
- Dense layers for latent space encoding
- Variational sampling for robust representations
- Reconstruction loss for training

---

## M - Memory (MDN-RNN - Mixture Density Network Recurrent Neural Network)

### Purpose

The Memory component predicts future states and enables "dreaming" about possible paths, allowing the agent to plan ahead before taking actions.

### How It Works

- The MDN-RNN uses LSTM (Long Short-Term Memory) networks to predict future latent states
- Given the current latent vector and a sequence of actions, it predicts what the next states will be
- This allows the agent to "dream" about different paths before actually taking them
- Every 5 steps, the Memory component generates a dream prediction showing the likely path forward

### Technical Details

- **Architecture:** Two-layer LSTM (256 → 128 units) with mixture density output
- **Input:** Latent vector (8 dims) + action encoding (4 dims) = 12 dimensions
- **Output:** Predicted next latent vector (8 dimensions)
- **Dream Sequence:** Predicts 5 steps ahead with confidence scores
- **Temporal Modeling:** Maintains hidden state across time steps

### Implementation

The MDN-RNN is implemented using TensorFlow.js with:
- LSTM layers for temporal sequence modeling
- State management (hidden state and cell state)
- Action encoding (one-hot vectors for 4 directions)
- Sequence prediction for multi-step planning

### Dream Generation

Every 5 steps, the Memory component:
1. Takes the current latent vector
2. Generates a sequence of 5 predicted actions
3. Predicts the resulting latent states
4. Calculates confidence scores
5. Displays the dream path visualization

---

## C - Controller (Policy Network)

### Purpose

The Controller component decides which action to take based on the current state, learning optimal policies through reinforcement learning.

### How It Works

- The Policy Network is a deep neural network that takes state features as input
- It outputs probabilities for each possible action (up, down, left, right)
- The network learns through reinforcement learning, improving its decisions based on rewards
- Can operate in two modes:
  - **Neural Network Mode:** Uses the trained policy network for decision-making
  - **Epsilon-Greedy Mode:** Classic exploration-exploitation strategy

### Technical Details

- **Architecture:** Fully connected network (Input → 64 → 32 → 4 outputs)
- **Input Features:** 
  - Agent position (normalized)
  - Goal position (normalized)
  - Distance to goal
  - Visited ratio
  - Curiosity multiplier
  - Epsilon value
- **Output:** Action probabilities for 4 directions
- **Training:** REINFORCE algorithm with experience replay
- **Activation:** ReLU for hidden layers, softmax for output

### Decision Making Process

1. **State Feature Extraction:** Converts maze state into numerical features
2. **Forward Pass:** Network computes action probabilities
3. **Action Selection:** Samples action based on probabilities (or uses epsilon-greedy)
4. **Experience Storage:** Stores state, action, reward for training
5. **Policy Update:** Trains network using collected experiences

### Training Process

- **Experience Collection:** Stores (state, action, reward, next_state) tuples
- **Batch Sampling:** Randomly samples batches from experience buffer
- **Gradient Computation:** Computes policy gradients using REINFORCE
- **Weight Update:** Updates network weights using Adam optimizer
- **Learning Rate Decay:** Gradually reduces learning rate for stability

---

## Component Interaction

### The V-M-C Loop

1. **Vision (V):** Encodes current maze state → latent vector
2. **Memory (M):** Predicts future states from latent vector + actions
3. **Controller (C):** Decides action based on state features
4. **Action Execution:** Agent moves in the maze
5. **Reward Calculation:** Computes reward based on step, curiosity, goal
6. **Experience Storage:** Stores experience for training
7. **Network Training:** Updates all three networks periodically

### Information Flow

```
Maze State → VAE (Vision) → Latent Vector
                ↓
         MDN-RNN (Memory) → Dream Prediction
                ↓
      Policy Network (Controller) → Action
                ↓
            Maze Update → New State
```

---

## Environment Types

The agent can learn in 5 different environment types, each presenting unique challenges:

### 1. Maze
- **Type:** Classic recursive backtracking maze
- **Characteristics:** Walls and paths, single solution
- **Challenge:** Finding optimal path through complex structure

### 2. Open Field
- **Type:** Sparse obstacles in open space
- **Characteristics:** Large open areas, few obstacles
- **Challenge:** Efficient navigation in open space

### 3. Corridor
- **Type:** Winding corridor-like paths
- **Characteristics:** Narrow paths, winding routes
- **Challenge:** Following corridors without getting lost

### 4. Spiral
- **Type:** Circular spiral patterns
- **Characteristics:** Circular movement, center-focused
- **Challenge:** Navigating spiral patterns efficiently

### 5. Grid
- **Type:** Checkerboard-style grid patterns
- **Characteristics:** Regular grid structure, multiple paths
- **Challenge:** Choosing optimal path in grid structure

---

## Transfer Learning

The architecture supports knowledge transfer between environments:

### Knowledge Base

- Stores learned weights from all three components:
  - Policy Network weights
  - VAE encoder/decoder weights
  - MDN-RNN weights

### Transfer Modes

1. **Full Transfer:** Copy all weights directly
2. **Partial Transfer:** Copy only hidden layers, retrain output
3. **Fine-Tuning:** Transfer weights and continue training with lower learning rate

### Benefits

- Faster learning in new environments
- Better initial performance
- Knowledge sharing across environment types
- Improved generalization

---

## Research Background

This architecture is inspired by:

- **World Models (Ha & Schmidhuber, 2018):** Original V-M-C architecture paper
- **Variational Autoencoders:** Latent space representation learning
- **LSTM Networks:** Temporal sequence prediction
- **Policy Gradient Methods:** REINFORCE and related algorithms

---

## Related Documentation

- **[ML Architecture Overview](./ml-architecture-overview.md)** - Implementation status, research comparisons, and complexity analysis
- **[Learning Algorithms](./learning-algorithms.md)** - Detailed learning algorithm explanations
- **[User Guide](./user-guide.md)** - How to use the application

---

[← Back to README](../README.md)
