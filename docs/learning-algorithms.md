# 🎓 Learning Algorithms

This document explains the machine learning algorithms and training mechanisms used in the Dreaming Maze Solver.

---

## 1. Reinforcement Learning (RL)

### Overview

The agent learns through trial and error, receiving rewards for good actions and penalties for poor ones. This is the core learning mechanism that drives all improvements.

### Reward Function

The agent receives rewards based on its actions:

```
Total Reward = Step Cost + Goal Bonus + Curiosity Bonus
```

**Components:**

- **Step Cost:** `-1` for each step
  - Encourages finding the shortest path
  - Penalizes taking too many steps
  - Creates time pressure

- **Goal Bonus:** `+100` for reaching the goal
  - Primary objective reward
  - Large positive signal for success
  - Motivates goal-seeking behavior

- **Curiosity Bonus:** `+curiosityMultiplier` for visiting new areas
  - Encourages exploration
  - Adjustable via slider (0-25)
  - Intrinsic motivation mechanism

### Policy Gradient Learning

The Policy Network is trained using the **REINFORCE algorithm**:

1. **Collect Trajectory:** Agent collects (state, action, reward) sequences
2. **Compute Returns:** Calculate discounted future rewards
3. **Compute Gradients:** Estimate policy gradient
4. **Update Weights:** Update network using gradient ascent

### Experience Replay

Past experiences are stored and replayed for efficient learning:

- **Buffer Size:** Up to 10,000 experiences
- **Batch Sampling:** Random batches of 32 experiences
- **Benefits:**
  - Breaks correlation between consecutive experiences
  - Reuses past experiences multiple times
  - Improves sample efficiency

---

## 2. Curiosity-Driven Exploration

### Concept

The agent has an intrinsic motivation to explore novel areas, not just external rewards. This prevents the agent from getting stuck in local optima.

### Novelty Detection

- **Visited History:** Tracks all previously visited positions
- **New Area Detection:** Checks if current position is in history
- **Reward Calculation:** Awards curiosity bonus for novel areas

### Curiosity Multiplier

- **Range:** 0 to 25 (adjustable via UI)
- **Low (0-5):** Minimal exploration, focus on goal
- **Medium (5-15):** Balanced exploration and exploitation
- **High (15-25):** Extensive exploration, comprehensive mapping

### Impact on Behavior

- **Low Curiosity:** Direct paths, faster goal reaching, less exploration
- **High Curiosity:** Wandering paths, slower goal reaching, better map coverage

### Exploration-Exploitation Tradeoff

Curiosity creates a natural balance:
- **Exploration:** Visiting new areas (curiosity reward)
- **Exploitation:** Using known paths to goal (goal reward)

---

## 3. Epsilon-Greedy Strategy

### Overview

A classic exploration-exploitation strategy that balances random exploration with learned knowledge.

### Epsilon (ε) Parameter

- **Definition:** Probability of exploring (vs exploiting)
- **Range:** 0.15 to 1.0
- **Initial Value:** 1.0 (100% exploration)
- **Final Value:** 0.15 (15% exploration)

### Decay Formula

```
ε = max(0.15, 1 - steps/100)
```

### Behavior Over Time

- **Early Steps (0-50):** High epsilon (0.5-1.0)
  - Mostly random exploration
  - Discovers environment structure
  - Builds initial knowledge

- **Middle Steps (50-100):** Medium epsilon (0.15-0.5)
  - Mixed exploration and exploitation
  - Uses learned knowledge
  - Still explores occasionally

- **Late Steps (100+):** Low epsilon (0.15)
  - Mostly exploitation
  - Uses learned optimal paths
  - Minimal random exploration

### Decision Process

1. **Generate Random Number:** 0 to 1
2. **Compare to Epsilon:**
   - If random < ε: **Explore** (random action)
   - If random ≥ ε: **Exploit** (best known action)
3. **Execute Action**

---

## 4. Experience Replay Training System

### Architecture

Advanced training mechanism that stores and replays past experiences.

### Components

#### Replay Buffer

- **Size:** Up to 10,000 experiences
- **Structure:** Array of (state, action, reward, next_state, done) tuples
- **Management:** FIFO (First In, First Out) when full

#### Batch Training

- **Batch Size:** 32 experiences per training step
- **Sampling:** Random uniform sampling from buffer
- **Frequency:** Every 20 steps + at episode completion

#### Learning Rate Scheduling

- **Initial Rate:** 0.001
- **Decay Factor:** 0.995 per training step
- **Minimum Rate:** 0.0001
- **Purpose:** Stable convergence, prevents overshooting

### Training Process

1. **Experience Collection:** Store experiences during navigation
2. **Buffer Management:** Maintain buffer size limit
3. **Batch Sampling:** Sample random batch when training
4. **Gradient Computation:** Compute policy gradients
5. **Weight Update:** Update network weights
6. **Learning Rate Decay:** Reduce learning rate gradually

### Benefits

- **Sample Efficiency:** Reuse experiences multiple times
- **Stability:** Breaks correlation between consecutive samples
- **Convergence:** More stable learning curves
- **Memory:** Efficient use of past experiences

---

## 5. Transfer Learning

### Concept

Knowledge sharing between different environments, allowing the agent to apply learned knowledge to new tasks.

### Knowledge Base

Stores learned weights from:
- **Policy Network:** Decision-making knowledge
- **VAE:** Visual representation knowledge
- **MDN-RNN:** Temporal prediction knowledge

### Transfer Modes

#### Full Transfer

- **Method:** Copy all weights directly
- **Use Case:** Similar environments
- **Pros:** Fast adaptation
- **Cons:** May not fit perfectly

#### Partial Transfer

- **Method:** Copy only hidden layers, retrain output
- **Use Case:** Related but different environments
- **Pros:** Good balance
- **Cons:** Requires some retraining

#### Fine-Tuning

- **Method:** Transfer weights + continue training with lower LR
- **Use Case:** Different but related environments
- **Pros:** Best adaptation
- **Cons:** Requires more training

### Transfer Process

1. **Save Knowledge:** Store weights from trained networks
2. **Load Knowledge:** Load weights into new environment
3. **Adaptation:** Fine-tune or retrain as needed
4. **Evaluation:** Test performance in new environment

### Benefits

- **Faster Learning:** Start with good initial weights
- **Better Performance:** Improved initial performance
- **Knowledge Reuse:** Apply knowledge across environments
- **Generalization:** Better understanding of navigation

---

## 6. Neural Network Training

### Policy Network Training

#### Architecture

```
Input (8 features) → Dense(64) → ReLU → Dense(32) → ReLU → Output(4 actions)
```

#### Training Algorithm

- **Method:** REINFORCE (Policy Gradient)
- **Optimizer:** Adam (learning rate: 0.001)
- **Loss Function:** Policy gradient loss
- **Regularization:** None (simple architecture)

#### Training Schedule

- **Periodic Training:** Every 20 steps (if buffer has enough experiences)
- **Episode Training:** At episode completion
- **Batch Size:** 32 experiences
- **Epochs:** 3-10 per training session

### VAE Training

#### Architecture

- **Encoder:** Convolutional layers → Latent space
- **Decoder:** Latent space → Reconstruction

#### Training Process

- **Loss Function:** Reconstruction loss + KL divergence
- **Purpose:** Learn efficient latent representations
- **Update Frequency:** Continuous during navigation

### MDN-RNN Training

#### Architecture

- **LSTM Layers:** 256 → 128 units
- **Output:** Mixture density prediction

#### Training Process

- **Loss Function:** Prediction error
- **Purpose:** Learn temporal patterns
- **Update Frequency:** Continuous during navigation

---

## 7. Learning Metrics

### Performance Indicators

- **Steps to Goal:** Number of steps to reach goal
- **Reward Accumulation:** Total reward earned
- **Curiosity Score:** Total curiosity bonuses
- **Training Experiences:** Number of experiences collected
- **Epsilon Decay:** Exploration probability over time

### Learning Curves

- **Improvement Over Time:** Agent gets better with more experience
- **Path Efficiency:** Paths become shorter and more direct
- **Exploration Coverage:** Better map coverage with higher curiosity
- **Network Performance:** Policy network improves decision-making

---

## 8. Hyperparameters

### Key Parameters

- **Learning Rate:** 0.001 (with decay)
- **Batch Size:** 32
- **Buffer Size:** 10,000
- **Epsilon Decay:** Linear from 1.0 to 0.15 over 100 steps
- **Curiosity Range:** 0-25 (user adjustable)
- **Training Frequency:** Every 20 steps
- **Dream Frequency:** Every 5 steps

### Tuning Guidelines

- **Learning Rate:** Too high = unstable, too low = slow learning
- **Batch Size:** Larger = more stable, smaller = faster updates
- **Buffer Size:** Larger = more diverse experiences
- **Epsilon Decay:** Faster = less exploration, slower = more exploration

---

[← Back to README](../README.md)
