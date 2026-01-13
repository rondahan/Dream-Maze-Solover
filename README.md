# 🧠 Dreaming Maze Solver

An interactive visualization of an autonomous AI agent that learns to navigate mazes using a **World Model Architecture (V-M-C)** - a cutting-edge machine learning approach that combines Vision, Memory, and Controller components to create an intelligent navigation system.

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.8-blue)](https://www.typescriptlang.org/)
[![React](https://img.shields.io/badge/React-19.2-blue)](https://react.dev/)
[![TensorFlow.js](https://img.shields.io/badge/TensorFlow.js-4.22-orange)](https://www.tensorflow.org/js)

---

## 📋 Project Overview

The **Dreaming Maze Solver** is a web-based simulation that demonstrates how an AI agent can learn to solve mazes through reinforcement learning, curiosity-driven exploration, and predictive "dreaming." The agent uses neural networks to perceive its environment, predict future states, and make intelligent decisions about which path to take.

### What This Project Does

This project implements a complete **World Model** architecture where an AI agent:

1. **Perceives** the maze environment through a Vision component (VAE)
2. **Predicts** future states through a Memory component (MDN-RNN) 
3. **Decides** actions through a Controller component (Policy Network)
4. **Learns** from experience using reinforcement learning
5. **Explores** novel areas driven by intrinsic curiosity
6. **Transfers** knowledge between different environment types

The agent starts with no knowledge and gradually improves its navigation strategy through trial and error, learning from rewards and building an internal "world model" that allows it to predict and plan ahead.

---

## 🚀 Quick Start

### Prerequisites

- **Node.js 18+** installed
- Modern web browser (Chrome, Firefox, Safari, Edge)

### Installation

```bash
# Install dependencies
npm install

# Start development server
npm run dev

# Build for production
npm run build
```

Open `http://localhost:5173` in your browser.

**📖 For detailed setup instructions, see [Getting Started Guide](./docs/getting-started.md)**

---

## 📚 Documentation

Comprehensive documentation is available in the [`docs/`](./docs/) folder:

### Core Documentation

- **[🏗️ Architecture](./docs/architecture.md)** - Detailed explanation of the V-M-C framework, neural network components, and how they work together
- **[🎓 Learning Algorithms](./docs/learning-algorithms.md)** - Deep dive into reinforcement learning, curiosity-driven exploration, epsilon-greedy strategy, and training mechanisms
- **[🧠 Neural Network Modes](./docs/neural-network-modes.md)** - Complete guide to NN ON vs NN OFF: understanding learning mode vs fixed algorithm mode
- **[🧠 ML Architecture Overview](./docs/ml-architecture-overview.md)** - Implementation status, research comparisons, complexity analysis, and learning resources
- **[🎮 User Guide](./docs/user-guide.md)** - Complete guide on how to use the website, all controls, interface explanation, and usage tips
- **[🛠️ Technical Stack](./docs/technical-stack.md)** - Technology details, project structure, dependencies, and development information
- **[🚀 Getting Started](./docs/getting-started.md)** - Installation, setup, configuration, and troubleshooting guide
- **[🔄 Transfer Learning](./docs/transfer-learning.md)** - Complete guide to SAVE KNOWLEDGE button and transfer learning capabilities
- **[📋 Logs Reference](./docs/logs-reference.md)** - Understanding Controller (C) Logs: all message types, formats, and meanings

---

## 🏗️ Machine Learning Architecture

The project implements a **Vision-Memory-Controller (V-M-C)** architecture:

### V - Vision (VAE)
Compresses maze state into a compact latent representation using Variational Autoencoders.

### M - Memory (MDN-RNN)
Predicts future states and enables "dreaming" about possible paths using LSTM networks.

### C - Controller (Policy Network)
Decides which action to take using a deep neural network trained with reinforcement learning.

**📖 Learn more: [Architecture Documentation](./docs/architecture.md)**

---

## 🎓 Learning Algorithms

The agent uses several advanced learning techniques:

- **Reinforcement Learning** - Learns from rewards and penalties
- **Curiosity-Driven Exploration** - Intrinsic motivation to explore
- **Epsilon-Greedy Strategy** - Balances exploration vs exploitation
- **Experience Replay** - Efficient learning from past experiences
- **Transfer Learning** - Shares knowledge between environments

**📖 Learn more: [Learning Algorithms Documentation](./docs/learning-algorithms.md)**

---

## 🤖 Available Algorithms

The project supports **7 different algorithms** for navigation and learning:

### Machine Learning Algorithms (NN ON)

1. **REINFORCE** - Policy gradient method using Monte Carlo returns
   - On-policy algorithm
   - Uses epsilon-greedy exploration
   - Learns from complete episodes

2. **DQN (Deep Q-Network)** - Value-based learning with Q-function approximation
   - Off-policy algorithm
   - Uses target network for stability
   - Experience replay for sample efficiency

3. **Actor-Critic** - Combines policy and value learning
   - On-policy algorithm
   - Supports both epsilon-greedy and softmax exploration
   - Separate actor (policy) and critic (value) networks

4. **PPO (Proximal Policy Optimization)** - Advanced policy gradient with clipping
   - On-policy algorithm
   - Uses softmax exploration (required)
   - Clipped objective for stable learning

5. **A3C (Asynchronous Advantage Actor-Critic)** - Advanced Actor-Critic variant
   - On-policy algorithm
   - Uses n-step returns for better value estimation
   - Entropy regularization for exploration

6. **SAC (Soft Actor-Critic)** - Maximum entropy reinforcement learning
   - Off-policy algorithm
   - Twin Q-networks for better value estimation
   - Entropy regularization for natural exploration

### Classical Algorithms (NN OFF)

7. **A* Pathfinding** - Heuristic search algorithm
   - Deterministic pathfinding (no learning)
   - Uses Manhattan distance heuristic
   - Finds optimal paths efficiently
   - Available even when NN is OFF

8. **Epsilon-Greedy** - Simple exploration-exploitation strategy
   - Fixed greedy algorithm (no learning)
   - Default when NN is OFF
   - Balances random exploration with greedy exploitation

### Algorithm Comparison

| Algorithm | Type | Exploration | Learning | Best For |
|-----------|------|-------------|----------|----------|
| **REINFORCE** | On-policy | Epsilon-greedy | Policy gradient | Simple, stable learning |
| **DQN** | Off-policy | Epsilon-greedy | Value-based | Sample efficiency |
| **Actor-Critic** | On-policy | Epsilon-greedy / Softmax | Policy + Value | Balanced learning |
| **PPO** | On-policy | Softmax (required) | Policy gradient | Stable, advanced learning |
| **A3C** | On-policy | Softmax | Policy + Value (n-step) | Advanced value estimation |
| **SAC** | Off-policy | Softmax (entropy) | Maximum entropy | Natural exploration |
| **A*** | Deterministic | N/A | None | Optimal pathfinding |
| **Epsilon-Greedy** | Fixed | Epsilon-greedy | None | Baseline comparison |

**📖 Learn more: [Learning Algorithms Documentation](./docs/learning-algorithms.md)**

---

## 🌍 Environment Types

The project supports **5 different environment types** to test algorithm performance across various scenarios:

1. **Maze** - Classic recursive backtracking maze
   - Complex paths with walls and corridors
   - Single optimal solution path
   - Challenges: Navigation through narrow passages

2. **Open Field** - Wide open space with minimal obstacles
   - Mostly open paths with few walls
   - Multiple possible routes
   - Challenges: Efficient navigation in open space

3. **Corridor** - Winding corridor-like paths
   - Narrow, winding routes
   - Linear progression with turns
   - Challenges: Following winding paths efficiently

4. **Spiral** - Spiral pattern environment
   - Circular/spiral path structure
   - Unique geometric challenge
   - Challenges: Understanding spiral navigation

5. **Grid** - Regular grid structure
   - Regular grid pattern with multiple paths
   - Predictable structure
   - Challenges: Choosing optimal path in grid

Each environment type tests different aspects of navigation and learning, allowing you to compare algorithm performance across diverse scenarios.

---

## 🧠 Neural Network Mode: ON vs OFF

The **NN ON/OFF** toggle determines whether the agent uses machine learning to improve over time or relies on a fixed algorithm. Both modes use the **Epsilon-Greedy** strategy, but differ in the exploitation phase:

- **NN ON**: Uses a neural network that learns and improves from experience
- **NN OFF**: Uses a simple greedy algorithm with consistent behavior

**Key Differences:**
- NN ON learns from experience and improves over episodes
- NN OFF uses a fixed greedy algorithm that doesn't change
- Both use epsilon-greedy, but NN ON's exploitation improves with training

**📖 Complete guide: [Neural Network Modes Documentation](./docs/neural-network-modes.md)**

---

## 🎮 How to Use

### Main Controls

- **START SIMULATION** - Begin agent navigation
- **Speed Slider** - Adjust simulation speed (1x-10x)
- **Curiosity Multiplier** - Control exploration behavior (0-25)
- **NN ON/OFF** - Toggle neural network learning
- **Algorithm Selector** - Choose from 7 different algorithms (when NN is ON)
  - REINFORCE, DQN, Actor-Critic, PPO, A3C, SAC, A* Pathfinding
- **Environment Selector** - Choose maze type (5 different environments)
- **NEW WORLD** - Generate new maze
- **SAVE KNOWLEDGE** - Save learned knowledge for transfer learning

### Interface

- **Left Panel** - Vision & Memory visualizations
- **Center** - Maze display with statistics
- **Right Panel** - Controller (C) Logs and strategy information

**📖 Learn more: [Logs Reference](./docs/logs-reference.md)**

**📖 Complete guide: [User Guide](./docs/user-guide.md)**

---

## 🛠️ Technology Stack

- **Frontend:** React 19.2.3 with TypeScript
- **Build Tool:** Vite 6.2.0
- **Machine Learning:** TensorFlow.js 4.22.0
- **Styling:** Tailwind CSS

**📖 Technical details: [Technical Stack Documentation](./docs/technical-stack.md)**

---

## 📁 Project Structure

```
├── components/          # React UI components
│   ├── MazeBoard.tsx          # Main maze visualization
│   ├── LatentVisualizer.tsx   # VAE latent space visualization
│   └── DreamState.tsx         # MDN-RNN dream predictions
├── services/           # ML services and algorithms
│   ├── vae.ts                 # Vision: Variational Autoencoder
│   ├── mdnRnn.ts              # Memory: MDN-RNN for predictions
│   ├── policyNetwork.ts       # Controller: REINFORCE algorithm
│   ├── dqn.ts                 # Deep Q-Network algorithm
│   ├── actorCritic.ts         # Actor-Critic algorithm
│   ├── ppo.ts                 # Proximal Policy Optimization
│   ├── a3c.ts                 # Asynchronous Advantage Actor-Critic
│   ├── sac.ts                 # Soft Actor-Critic
│   ├── astar.ts               # A* Pathfinding algorithm
│   ├── trainingSystem.ts      # Experience replay system
│   ├── transferLearning.ts   # Knowledge transfer utilities
│   └── geminiService.ts       # AI monologue generation (optional)
├── utils/              # Utility functions
│   ├── mazeGenerator.ts       # Maze generation algorithms
│   ├── environmentGenerator.ts # Environment type generators
│   └── tfLoader.ts            # TensorFlow.js lazy loading
├── docs/               # Comprehensive documentation
│   ├── architecture.md        # V-M-C architecture details
│   ├── learning-algorithms.md # Algorithm explanations
│   ├── neural-network-modes.md # NN ON vs OFF guide
│   ├── ml-architecture-overview.md # Implementation status
│   ├── user-guide.md          # Complete user manual
│   ├── technical-stack.md     # Technology details
│   ├── getting-started.md     # Setup guide
│   ├── transfer-learning.md   # Knowledge transfer guide
│   └── logs-reference.md      # Log messages reference
├── types.ts            # TypeScript type definitions
├── constants.ts        # Configuration constants
└── App.tsx             # Main application component
```

---

## 🎯 Key Features

### Core Architecture
- ✅ **World Model Architecture** - Vision, Memory, Controller (V-M-C) framework
- ✅ **Neural Network Learning** - Deep learning for decision-making
- ✅ **7 Different Algorithms** - REINFORCE, DQN, Actor-Critic, PPO, A3C, SAC, A* Pathfinding
- ✅ **Multiple Learning Modes** - Neural network learning (NN ON) or fixed algorithms (NN OFF)

### Learning & Exploration
- ✅ **Curiosity-Driven Exploration** - Intrinsic motivation system with adjustable multiplier
- ✅ **Experience Replay** - Advanced training system with buffer management
- ✅ **Transfer Learning** - Knowledge sharing between environments
- ✅ **Multiple Exploration Strategies** - Epsilon-greedy, softmax, and algorithm-specific methods

### Environments & Visualization
- ✅ **5 Environment Types** - Maze, Open Field, Corridor, Spiral, Grid
- ✅ **Real-time Visualization** - Interactive learning visualization
- ✅ **VAE Latent Space Visualization** - See what the Vision component "sees"
- ✅ **Dream Predictions** - MDN-RNN future state predictions
- ✅ **Comprehensive Logging** - Detailed real-time logs of agent activity

### Controls & Customization
- ✅ **Adjustable Parameters** - Speed (1x-10x), curiosity (0-25), algorithm selection
- ✅ **Dynamic Speed Control** - Real-time simulation speed adjustment
- ✅ **Algorithm-Specific Settings** - Customizable exploration types for different algorithms
- ✅ **Knowledge Persistence** - Save and load learned knowledge

---

## 📚 Educational Value

This project demonstrates:

### Machine Learning Concepts
- **Reinforcement Learning principles** - Policy gradients, value functions, Q-learning
- **Neural network architectures** - Deep learning for sequential decision making
- **World Models and predictive learning** - Building internal models of the environment
- **On-policy vs Off-policy learning** - Different learning paradigms
- **Experience replay** - Efficient learning from past experiences

### Algorithm Comparison
- **7 Different Algorithms** - Compare performance across different RL approaches
- **Exploration-exploitation tradeoffs** - See how different strategies affect learning
- **Algorithm-specific features** - Understand unique characteristics of each method

### Advanced Topics
- **Transfer learning concepts** - Knowledge sharing between environments
- **Curiosity-driven AI** - Intrinsic motivation for exploration
- **Latent space representations** - Understanding compressed state representations
- **Predictive modeling** - Using RNNs for future state prediction

Perfect for students, developers, and researchers interested in AI navigation, reinforcement learning, and machine learning algorithms.

---

## 🔬 Research Background

Inspired by cutting-edge research in reinforcement learning and world models:

### Core Architecture
- **World Models (Ha & Schmidhuber, 2018)** - Original V-M-C architecture
  - Vision component using Variational Autoencoders
  - Memory component using MDN-RNN for state prediction
  - Controller component for action selection

### Learning Algorithms
- **REINFORCE Algorithm** - Policy gradient methods (Williams, 1992)
- **DQN (Mnih et al., 2015)** - Deep Q-Network with experience replay
- **Actor-Critic Methods** - Combining policy and value learning
- **PPO (Schulman et al., 2017)** - Proximal Policy Optimization
- **A3C (Mnih et al., 2016)** - Asynchronous Advantage Actor-Critic
- **SAC (Haarnoja et al., 2018)** - Soft Actor-Critic with maximum entropy

### Exploration & Motivation
- **Curiosity-Driven Learning (Pathak et al., 2017)** - Intrinsic motivation
- **Epsilon-Greedy Strategy** - Classic exploration-exploitation balance

### Pathfinding
- **A* Algorithm (Hart et al., 1968)** - Optimal pathfinding with heuristics

---

## 🤝 Contributing

Contributions are welcome! Areas for improvement:

### Algorithm Enhancements
- Additional RL algorithms (TD3, IMPALA, etc.)
- Algorithm-specific hyperparameter tuning
- Performance comparisons and benchmarks

### Features
- Additional environment types
- Enhanced visualizations (algorithm performance graphs, learning curves)
- Multi-agent scenarios
- Custom reward function editor

### Technical
- Performance optimizations
- Web Workers for parallel training
- Model compression and quantization
- Documentation improvements

---

## 📝 License

This project is open source and available for educational and research purposes.

---

## 🔗 Quick Links

### Getting Started
- **[Getting Started](./docs/getting-started.md)** - Installation and setup guide
- **[User Guide](./docs/user-guide.md)** - Complete guide on how to use the website

### Architecture & Algorithms
- **[Architecture](./docs/architecture.md)** - V-M-C architecture details
- **[Learning Algorithms](./docs/learning-algorithms.md)** - How the agent learns
- **[Neural Network Modes](./docs/neural-network-modes.md)** - NN ON vs NN OFF explained
- **[ML Architecture Overview](./docs/ml-architecture-overview.md)** - Implementation status and comparisons

### Features & Usage
- **[Transfer Learning](./docs/transfer-learning.md)** - SAVE KNOWLEDGE and transfer learning guide
- **[Logs Reference](./docs/logs-reference.md)** - Understanding Controller (C) Logs
- **[Technical Stack](./docs/technical-stack.md)** - Technology details and dependencies

---

**Enjoy exploring the fascinating world of AI navigation and machine learning!** 🚀
