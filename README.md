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
- **[🧠 ML Architecture Overview](./docs/ml-architecture-overview.md)** - Implementation status, research comparisons, complexity analysis, and learning resources
- **[🎮 User Guide](./docs/user-guide.md)** - Complete guide on how to use the website, all controls, interface explanation, and usage tips
- **[🛠️ Technical Stack](./docs/technical-stack.md)** - Technology details, project structure, dependencies, and development information
- **[🚀 Getting Started](./docs/getting-started.md)** - Installation, setup, configuration, and troubleshooting guide

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

## 🎮 How to Use

### Main Controls

- **START SIMULATION** - Begin agent navigation
- **Speed Slider** - Adjust simulation speed (1x-10x)
- **Curiosity Multiplier** - Control exploration behavior (0-25)
- **NN ON/OFF** - Toggle neural network learning
- **Environment Selector** - Choose maze type
- **NEW WORLD** - Generate new maze

### Interface

- **Left Panel** - Vision & Memory visualizations
- **Center** - Maze display with statistics
- **Right Panel** - Logs and strategy information

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
├── services/           # ML services (VAE, MDN-RNN, Policy Network)
├── utils/              # Utility functions
├── docs/               # Comprehensive documentation
├── types.ts            # TypeScript definitions
├── constants.ts        # Configuration
└── App.tsx             # Main application
```

---

## 🎯 Key Features

- ✅ **Neural Network Learning** - Deep learning for decision-making
- ✅ **World Model Architecture** - Vision, Memory, Controller framework
- ✅ **Curiosity-Driven Exploration** - Intrinsic motivation system
- ✅ **Multiple Environments** - 5 different environment types
- ✅ **Transfer Learning** - Knowledge sharing between environments
- ✅ **Real-time Visualization** - Interactive learning visualization
- ✅ **Experience Replay** - Advanced training system
- ✅ **Adjustable Parameters** - Speed, curiosity, and more

---

## 📚 Educational Value

This project demonstrates:

- Reinforcement Learning principles
- Neural network architectures
- World Models and predictive learning
- Exploration-exploitation tradeoffs
- Transfer learning concepts
- Curiosity-driven AI

Perfect for students, developers, and researchers interested in AI navigation and machine learning.

---

## 🔬 Research Background

Inspired by:

- **World Models (Ha & Schmidhuber, 2018)** - Original V-M-C architecture
- **Curiosity-Driven Learning (Pathak et al., 2017)** - Intrinsic motivation
- **REINFORCE Algorithm** - Policy gradient methods
- **Experience Replay** - DQN and related RL techniques

---

## 🤝 Contributing

Contributions are welcome! Areas for improvement:

- Additional environment types
- Enhanced visualizations
- Performance optimizations
- Documentation improvements

---

## 📝 License

This project is open source and available for educational and research purposes.

---

## 🔗 Quick Links

- **[Getting Started](./docs/getting-started.md)** - Installation and setup
- **[Architecture](./docs/architecture.md)** - ML architecture details
- **[Learning Algorithms](./docs/learning-algorithms.md)** - How the agent learns
- **[User Guide](./docs/user-guide.md)** - How to use the website
- **[Technical Stack](./docs/technical-stack.md)** - Technology details

---

**Enjoy exploring the fascinating world of AI navigation and machine learning!** 🚀
