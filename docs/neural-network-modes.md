# 🧠 Neural Network Mode: ON vs OFF

Complete guide to understanding the difference between neural network learning mode and fixed algorithm mode.

---

## Overview

The **NN ON/OFF** toggle is a crucial feature that determines whether the agent uses machine learning to improve over time or relies on a fixed algorithm. Understanding this distinction is key to using the simulation effectively.

Both modes use the **Epsilon-Greedy** strategy, which balances exploration (trying new actions) and exploitation (using learned knowledge). The fundamental difference lies in what happens during the **exploitation phase**:

- **NN ON**: Uses a neural network that learns and improves from experience
- **NN OFF**: Uses a simple greedy algorithm that always behaves the same way

---

## How NN ON Works (Neural Network Learning)

When the neural network is **enabled**, the agent learns and improves over time through reinforcement learning.

### Input Features

The neural network receives 8 features for each decision:

- Agent position (X, Y)
- Goal position (X, Y)
- Distance to goal
- Percentage of explored area
- Curiosity level
- Epsilon value

These features provide the network with a complete picture of the current state, allowing it to make informed decisions.

### Action Selection

The agent uses an **Epsilon-Greedy** strategy with two phases:

1. **Exploration** (ε% of the time): Random action selection
   - Allows the agent to discover new paths
   - Prevents getting stuck in local optima
   - The epsilon value decreases over time (from 1.0 to 0.1)

2. **Exploitation** (1-ε% of the time): Neural network prediction
   - The network predicts probabilities for all 4 possible actions (up, down, left, right)
   - The action with the highest probability is chosen
   - These predictions improve as the network learns

### Learning Process - REINFORCE (Policy Gradient)

The neural network learns using the **REINFORCE** algorithm, a policy gradient method:

#### Experience Collection
- Each step is saved as an experience: `(state, action, reward)`
- Experiences are stored in a replay buffer
- This allows the network to learn from past actions

#### Training Schedule
- **Periodic Training**: Every 20 steps, the network trains on 32 random experiences (Experience Replay)
  - Random sampling prevents overfitting to recent experiences
  - Batch training is more stable than single-step updates
  
- **Episode Training**: At the end of each episode, the network trains on all collected experiences
  - Ensures the network learns from the complete episode
  - Helps consolidate learning from the entire navigation attempt

#### Weight Updates
- Network weights are adjusted based on rewards received
- Actions that lead to positive outcomes (reaching goal, exploring new areas) are reinforced
- Actions that lead to negative outcomes (hitting walls, going in circles) are discouraged
- The learning rate controls how quickly the network adapts

### Performance Improvement Over Time

The neural network's performance evolves through training:

- **Initially**: The network behaves randomly (weights are randomly initialized)
  - Predictions are essentially random
  - Similar performance to NN OFF mode
  
- **After a few episodes**: Begins learning patterns
  - Starts recognizing beneficial moves
  - Begins avoiding repeated mistakes
  - Performance starts to improve
  
- **After many episodes**: Significantly improves and finds shorter paths
  - Learns complex navigation strategies
  - Can find paths that greedy algorithms miss
  - Adapts to different maze layouts

**Example progression:**
- Episode 1: 200 steps
- Episode 2: 180 steps (improvement)
- Episode 3: 150 steps (improvement)
- Episode 10: 80 steps (significant improvement)

---

## How NN OFF Works (Fixed Algorithm)

When the neural network is **disabled**, the agent uses a simple, non-learning algorithm that always behaves consistently.

### Action Selection

The agent still uses **Epsilon-Greedy**, but with a simpler exploitation strategy:

1. **Exploration** (ε% of the time): Random action selection
   - Same as NN ON mode
   - May prefer unexplored areas when possible
   - Allows discovery of new paths

2. **Exploitation** (1-ε% of the time): Greedy distance calculation
   - Calculates Manhattan distance to goal for all possible moves
   - Chooses the move that minimizes distance to goal
   - Simple and deterministic

### Algorithm Details

The greedy algorithm works as follows:

```typescript
// Find all possible moves
const possibleMoves = getValidMoves(currentPosition);

// Calculate distance to goal for each move
let bestMove = possibleMoves[0];
let minDist = Infinity;

for (const move of possibleMoves) {
  const dist = Math.abs(move.x - goal.x) + Math.abs(move.y - goal.y);
  if (dist < minDist) {
    minDist = dist;
    bestMove = move;
  }
}

return bestMove; // Move closest to goal
```

### No Learning

Key characteristics of NN OFF mode:

- **Same algorithm every time**: Always uses the same greedy strategy
- **Does not learn from experience**: Past episodes don't influence behavior
- **Does not improve between episodes**: Performance remains constant
- **No experience collection**: No memory of past actions
- **No training**: No neural network updates

**Example progression:**
- Episode 1: 200 steps
- Episode 2: 200 steps (same)
- Episode 3: 200 steps (same)
- Episode 10: 200 steps (same)

---

## Comparison Table

| Feature | NN ON | NN OFF |
|---------|-------|--------|
| **Learning** | ✅ Yes - improves over time | ❌ No - fixed algorithm |
| **Experience Collection** | ✅ Yes - saves experiences | ❌ No |
| **Training** | ✅ Every 20 steps + end of episode | ❌ None |
| **Performance Improvement** | ✅ Yes - gets better | ❌ No - constant |
| **Complexity** | High - neural network | Low - simple rules |
| **Speed** | Slower (network computation) | Faster (simple calculation) |
| **Exploration Phase** | Random (same as NN OFF) | Random (same as NN ON) |
| **Exploitation Phase** | Neural network prediction (improves) | Greedy distance calculation (fixed) |
| **Memory Usage** | Higher (stores experiences) | Lower (no memory) |
| **Adaptability** | ✅ Adapts to different mazes | ❌ Same strategy always |
| **Best For** | Long-term learning, complex mazes | Quick testing, baseline comparison |

---

## Technical Details: Epsilon-Greedy in Both Modes

Both modes use the same **Epsilon-Greedy** strategy structure, but differ in the exploitation phase:

### Code Structure

```typescript
if (Math.random() < epsilon) {
    // Exploration: Random action
    return randomAction();
} else {
    // Exploitation: Different behavior based on mode
    if (neuralNetworkEnabled) {
        // NN ON: Use neural network predictions
        const probabilities = await neuralNetwork.predict(state);
        return actionWithHighestProbability(probabilities);
    } else {
        // NN OFF: Use greedy distance calculation
        return closestMoveToGoal();
    }
}
```

### Key Insight

The difference is **only** in the exploitation phase:

- **NN ON**: 
  - Uses learned neural network predictions
  - Starts random (weights are random)
  - Improves over time as network learns
  - Can discover complex strategies

- **NN OFF**: 
  - Uses simple distance calculation
  - Always the same behavior
  - No learning or adaptation
  - Simple and predictable

### Epsilon Decay

In both modes, epsilon typically starts at 1.0 (100% exploration) and decays over time:

- **Initial**: High exploration (ε = 1.0)
- **Gradual decay**: More exploitation as epsilon decreases
- **Final**: Low exploration (ε = 0.1)

This allows the agent to explore initially and then exploit learned knowledge.

---

## When to Use Each Mode

### Use NN ON When:

- ✅ You want to observe learning and improvement
- ✅ You want the agent to adapt and get better over time
- ✅ You're experimenting with transfer learning
- ✅ You want to see how neural networks learn navigation
- ✅ You have time for multiple episodes
- ✅ You're interested in machine learning research
- ✅ You want to see the agent discover complex strategies

### Use NN OFF When:

- ✅ You need quick testing
- ✅ You want a baseline for comparison
- ✅ You don't need learning capabilities
- ✅ You want consistent, predictable behavior
- ✅ You're debugging or testing specific scenarios
- ✅ You want faster simulation speed
- ✅ You're demonstrating the difference between learning and non-learning approaches

---

## Initial Behavior and Learning Curve

### Important Note

When NN ON is first enabled, the neural network starts with **random weights**, so its predictions are essentially random. This means:

- **Early episodes**: NN ON behaves similarly to NN OFF (both appear random)
  - Network hasn't learned yet
  - Predictions are not meaningful
  - Performance is similar to greedy algorithm

- **After training**: NN ON begins to outperform NN OFF as it learns patterns
  - Network starts recognizing good moves
  - Performance improves episode by episode
  - Begins finding better paths

- **Long-term**: NN ON can find more efficient paths than the simple greedy approach
  - Learns complex navigation strategies
  - Adapts to maze structure
  - Can discover paths that greedy algorithms miss

### Learning Curve Example

```
Episode  Performance (steps)
--------  -------------------
1         200 (random)
2         190 (starting to learn)
3         170 (improving)
4         150 (better)
5         130 (good progress)
10        80  (excellent)
20        60  (optimized)
```

The neural network needs time to learn, but once trained, it can discover complex strategies that the simple greedy algorithm cannot.

---

## Practical Tips

### For Best Results with NN ON:

1. **Run multiple episodes**: The network needs time to learn
2. **Be patient**: Early episodes may seem random
3. **Watch the improvement**: Observe how performance improves over time
4. **Try different environments**: Test transfer learning capabilities
5. **Adjust curiosity**: Higher curiosity encourages more exploration

### For Testing with NN OFF:

1. **Use for baselines**: Compare against learning performance
2. **Quick validation**: Test if mazes are solvable
3. **Debugging**: Isolate issues without learning complexity
4. **Consistent results**: Same behavior every time

---

## Summary

- **NN ON** = Learning mode with neural network that improves over time
- **NN OFF** = Fixed algorithm mode with consistent, non-learning behavior
- Both use **Epsilon-Greedy** strategy
- Difference is in the **exploitation phase**
- NN ON starts random but improves; NN OFF stays constant
- Choose based on your goals: learning vs. testing

**📖 Related Documentation:**
- [Learning Algorithms](./learning-algorithms.md) - Deep dive into reinforcement learning
- [User Guide](./user-guide.md) - How to use the NN ON/OFF toggle
- [Architecture](./architecture.md) - Neural network architecture details
