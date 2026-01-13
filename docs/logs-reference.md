# 📋 Logs Reference

Complete guide to understanding the Controller (C) Logs panel and all message types.

---

## Overview

The **Controller (C) Logs** panel (right side of the interface) provides real-time information about what the agent is doing, thinking, and experiencing. All messages are color-coded by type and timestamped for easy tracking.

---

## Log Panel Location

**Right Panel** - Below the Strategy Mode indicator

**Features:**
- Scrollable log history (up to 50 messages)
- Color-coded by message type
- Timestamped entries
- Real-time updates

---

## Message Types

### 🔵 ACTION (Blue)

**Purpose:** Records agent actions, decisions, and training activities.

**Common Messages:**

#### Action Selection
```
Action: DOWN | Mode: EXPLOIT | Algorithm: REINFORCE | ε=0.88
```
- **Action:** Direction chosen (UP/DOWN/LEFT/RIGHT)
- **Mode:** Current strategy mode
  - `EXPLORE` = Seeking new areas
  - `EXPLOIT` = Using learned knowledge
- **Algorithm:** Learning algorithm in use
- **ε (Epsilon):** Exploration probability (1.0 = full exploration, 0.15 = mostly exploitation)

#### Reward Information
```
Reward: -1 (step cost)
Reward: -1 (step) + 5 (curiosity) = 4
🎯 GOAL REACHED! Reward: -1 (step) + 100 (goal) = 99
```
- **Step Cost:** -1 for each action (encourages efficiency)
- **Curiosity Bonus:** +X for discovering new areas
- **Goal Bonus:** +100 for reaching the goal

#### State Information
```
State: Agent[5,7] → Goal[12,3] | Distance: 7 | Visited: 45/225 (20.0%)
```
- Current agent position
- Goal position
- Distance to goal
- Percentage of maze visited

#### Training Information
```
Training buffer: 20 experiences collected
REINFORCE: Trained on 50 experiences (episode end)
DQN: Trained on 50 experiences (Q-learning update)
```
- Number of experiences collected
- Training algorithm used
- Training method description

#### Mode Transitions
```
Mode transition: EXPLORE → EXPLOIT
```
- When agent switches between exploration and exploitation

#### Knowledge Saving
```
Attempting to save knowledge...
✓ Policy Network saved
✓ VAE (Vision) saved
✓ MDN-RNN (Memory) saved
✅ Knowledge saved successfully!
```

---

### 🟢 VISION (Green)

**Purpose:** Information from the Vision component (VAE - Variational Autoencoder).

**Common Messages:**

#### Encoding Status
```
VAE: Encoded state → latent[0.23, -0.45, 0.67...] (dim=8)
```
- Shows latent vector values (compressed maze representation)
- Dimension count (usually 8)

#### Optimization Messages
```
VAE: Reusing previous latent (encoding every 3 steps)
```
- VAE encodes every 3 steps to save computation
- Reuses previous encoding between steps

#### Initialization
```
VAE (Vision) initialized. Encoding maze states.
VAE initialization skipped (will use fallback).
```

---

### 🟣 DREAM (Purple)

**Purpose:** Information from the Memory component (MDN-RNN) about future predictions.

**Common Messages:**

#### Prediction Start
```
MDN-RNN: Predicting future sequence (5 steps ahead)
```
- Starting a prediction sequence
- Number of steps being predicted

#### Prediction Results
```
MDN-RNN: Dream sequence generated (5 steps, confidence: 87.3%, end: [8,12])
```
- Number of steps predicted
- Confidence level (0-100%)
- Predicted end position

#### Internal State
```
Internal State: Novelty seeking (ε=0.88, Curiosity x5).
```
- Current internal motivation
- Epsilon value
- Curiosity multiplier

#### Monologue Updates
```
Generating internal monologue...
Monologue updated
```
- Updates from Gemini API (if enabled)
- Agent's "thoughts" about current state

#### Fallback
```
Dream: Using simple prediction (MDN-RNN disabled)
```
- Using fallback when MDN-RNN is unavailable

---

### 🔴 ERROR (Red)

**Purpose:** Error messages and warnings.

**Common Messages:**

#### MDN-RNN Errors
```
MDN-RNN: Prediction failed, using fallback
```
- **Meaning:** MDN-RNN prediction encountered an error
- **Impact:** System continues with fallback prediction
- **Not Critical:** Agent continues functioning normally
- **Note:** This is handled gracefully - the system is designed to work even if MDN-RNN fails

#### Saving Errors
```
Cannot save: Missing components: Policy Network, VAE
Cannot save: Neural Network is OFF. Turn NN ON to enable knowledge saving.
❌ Error saving knowledge: [error details]
```

#### Training Errors
```
Training error: [error details]
Periodic training error: [error details]
```

#### Initialization Errors
```
VAE initialization failed or timed out
MDN-RNN initialization failed or timed out
[Algorithm] initialization failed.
```

---

## Understanding Log Flow

### Typical Episode Flow

1. **Initialization**
   ```
   System initialized. Maze environment generated.
   UI ready. Loading neural networks in background...
   REINFORCE (Policy Network) initialized.
   VAE (Vision) initialized.
   MDN-RNN (Memory) initialized.
   ```

2. **Action Selection** (every step)
   ```
   State: Agent[5,7] → Goal[12,3] | Distance: 7
   Action: DOWN | Mode: EXPLORE | Algorithm: REINFORCE | ε=0.95
   Reward: -1 (step) + 5 (curiosity) = 4
   ```

3. **Periodic Updates** (every 5-10 steps)
   ```
   VAE: Encoded state → latent[...]
   MDN-RNN: Predicting future sequence (5 steps ahead)
   Training buffer: 20 experiences collected
   ```

4. **Training** (every 20 steps or episode end)
   ```
   Periodic training: REINFORCE on last 20 steps
   REINFORCE: Trained on 50 experiences (episode end)
   ```

5. **Goal Reached**
   ```
   🎯 GOAL REACHED! Reward: -1 (step) + 100 (goal) = 99
   Goal state achieved. Terminating episode.
   ```

---

## Log Message Format

All log messages follow this structure:

```
[Timestamp] TYPE
Message content
```

**Example:**
```
[12:14:32 AM] ACTION
Action: DOWN | Mode: EXPLOIT | Algorithm: REINFORCE | ε=0.88
```

---

## Tips for Reading Logs

1. **Color Coding** - Use colors to quickly identify message types
2. **Timestamps** - Track when events occur relative to each other
3. **Error Messages** - Red messages indicate issues, but many are non-critical
4. **Training Messages** - Look for training confirmations to verify learning
5. **Mode Transitions** - Watch for EXPLORE ↔ EXPLOIT switches

---

## Common Patterns

### Successful Learning
```
Action: ... | Mode: EXPLORE
Reward: -1 (step) + 5 (curiosity)
Training buffer: 20 experiences collected
REINFORCE: Trained on 50 experiences
```

### Goal Reached
```
Action: ... | Mode: EXPLOIT
🎯 GOAL REACHED! Reward: -1 (step) + 100 (goal) = 99
Goal state achieved. Terminating episode.
```

### MDN-RNN Fallback (Normal)
```
MDN-RNN: Predicting future sequence (5 steps ahead)
MDN-RNN: Prediction failed, using fallback
Dream: Using simple prediction (MDN-RNN disabled)
```
**Note:** This is normal and non-critical. The system continues working.

---

## Troubleshooting

### Too Many Logs
- Logs are limited to 50 most recent messages
- Older messages are automatically removed

### Missing Logs
- Check if simulation is running
- Verify networks are initialized
- Check browser console for errors

### Error Messages
- **Red messages** don't always mean failure
- Many errors have automatic fallbacks
- System continues functioning even with some errors

---

## Related Documentation

- **[User Guide](./user-guide.md)** - Complete interface guide
- **[Architecture](./architecture.md)** - V-M-C architecture details
- **[Transfer Learning](./transfer-learning.md)** - Knowledge saving guide
- **[Learning Algorithms](./learning-algorithms.md)** - How the agent learns

---

**Note:** The log system is designed to be informative but not overwhelming. Critical errors are highlighted in red, but many are handled gracefully with fallback mechanisms.
