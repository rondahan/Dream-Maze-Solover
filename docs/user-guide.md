# 🎮 User Guide

Complete guide on how to use the Dreaming Maze Solver website.

---

## Initial Setup

When you first open the website, you'll see:

- **Center:** A randomly generated maze with agent (blue dot), start (green), and goal (orange)
- **Left Panel:** Vision & Memory visualization components
- **Right Panel:** Logs and strategy information
- **Top Bar:** Control buttons and sliders

---

## Main Controls

### 1. START SIMULATION / HALT PROCESS Button

**Location:** Top control bar

**Function:**
- **Green "START SIMULATION":** Begins the agent's navigation
- **Red "HALT PROCESS":** Pauses the simulation at any time
- Automatically stops when goal is reached

**Usage:**
1. Click "START SIMULATION" to begin
2. Watch the agent navigate the maze
3. Click "HALT PROCESS" to pause if needed
4. Click "START SIMULATION" again to resume

**Tips:**
- Let it run to completion to see full learning process
- Pause to examine the current state
- Restart with "NEW WORLD" to try again

---

### 2. Speed Control Slider

**Location:** Top control bar (next to Curiosity slider)

**Range:** 1x to 10x speed

**Speed Levels:**
- **1x:** Normal speed (300ms per step) - Best for observing details
- **5x:** Fast speed (60ms per step) - Good balance
- **10x:** Maximum speed (30ms per step) - Fastest results

**Usage:**
- Drag slider left/right to adjust speed
- Higher values = faster simulation
- Lower values = slower, more detailed observation

**Tips:**
- Use 1-2x to observe learning process in detail
- Use 5-10x to see results quickly
- Adjust during simulation if needed

---

### 3. Curiosity Multiplier Slider

**Location:** Top control bar

**Range:** 0 to 25

**Curiosity Levels:**
- **0-5:** Low curiosity - Agent focuses on reaching goal quickly
- **5-15:** Medium curiosity - Balanced exploration and exploitation
- **15-25:** High curiosity - Agent prioritizes exploring new areas

**Usage:**
- Drag slider to adjust curiosity level
- Higher values = more exploration
- Lower values = more direct paths

**Behavioral Impact:**
- **Low Curiosity (0-5):**
  - Direct paths to goal
  - Faster goal reaching
  - Less map coverage
  - Fewer steps but may miss optimal paths

- **Medium Curiosity (5-15):**
  - Balanced behavior
  - Good exploration
  - Reasonable path efficiency
  - Recommended starting point

- **High Curiosity (15-25):**
  - Extensive exploration
  - Better map coverage
  - Longer paths
  - More comprehensive learning

**Tips:**
- Start with 5-10 for balanced behavior
- Increase to see more exploration
- Decrease to see faster goal reaching
- Experiment to find your preference

---

### 4. NEW WORLD Button

**Location:** Top control bar

**Function:**
- Generates a completely new maze/environment
- Resets all statistics
- Clears agent's current progress
- Starts fresh with new random maze

**Usage:**
- Click to reset and start fresh
- Use after completing a run
- Use to try different maze configurations

**Note:**
- This resets steps, rewards, and all statistics
- Training experiences are cleared
- Agent starts from beginning

---

### 5. Environment Type Selector

**Location:** Top control bar (dropdown menu)

**Options:**
- **Maze:** Classic recursive backtracking maze
- **Open Field:** Sparse obstacles in open space
- **Corridor:** Winding corridor-like paths
- **Spiral:** Circular spiral patterns
- **Grid:** Checkerboard-style grid patterns

**Function:**
- Changes the type of environment the agent navigates
- Each type presents different challenges
- Agent adapts to different structures

**Usage:**
1. Click dropdown menu
2. Select environment type
3. New maze generates automatically
4. Start simulation to see agent adapt

**Tips:**
- Try all types to see variety
- Compare agent performance across types
- Use transfer learning to share knowledge between types

---

### 6. NN ON / NN OFF Toggle

**Location:** Top control bar

**States:**
- **NN ON (Purple):** Neural network mode active
- **NN OFF (Grey):** Epsilon-greedy mode active

**Function:**
- Toggles between neural network learning and classic epsilon-greedy strategy
- Allows comparison of learning methods

**Neural Network Mode (NN ON):**
- Uses trained policy network for decisions
- Learns and improves over time
- More sophisticated decision-making
- Shows training experiences counter

**Epsilon-Greedy Mode (NN OFF):**
- Uses classic exploration-exploitation strategy
- More predictable behavior
- No learning/improvement
- Simpler decision-making

**Usage:**
- Click to toggle between modes
- Compare behavior in both modes
- Keep ON to see learning in action

**Tips:**
- Start with NN ON to see learning
- Try NN OFF to see baseline behavior
- Compare path efficiency between modes

---

### 7. SAVE KNOWLEDGE Button

**Location:** Top control bar

**Function:**
- Saves current learned knowledge (network weights) to knowledge base
- Enables transfer learning to new environments
- Stores Policy Network, VAE, and MDN-RNN weights

**Usage:**
1. Let agent learn in one environment
2. Click "SAVE KNOWLEDGE" after learning
3. Switch to new environment
4. Agent can use saved knowledge (if transfer learning is implemented)

**Tips:**
- Save after successful learning
- Use for transfer between environment types
- Helps agent start with better initial knowledge

---

## Understanding the Interface

### Left Panel - Vision & Memory

#### Top Section: Latent Visualizer

**What it shows:**
- **8-Dimensional Latent Vector:** Compressed state representation
- **VAE Reconstruction:** 5×5 grid showing how Vision "sees" the maze
- **Real-time Updates:** Changes as agent moves

**How to read:**
- Latent vector values represent compressed maze state
- Reconstruction shows visual interpretation
- Updates continuously during navigation

#### Bottom Section: Dream State

**What it shows:**
- **Dream Prediction:** Next 5 predicted steps (purple path)
- **Internal Monologue:** Text describing agent's current state
- **Confidence Score:** How confident the prediction is

**How to read:**
- Dream shows where agent thinks it will go
- Monologue describes agent's "thoughts"
- Updates every 5 steps

---

### Center Panel - The Maze

#### Maze Display

**Visual Elements:**
- **Blue Dot:** Agent's current position
- **Green Square:** Starting position
- **Orange Square:** Goal position
- **Dark Cells:** Walls (cannot pass through)
- **Light Cells:** Navigable paths
- **Trail:** Agent's path history (shows where it's been)

**How to read:**
- Watch blue dot move through the maze
- Trail shows exploration path
- Goal is the orange square

#### Statistics Display (Below Maze)

**Metrics Shown:**

1. **STEPS**
   - Total number of steps taken
   - Increases with each move
   - Lower is better (more efficient)

2. **EPSILON (ε)**
   - Current exploration probability
   - Starts at 1.0, decays to 0.15
   - Shows exploration vs exploitation balance

3. **REWARD**
   - Cumulative reward earned
   - Can be negative (step penalties)
   - Higher is better

4. **CURIOSITY**
   - Total curiosity bonus earned
   - Increases when visiting new areas
   - Shows exploration activity

#### Neural Network Info (If NN is ON)

**What it shows:**
- Number of training experiences collected
- Indicates network is learning
- Updates as agent collects experiences

---

### Right Panel - Logs & Strategy

#### Controller (C) Logs

**What it shows:**
- Real-time log of agent actions and decisions
- Timestamped entries
- Color-coded by type

**Log Types:**
- **Blue:** Action logs (movement decisions)
- **Purple:** Dream/Memory logs (predictions)
- **Green:** Vision/Sensory logs (perception)

**How to read:**
- Scroll to see recent activity
- Timestamps show when events occurred
- Messages describe what agent is doing

#### Strategy Mode Display

**What it shows:**
- Current strategy mode (EXPLORE or EXPLOIT)
- Explanation of current behavior
- Visual indicator (colored dot)

**Modes:**
- **EXPLORE (Purple):** Agent is exploring new areas
- **EXPLOIT (Blue):** Agent is using learned knowledge

**How to read:**
- Mode changes based on agent behavior
- Explanation describes current strategy
- Indicator shows mode visually

---

## Step-by-Step Usage Guide

### Basic Navigation

**First Time User:**

1. **Start the simulation:**
   - Click "START SIMULATION" button
   - Watch the agent begin navigating

2. **Observe the learning:**
   - Watch the maze to see agent's path
   - Check logs to see decision-making
   - Monitor statistics to track progress

3. **Wait for completion:**
   - Agent stops automatically at goal
   - Statistics show final results
   - Logs show completion message

4. **Try again:**
   - Click "NEW WORLD" for new maze
   - Adjust settings if desired
   - Start new simulation

---

### Advanced Usage

#### Compare Learning Methods

1. **Run with NN OFF:**
   - Toggle NN to OFF
   - Start simulation
   - Note path and steps

2. **Run with NN ON:**
   - Toggle NN to ON
   - Start simulation
   - Compare to previous run

3. **Observe differences:**
   - Path efficiency
   - Learning speed
   - Decision quality

#### Experiment with Curiosity

1. **Low Curiosity (0-5):**
   - Set slider to low value
   - Run simulation
   - Observe direct paths

2. **High Curiosity (15-25):**
   - Set slider to high value
   - Run simulation
   - Observe extensive exploration

3. **Compare results:**
   - Path length differences
   - Exploration coverage
   - Goal reaching time

#### Test Different Environments

1. **Try each type:**
   - Select each environment type
   - Run simulation
   - Observe agent adaptation

2. **Compare performance:**
   - Steps to goal
   - Path efficiency
   - Learning speed

3. **Use transfer learning:**
   - Save knowledge from one type
   - Switch to another type
   - See if knowledge transfers

#### Monitor Learning Progress

1. **Watch training experiences:**
   - Check NN info panel
   - See experiences increase
   - Indicates learning activity

2. **Observe epsilon decay:**
   - Watch epsilon decrease
   - Shows exploration → exploitation
   - Natural learning progression

3. **Track reward accumulation:**
   - Monitor reward metric
   - Should trend upward
   - Indicates improvement

---

## Tips for Best Experience

### First Time Users

- Start with default settings
- Watch a few complete runs
- Don't adjust settings initially
- Focus on understanding the interface

### Speed Settings

- **1-2x:** Best for observing learning process
- **5-10x:** Best for seeing results quickly
- Adjust based on your preference

### Curiosity Settings

- **5-10:** Recommended starting point
- **0-5:** For faster goal reaching
- **15-25:** For extensive exploration

### Neural Network

- Keep ON to see learning
- Compare with OFF mode
- Watch training experiences increase

### Multiple Runs

- Run multiple times to see improvement
- Try different settings
- Compare results
- Learn what works best

### Different Environments

- Try all environment types
- See how agent adapts
- Compare performance
- Use transfer learning

---

## Understanding the Output

### Successful Navigation

**Indicators:**
- Agent reaches orange goal square
- Simulation stops automatically
- Final statistics displayed
- Logs show "Goal state achieved"

**What to look for:**
- Steps taken (lower is better)
- Final reward (higher is better)
- Curiosity score (shows exploration)
- Path efficiency

### Learning Indicators

**Signs of Learning:**
- Training experiences increase (NN mode)
- Epsilon decreases over time
- Reward trends upward
- Paths become more efficient

**Improvement Over Time:**
- Later runs should be better
- Fewer steps to goal
- Higher rewards
- Better path choices

---

## Troubleshooting

### Agent Not Moving

- Check if simulation is running (green button)
- Check if goal is reached (stops automatically)
- Try clicking "NEW WORLD" to reset

### Slow Performance

- Reduce speed slider
- Check browser performance
- Close other browser tabs

### Confusing Behavior

- Read logs to understand decisions
- Check strategy mode display
- Adjust curiosity for different behavior

---

[← Back to README](../README.md)
