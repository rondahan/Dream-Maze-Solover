import * as tf from '@tensorflow/tfjs';
import { MazeState, Position } from '../types';
import { MAZE_SIZE } from '../constants';

/**
 * Policy Network - Neural Network that decides which action to take
 * Input: State representation (agent position, goal position, visited history)
 * Output: Action probabilities for each possible move
 */
export class PolicyNetwork {
  private model: tf.Sequential | null = null;
  private optimizer: tf.Optimizer;
  private isInitialized = false;

  constructor() {
    // Use Adam optimizer for training
    this.optimizer = tf.train.adam(0.001);
  }

  /**
   * Initialize the neural network
   * Architecture: Input -> Dense(64) -> ReLU -> Dense(32) -> ReLU -> Output(4 actions)
   */
  async initialize() {
    if (this.isInitialized) return;

    this.model = tf.sequential({
      layers: [
        // Input layer: state representation (agent pos, goal pos, distance, visited info)
        tf.layers.dense({
          inputShape: [8], // 8 features: agentX, agentY, goalX, goalY, distance, visitedRatio, curiosity, epsilon
          units: 64,
          activation: 'relu',
          name: 'hidden1'
        }),
        tf.layers.dense({
          units: 32,
          activation: 'relu',
          name: 'hidden2'
        }),
        // Output layer: 4 possible actions (up, down, left, right)
        tf.layers.dense({
          units: 4,
          activation: 'softmax', // Probabilities for each action
          name: 'output'
        })
      ]
    });

    this.isInitialized = true;
    console.log('Policy Network initialized');
  }

  /**
   * Convert maze state to feature vector
   */
  private stateToFeatures(
    mazeState: MazeState,
    currentSteps: number,
    curiosityWeight: number
  ): number[] {
    const { agentPos, goalPos, history } = mazeState;
    
    // Calculate distance to goal
    const distance = Math.abs(agentPos.x - goalPos.x) + Math.abs(agentPos.y - goalPos.y);
    
    // Normalize positions to [0, 1]
    const normalizedAgentX = agentPos.x / MAZE_SIZE;
    const normalizedAgentY = agentPos.y / MAZE_SIZE;
    const normalizedGoalX = goalPos.x / MAZE_SIZE;
    const normalizedGoalY = goalPos.y / MAZE_SIZE;
    const normalizedDistance = distance / (MAZE_SIZE * 2);
    
    // Calculate visited ratio
    const visitedRatio = history.length / (MAZE_SIZE * MAZE_SIZE);
    
    // Epsilon decay
    const epsilon = Math.max(0.15, 1 - currentSteps / 100);
    
    // Normalize curiosity
    const normalizedCuriosity = curiosityWeight / 25;
    
    return [
      normalizedAgentX,
      normalizedAgentY,
      normalizedGoalX,
      normalizedGoalY,
      normalizedDistance,
      visitedRatio,
      normalizedCuriosity,
      epsilon
    ];
  }

  /**
   * Get action probabilities from the network
   */
  async predict(
    mazeState: MazeState,
    currentSteps: number,
    curiosityWeight: number
  ): Promise<number[]> {
    if (!this.model || !this.isInitialized) {
      await this.initialize();
    }

    const features = this.stateToFeatures(mazeState, currentSteps, curiosityWeight);
    const input = tf.tensor2d([features]);
    
    const prediction = this.model.predict(input) as tf.Tensor;
    const probabilities = await prediction.data();
    
    input.dispose();
    prediction.dispose();
    
    return Array.from(probabilities);
  }

  /**
   * Select action based on network probabilities
   * Uses epsilon-greedy: with probability epsilon, explore randomly
   */
  async selectAction(
    mazeState: MazeState,
    possibleMoves: Position[],
    currentSteps: number,
    curiosityWeight: number,
    epsilon: number
  ): Promise<{ pos: Position; actionIndex: number }> {
    if (possibleMoves.length === 0) {
      return { pos: mazeState.agentPos, actionIndex: -1 };
    }

    // With probability epsilon, explore randomly
    if (Math.random() < epsilon) {
      const randomIndex = Math.floor(Math.random() * possibleMoves.length);
      return { pos: possibleMoves[randomIndex], actionIndex: randomIndex };
    }

    // Otherwise, use network prediction
    const probabilities = await this.predict(mazeState, currentSteps, curiosityWeight);
    
    // Map network output (4 actions) to possible moves
    // Actions: [up, down, left, right] = [0, 1, 2, 3]
    const dirs = [
      { x: 0, y: 1 },   // up
      { x: 0, y: -1 },  // down
      { x: 1, y: 0 },   // right
      { x: -1, y: 0 }   // left
    ];

    // Find which actions are possible
    const actionScores: { action: number; score: number; move: Position }[] = [];
    
    for (let i = 0; i < dirs.length; i++) {
      const dir = dirs[i];
      const targetPos = {
        x: mazeState.agentPos.x + dir.x,
        y: mazeState.agentPos.y + dir.y
      };
      
      // Check if this move is in possibleMoves
      const isPossible = possibleMoves.some(
        p => p.x === targetPos.x && p.y === targetPos.y
      );
      
      if (isPossible) {
        actionScores.push({
          action: i,
          score: probabilities[i],
          move: targetPos
        });
      }
    }

    // Select action with highest probability
    if (actionScores.length === 0) {
      const randomIndex = Math.floor(Math.random() * possibleMoves.length);
      return { pos: possibleMoves[randomIndex], actionIndex: randomIndex };
    }

    actionScores.sort((a, b) => b.score - a.score);
    const bestAction = actionScores[0];
    
    return {
      pos: bestAction.move,
      actionIndex: bestAction.action
    };
  }

  /**
   * Train the network using REINFORCE algorithm (Policy Gradient)
   */
  async train(
    states: number[][],
    actions: number[],
    rewards: number[],
    learningRate: number = 0.001
  ) {
    if (!this.model || !this.isInitialized) return;

    // Convert to tensors
    const statesTensor = tf.tensor2d(states);
    const actionsTensor = tf.tensor1d(actions, 'int32');
    const rewardsTensor = tf.tensor1d(rewards);

    // Normalize rewards (reduce variance)
    const meanReward = tf.mean(rewardsTensor);
    const normalizedRewards = tf.sub(rewardsTensor, meanReward);

    // Compute loss using REINFORCE
    const loss = () => {
      const predictions = this.model!.predict(statesTensor) as tf.Tensor;
      const actionProbs = tf.gather(predictions, actionsTensor, 1);
      const logProbs = tf.log(tf.add(actionProbs, 1e-8)); // Add small epsilon to avoid log(0)
      const weightedLogProbs = tf.mul(logProbs, normalizedRewards);
      return tf.neg(tf.mean(weightedLogProbs)); // Negative because we want to maximize
    };

    // Train for a few steps
    for (let i = 0; i < 3; i++) {
      this.optimizer.minimize(loss);
    }

    // Cleanup
    statesTensor.dispose();
    actionsTensor.dispose();
    rewardsTensor.dispose();
    normalizedRewards.dispose();
    meanReward.dispose();
  }

  /**
   * Get network weights (for transfer learning)
   */
  getWeights(): tf.Tensor[] {
    if (!this.model) return [];
    return this.model.getWeights();
  }

  /**
   * Get network summary
   */
  getSummary(): string {
    if (!this.model) return 'Not initialized';
    return this.model.summary();
  }

  /**
   * Dispose resources
   */
  dispose() {
    if (this.model) {
      this.model.dispose();
      this.model = null;
    }
    this.isInitialized = false;
  }
}
