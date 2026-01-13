import type * as tf from '@tensorflow/tfjs';
import { loadTensorFlow } from '../utils/tfLoader';
import { MazeState, Position } from '../types';
import { MAZE_SIZE } from '../constants';

/**
 * Deep Q-Network (DQN)
 * Learns Q-values for state-action pairs
 * Uses epsilon-greedy exploration
 */
export class DQN {
  private qNetwork: tf.LayersModel | null = null;
  private targetNetwork: tf.LayersModel | null = null;
  private optimizer: tf.Optimizer | null = null;
  private isInitialized = false;
  private tf: typeof import('@tensorflow/tfjs') | null = null;
  private updateTargetCounter = 0;
  private readonly targetUpdateFrequency = 100; // Update target network every 100 steps

  constructor() {
    // Optimizer will be created after TensorFlow loads
  }

  /**
   * Initialize DQN networks
   * Architecture: Input(8) -> Dense(64) -> ReLU -> Dense(32) -> ReLU -> Output(4 Q-values)
   */
  async initialize() {
    if (this.isInitialized) return;

    // Lazy load TensorFlow.js
    this.tf = await loadTensorFlow();
    const tf = this.tf;

    // Create optimizer
    this.optimizer = tf.train.adam(0.001);

    // Q-Network (main network)
    this.qNetwork = tf.sequential({
      layers: [
        tf.layers.dense({
          inputShape: [8],
          units: 64,
          activation: 'relu',
          name: 'dense1'
        }),
        tf.layers.dense({
          units: 32,
          activation: 'relu',
          name: 'dense2'
        }),
        tf.layers.dense({
          units: 4, // Q-values for 4 actions
          activation: 'linear',
          name: 'output'
        })
      ]
    });

    // Target Network (copy of Q-network, updated less frequently)
    this.targetNetwork = tf.sequential({
      layers: [
        tf.layers.dense({
          inputShape: [8],
          units: 64,
          activation: 'relu',
          name: 'dense1'
        }),
        tf.layers.dense({
          units: 32,
          activation: 'relu',
          name: 'dense2'
        }),
        tf.layers.dense({
          units: 4,
          activation: 'linear',
          name: 'output'
        })
      ]
    });

    // Initialize target network with Q-network weights
    this.targetNetwork.setWeights(this.qNetwork.getWeights().map(w => w.clone()));

    this.isInitialized = true;
    console.log('DQN initialized');
  }

  /**
   * Convert maze state to feature vector (same as PolicyNetwork)
   */
  private stateToFeatures(
    mazeState: MazeState,
    currentSteps: number,
    curiosityWeight: number
  ): number[] {
    const { agentPos, goalPos, history } = mazeState;
    
    const distance = Math.abs(agentPos.x - goalPos.x) + Math.abs(agentPos.y - goalPos.y);
    const normalizedAgentX = agentPos.x / MAZE_SIZE;
    const normalizedAgentY = agentPos.y / MAZE_SIZE;
    const normalizedGoalX = goalPos.x / MAZE_SIZE;
    const normalizedGoalY = goalPos.y / MAZE_SIZE;
    const normalizedDistance = distance / (MAZE_SIZE * 2);
    const visitedRatio = history.length / (MAZE_SIZE * MAZE_SIZE);
    const epsilon = Math.max(0.15, 1 - currentSteps / 100);
    const normalizedCuriosity = curiosityWeight / 25;
    
    return [
      normalizedAgentX, normalizedAgentY, normalizedGoalX, normalizedGoalY,
      normalizedDistance, visitedRatio, normalizedCuriosity, epsilon
    ];
  }

  /**
   * Get Q-values for all actions
   */
  async getQValues(
    mazeState: MazeState,
    currentSteps: number,
    curiosityWeight: number
  ): Promise<number[]> {
    if (!this.qNetwork || !this.isInitialized || !this.tf) {
      await this.initialize();
    }
    const tf = this.tf!;

    const features = this.stateToFeatures(mazeState, currentSteps, curiosityWeight);
    const input = tf.tensor2d([features]);
    
    const qValues = this.qNetwork!.predict(input) as tf.Tensor;
    const values = await qValues.data();
    
    input.dispose();
    qValues.dispose();
    
    return Array.from(values);
  }

  /**
   * Select action using epsilon-greedy strategy
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

    // Epsilon-greedy: explore with probability epsilon
    if (Math.random() < epsilon) {
      const randomIndex = Math.floor(Math.random() * possibleMoves.length);
      return { pos: possibleMoves[randomIndex], actionIndex: randomIndex };
    }

    // Exploit: choose action with highest Q-value
    const qValues = await this.getQValues(mazeState, currentSteps, curiosityWeight);
    
    // Map Q-values to possible moves
    const dirs = [
      { x: 0, y: 1 },   // up
      { x: 0, y: -1 },  // down
      { x: 1, y: 0 },   // right
      { x: -1, y: 0 }   // left
    ];

    const actionScores: { action: number; qValue: number; move: Position }[] = [];
    
    for (let i = 0; i < dirs.length; i++) {
      const dir = dirs[i];
      const targetPos = {
        x: mazeState.agentPos.x + dir.x,
        y: mazeState.agentPos.y + dir.y
      };
      
      const isPossible = possibleMoves.some(
        p => p.x === targetPos.x && p.y === targetPos.y
      );
      
      if (isPossible) {
        actionScores.push({
          action: i,
          qValue: qValues[i],
          move: targetPos
        });
      }
    }

    if (actionScores.length === 0) {
      const randomIndex = Math.floor(Math.random() * possibleMoves.length);
      return { pos: possibleMoves[randomIndex], actionIndex: randomIndex };
    }

    // Select action with highest Q-value
    actionScores.sort((a, b) => b.qValue - a.qValue);
    const bestAction = actionScores[0];
    
    return {
      pos: bestAction.move,
      actionIndex: bestAction.action
    };
  }

  /**
   * Train DQN using experience replay
   */
  async train(
    experiences: Array<{
      state: number[];
      action: number;
      reward: number;
      nextState: number[];
      done: boolean;
    }>,
    gamma: number = 0.99
  ) {
    if (!this.qNetwork || !this.targetNetwork || !this.isInitialized || !this.tf || !this.optimizer) {
      return;
    }
    const tf = this.tf;

    if (experiences.length === 0) return;

    // Prepare batch
    const states = experiences.map(e => e.state);
    const actions = experiences.map(e => e.action);
    const rewards = experiences.map(e => e.reward);
    const nextStates = experiences.map(e => e.nextState);
    const dones = experiences.map(e => e.done ? 1 : 0);

    const statesTensor = tf.tensor2d(states);
    const nextStatesTensor = tf.tensor2d(nextStates);
    const rewardsTensor = tf.tensor1d(rewards);
    const donesTensor = tf.tensor1d(dones);

    // Compute target Q-values using target network
    const nextQValues = this.targetNetwork.predict(nextStatesTensor) as tf.Tensor;
    const nextQValuesData = await nextQValues.data();
    const nextQValuesArray = Array.from(nextQValuesData);

    // Compute targets: r + γ * max(Q(next_state)) if not done, else r
    const targets: number[] = [];
    for (let i = 0; i < experiences.length; i++) {
      const batchSize = experiences.length;
      const maxNextQ = Math.max(
        nextQValuesArray[i * 4],
        nextQValuesArray[i * 4 + 1],
        nextQValuesArray[i * 4 + 2],
        nextQValuesArray[i * 4 + 3]
      );
      const target = rewards[i] + (1 - dones[i]) * gamma * maxNextQ;
      targets.push(target);
    }

    // Get current Q-values
    const currentQValues = this.qNetwork.predict(statesTensor) as tf.Tensor;
    const currentQValuesData = await currentQValues.data();
    const currentQValuesArray = Array.from(currentQValuesData);

    // Create target Q-values tensor (update only the selected action)
    const targetQValues = [...currentQValuesArray];
    for (let i = 0; i < experiences.length; i++) {
      targetQValues[i * 4 + actions[i]] = targets[i];
    }

    const targetQValuesTensor = tf.tensor2d(
      Array.from({ length: experiences.length }, (_, i) => 
        targetQValues.slice(i * 4, (i + 1) * 4)
      )
    );

    // Compute loss
    const loss = () => {
      const predictions = this.qNetwork!.predict(statesTensor) as tf.Tensor;
      const lossValue = tf.losses.meanSquaredError(targetQValuesTensor, predictions);
      return lossValue.asScalar();
    };

    // Train
    this.optimizer!.minimize(loss);

    // Update target network periodically
    this.updateTargetCounter++;
    if (this.updateTargetCounter >= this.targetUpdateFrequency) {
      this.targetNetwork.setWeights(
        this.qNetwork.getWeights().map(w => w.clone())
      );
      this.updateTargetCounter = 0;
    }

    // Cleanup
    statesTensor.dispose();
    nextStatesTensor.dispose();
    rewardsTensor.dispose();
    donesTensor.dispose();
    nextQValues.dispose();
    currentQValues.dispose();
    targetQValuesTensor.dispose();
  }

  /**
   * Get network weights (for transfer learning)
   */
  getWeights(): tf.Tensor[] {
    if (!this.qNetwork) return [];
    return this.qNetwork.getWeights();
  }

  /**
   * Dispose resources
   */
  dispose() {
    if (this.qNetwork) {
      this.qNetwork.dispose();
      this.qNetwork = null;
    }
    if (this.targetNetwork) {
      this.targetNetwork.dispose();
      this.targetNetwork = null;
    }
    this.isInitialized = false;
  }
}
