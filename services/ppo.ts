import type * as tf from '@tensorflow/tfjs';
import { loadTensorFlow } from '../utils/tfLoader';
import { MazeState, Position } from '../types';
import { MAZE_SIZE } from '../constants';

/**
 * Proximal Policy Optimization (PPO)
 * Advanced policy gradient method with clipped objective
 * Uses softmax exploration (required for PPO)
 */
export class PPO {
  private policy: tf.LayersModel | null = null;
  private value: tf.LayersModel | null = null;
  private optimizer: tf.Optimizer | null = null;
  private isInitialized = false;
  private tf: typeof import('@tensorflow/tfjs') | null = null;
  private temperature: number = 1.0; // For softmax sampling
  private clipEpsilon: number = 0.2; // PPO clipping parameter

  constructor(temperature: number = 1.0, clipEpsilon: number = 0.2) {
    this.temperature = temperature;
    this.clipEpsilon = clipEpsilon;
  }

  /**
   * Set temperature for softmax
   */
  setTemperature(temperature: number) {
    this.temperature = temperature;
  }

  /**
   * Initialize PPO networks
   */
  async initialize() {
    if (this.isInitialized) return;

    // Lazy load TensorFlow.js
    this.tf = await loadTensorFlow();
    const tf = this.tf;

    // Create optimizer
    this.optimizer = tf.train.adam(0.001);

    // Policy Network: state -> action probabilities
    this.policy = tf.sequential({
      layers: [
        tf.layers.dense({
          inputShape: [8],
          units: 64,
          activation: 'relu',
          name: 'policy_dense1'
        }),
        tf.layers.dense({
          units: 32,
          activation: 'relu',
          name: 'policy_dense2'
        }),
        tf.layers.dense({
          units: 4,
          activation: 'softmax', // Probabilities
          name: 'policy_output'
        })
      ]
    });

    // Value Network: state -> value estimate
    this.value = tf.sequential({
      layers: [
        tf.layers.dense({
          inputShape: [8],
          units: 64,
          activation: 'relu',
          name: 'value_dense1'
        }),
        tf.layers.dense({
          units: 32,
          activation: 'relu',
          name: 'value_dense2'
        }),
        tf.layers.dense({
          units: 1,
          activation: 'linear',
          name: 'value_output'
        })
      ]
    });

    this.isInitialized = true;
    console.log('PPO initialized');
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
   * Get action probabilities from policy
   */
  async getActionProbabilities(
    mazeState: MazeState,
    currentSteps: number,
    curiosityWeight: number
  ): Promise<number[]> {
    if (!this.policy || !this.isInitialized || !this.tf) {
      await this.initialize();
    }
    const tf = this.tf!;

    const features = this.stateToFeatures(mazeState, currentSteps, curiosityWeight);
    const input = tf.tensor2d([features]);
    
    const probabilities = this.policy!.predict(input) as tf.Tensor;
    const probs = await probabilities.data();
    
    input.dispose();
    probabilities.dispose();
    
    return Array.from(probs);
  }

  /**
   * Sample from softmax distribution (required for PPO)
   */
  private sampleFromSoftmax(probabilities: number[], temperature: number = 1.0): number {
    // Apply temperature
    const scaled = probabilities.map(p => Math.exp(Math.log(p + 1e-8) / temperature));
    const sum = scaled.reduce((a, b) => a + b, 0);
    const normalized = scaled.map(s => s / sum);

    // Sample
    let random = Math.random();
    for (let i = 0; i < normalized.length; i++) {
      random -= normalized[i];
      if (random <= 0) {
        return i;
      }
    }
    return normalized.length - 1;
  }

  /**
   * Select action using softmax (PPO always uses softmax)
   */
  async selectAction(
    mazeState: MazeState,
    possibleMoves: Position[],
    currentSteps: number,
    curiosityWeight: number,
    epsilon: number // Ignored for PPO, but kept for interface compatibility
  ): Promise<{ pos: Position; actionIndex: number }> {
    if (possibleMoves.length === 0) {
      return { pos: mazeState.agentPos, actionIndex: -1 };
    }

    const probabilities = await this.getActionProbabilities(mazeState, currentSteps, curiosityWeight);

    // Map probabilities to possible moves
    const dirs = [
      { x: 0, y: 1 },   // up
      { x: 0, y: -1 },  // down
      { x: 1, y: 0 },   // right
      { x: -1, y: 0 }   // left
    ];

    const actionScores: { action: number; prob: number; move: Position }[] = [];
    
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
          prob: probabilities[i],
          move: targetPos
        });
      }
    }

    if (actionScores.length === 0) {
      const randomIndex = Math.floor(Math.random() * possibleMoves.length);
      return { pos: possibleMoves[randomIndex], actionIndex: randomIndex };
    }

    // PPO always uses softmax sampling
    const possibleProbs = actionScores.map(a => a.prob);
    const possibleActions = actionScores.map(a => a.action);
    const sampledIdx = this.sampleFromSoftmax(possibleProbs, this.temperature);
    const selectedAction = possibleActions[sampledIdx];

    const selected = actionScores.find(a => a.action === selectedAction)!;
    
    return {
      pos: selected.move,
      actionIndex: selectedAction
    };
  }

  /**
   * Train PPO using clipped objective
   */
  async train(
    experiences: Array<{
      state: number[];
      action: number;
      reward: number;
      nextState: number[];
      done: boolean;
      oldProb: number; // Old probability (for importance sampling)
    }>,
    gamma: number = 0.99,
    epochs: number = 4
  ) {
    if (!this.policy || !this.value || !this.isInitialized || !this.tf || !this.optimizer) {
      return;
    }
    const tf = this.tf;

    if (experiences.length === 0) return;

    // Prepare batch
    const states = experiences.map(e => e.state);
    const nextStates = experiences.map(e => e.nextState);
    const actions = experiences.map(e => e.action);
    const rewards = experiences.map(e => e.reward);
    const dones = experiences.map(e => e.done ? 1 : 0);
    const oldProbs = experiences.map(e => e.oldProb);

    const statesTensor = tf.tensor2d(states);
    const nextStatesTensor = tf.tensor2d(nextStates);
    const actionsTensor = tf.tensor1d(actions, 'int32');
    const rewardsTensor = tf.tensor1d(rewards);
    const donesTensor = tf.tensor1d(dones);
    const oldProbsTensor = tf.tensor1d(oldProbs);

    // Compute values and next values
    const values = this.value.predict(statesTensor) as tf.Tensor;
    const nextValues = this.value.predict(nextStatesTensor) as tf.Tensor;

    const valuesData = await values.data();
    const nextValuesData = await nextValues.data();
    const valuesArray = Array.from(valuesData);
    const nextValuesArray = Array.from(nextValuesData);

    // Compute returns and advantages
    const returns: number[] = [];
    let advantage = 0;
    for (let i = experiences.length - 1; i >= 0; i--) {
      if (dones[i]) {
        advantage = 0;
      }
      advantage = rewards[i] + (1 - dones[i]) * gamma * nextValuesArray[i] - valuesArray[i] + gamma * advantage;
      returns.unshift(rewards[i] + (1 - dones[i]) * gamma * (i < experiences.length - 1 ? returns[0] : nextValuesArray[i]));
    }

    const returnsTensor = tf.tensor1d(returns);
    const advantagesTensor = tf.tensor1d(returns.map((r, i) => r - valuesArray[i]));

    // Normalize advantages
    const advMean = tf.mean(advantagesTensor);
    const advStd = tf.moments(advantagesTensor).variance.sqrt();
    const normalizedAdvantages = tf.div(tf.sub(advantagesTensor, advMean), tf.add(advStd, 1e-8));

    // Train for multiple epochs (PPO characteristic)
    for (let epoch = 0; epoch < epochs; epoch++) {
      // Get current policy probabilities
      const currentProbs = this.policy.predict(statesTensor) as tf.Tensor;
      const selectedProbs = tf.gather(currentProbs, actionsTensor, 1);
      const selectedProbsData = await selectedProbs.data();
      const selectedProbsArray = Array.from(selectedProbsData);

      // Compute importance sampling ratio
      const ratio = tf.div(selectedProbs, tf.add(oldProbsTensor, 1e-8));

      // Clipped objective
      const clippedRatio = tf.clipByValue(
        ratio,
        1 - this.clipEpsilon,
        1 + this.clipEpsilon
      );

      const unclipped = tf.mul(ratio, normalizedAdvantages);
      const clipped = tf.mul(clippedRatio, normalizedAdvantages);
      const policyLoss = tf.neg(tf.mean(tf.minimum(unclipped, clipped))).asScalar();

      // Value loss
      const valueLoss = () => {
        const valuePred = this.value!.predict(statesTensor) as tf.Tensor;
        return tf.losses.meanSquaredError(returnsTensor, valuePred).asScalar();
      };

      // Train
      this.optimizer!.minimize(() => policyLoss);
      this.optimizer!.minimize(valueLoss);

      selectedProbs.dispose();
      currentProbs.dispose();
    }

    // Cleanup
    statesTensor.dispose();
    nextStatesTensor.dispose();
    actionsTensor.dispose();
    rewardsTensor.dispose();
    donesTensor.dispose();
    oldProbsTensor.dispose();
    values.dispose();
    nextValues.dispose();
    returnsTensor.dispose();
    advantagesTensor.dispose();
    normalizedAdvantages.dispose();
    advMean.dispose();
    advStd.dispose();
  }

  /**
   * Get network weights (for transfer learning)
   */
  getWeights(): { policy: tf.Tensor[]; value: tf.Tensor[] } {
    return {
      policy: this.policy ? this.policy.getWeights() : [],
      value: this.value ? this.value.getWeights() : []
    };
  }

  /**
   * Dispose resources
   */
  dispose() {
    if (this.policy) {
      this.policy.dispose();
      this.policy = null;
    }
    if (this.value) {
      this.value.dispose();
      this.value = null;
    }
    this.isInitialized = false;
  }
}
