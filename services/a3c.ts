import type * as tf from '@tensorflow/tfjs';
import { loadTensorFlow } from '../utils/tfLoader';
import { MazeState, Position } from '../types';
import { MAZE_SIZE } from '../constants';

/**
 * A3C (Asynchronous Advantage Actor-Critic)
 * Advanced variant of Actor-Critic with advantage estimation
 * Uses n-step returns for better value estimation
 */
export class A3C {
  private actor: tf.LayersModel | null = null;
  private critic: tf.LayersModel | null = null;
  private optimizer: tf.Optimizer | null = null;
  private isInitialized = false;
  private tf: typeof import('@tensorflow/tfjs') | null = null;
  private nSteps: number = 5; // n-step returns
  private entropyCoeff: number = 0.01; // Entropy bonus for exploration

  constructor(nSteps: number = 5, entropyCoeff: number = 0.01) {
    this.nSteps = nSteps;
    this.entropyCoeff = entropyCoeff;
  }

  /**
   * Initialize A3C networks
   */
  async initialize() {
    if (this.isInitialized) return;

    this.tf = await loadTensorFlow();
    const tf = this.tf;

    this.optimizer = tf.train.adam(0.001);

    // Actor Network: state -> action probabilities
    this.actor = tf.sequential({
      layers: [
        tf.layers.dense({
          inputShape: [8],
          units: 128,
          activation: 'relu',
          name: 'actor_dense1'
        }),
        tf.layers.dense({
          units: 64,
          activation: 'relu',
          name: 'actor_dense2'
        }),
        tf.layers.dense({
          units: 4,
          activation: 'softmax',
          name: 'actor_output'
        })
      ]
    });

    // Critic Network: state -> value estimate
    this.critic = tf.sequential({
      layers: [
        tf.layers.dense({
          inputShape: [8],
          units: 128,
          activation: 'relu',
          name: 'critic_dense1'
        }),
        tf.layers.dense({
          units: 64,
          activation: 'relu',
          name: 'critic_dense2'
        }),
        tf.layers.dense({
          units: 1,
          activation: 'linear',
          name: 'critic_output'
        })
      ]
    });

    this.isInitialized = true;
    console.log('A3C initialized');
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
   * Get action probabilities
   */
  async getActionProbabilities(
    mazeState: MazeState,
    currentSteps: number,
    curiosityWeight: number
  ): Promise<number[]> {
    if (!this.actor || !this.isInitialized || !this.tf) {
      await this.initialize();
    }
    const tf = this.tf!;

    const features = this.stateToFeatures(mazeState, currentSteps, curiosityWeight);
    const input = tf.tensor2d([features]);
    
    const probabilities = this.actor!.predict(input) as tf.Tensor;
    const probs = await probabilities.data();
    
    input.dispose();
    probabilities.dispose();
    
    return Array.from(probs);
  }

  /**
   * Select action using softmax sampling
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

    // Epsilon-greedy exploration
    if (Math.random() < epsilon) {
      const randomIndex = Math.floor(Math.random() * possibleMoves.length);
      return { pos: possibleMoves[randomIndex], actionIndex: randomIndex };
    }

    const probabilities = await this.getActionProbabilities(mazeState, currentSteps, curiosityWeight);
    
    const dirs = [
      { x: 0, y: 1 },   // up
      { x: 0, y: -1 },  // down
      { x: 1, y: 0 },   // right
      { x: -1, y: 0 }   // left
    ];

    const actionScores: { action: number; score: number; move: Position }[] = [];
    
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
          score: probabilities[i],
          move: targetPos
        });
      }
    }

    if (actionScores.length === 0) {
      const randomIndex = Math.floor(Math.random() * possibleMoves.length);
      return { pos: possibleMoves[randomIndex], actionIndex: randomIndex };
    }

    // Softmax sampling
    const scores = actionScores.map(a => a.score);
    const maxScore = Math.max(...scores);
    const expScores = scores.map(s => Math.exp(s - maxScore));
    const sumExp = expScores.reduce((a, b) => a + b, 0);
    const probs = expScores.map(e => e / sumExp);

    let rand = Math.random();
    let cumulative = 0;
    for (let i = 0; i < probs.length; i++) {
      cumulative += probs[i];
      if (rand <= cumulative) {
        return { pos: actionScores[i].move, actionIndex: actionScores[i].action };
      }
    }

    return { pos: actionScores[0].move, actionIndex: actionScores[0].action };
  }

  /**
   * Estimate state value
   */
  async estimateValue(
    mazeState: MazeState,
    currentSteps: number,
    curiosityWeight: number
  ): Promise<number> {
    if (!this.critic || !this.isInitialized || !this.tf) {
      await this.initialize();
    }
    const tf = this.tf!;

    const features = this.stateToFeatures(mazeState, currentSteps, curiosityWeight);
    const input = tf.tensor2d([features]);
    
    const value = this.critic!.predict(input) as tf.Tensor;
    const val = await value.data();
    
    input.dispose();
    value.dispose();
    
    return val[0];
  }

  /**
   * Train A3C with n-step returns
   */
  async train(
    states: number[][],
    actions: number[],
    rewards: number[],
    nextStates: number[],
    dones: boolean[]
  ) {
    if (!this.actor || !this.critic || !this.optimizer || !this.tf) {
      return;
    }
    const tf = this.tf;

    if (states.length === 0) return;

    const batchSize = states.length;
    const statesTensor = tf.tensor2d(states);
    const nextStatesTensor = tf.tensor2d(nextStates);

    // Compute n-step returns
    const returns: number[] = [];
    const gamma = 0.99;
    
    for (let i = 0; i < batchSize; i++) {
      let nStepReturn = 0;
      for (let j = 0; j < Math.min(this.nSteps, batchSize - i); j++) {
        if (dones[i + j]) {
          nStepReturn += Math.pow(gamma, j) * rewards[i + j];
          break;
        }
        nStepReturn += Math.pow(gamma, j) * rewards[i + j];
      }
      if (!dones[i] && i + this.nSteps < batchSize) {
        const nextValue = await this.estimateValueFromFeatures(nextStates[Math.min(i + this.nSteps - 1, nextStates.length - 1)]);
        nStepReturn += Math.pow(gamma, this.nSteps) * nextValue;
      }
      returns.push(nStepReturn);
    }

    const returnsTensor = tf.tensor1d(returns);

    // Critic loss: MSE between predicted values and n-step returns
    const criticLoss = () => {
      const values = this.critic!.apply(statesTensor) as tf.Tensor;
      const valuesSqueezed = tf.squeeze(values);
      return tf.losses.meanSquaredError(returnsTensor, valuesSqueezed);
    };

    // Actor loss: policy gradient with advantage
    const actorLoss = () => {
      const actionProbs = this.actor!.apply(statesTensor) as tf.Tensor;
      const values = this.critic!.apply(statesTensor) as tf.Tensor;
      const valuesSqueezed = tf.squeeze(values);
      const advantages = tf.sub(returnsTensor, valuesSqueezed);
      
      // One-hot encode actions
      const actionsTensor = tf.oneHot(tf.tensor1d(actions, 'int32'), 4);
      
      // Policy gradient
      const selectedProbs = tf.sum(tf.mul(actionProbs, actionsTensor), 1);
      const logProbs = tf.log(tf.add(selectedProbs, 1e-8));
      const policyLoss = tf.neg(tf.mean(tf.mul(logProbs, advantages)));
      
      // Entropy bonus
      const entropy = tf.neg(tf.mean(tf.sum(tf.mul(actionProbs, tf.log(tf.add(actionProbs, 1e-8))), 1)));
      
      return tf.add(policyLoss, tf.mul(this.entropyCoeff, entropy));
    };

    // Train both networks
    await this.optimizer!.minimize(criticLoss);
    await this.optimizer!.minimize(actorLoss);

    statesTensor.dispose();
    nextStatesTensor.dispose();
    returnsTensor.dispose();
  }

  /**
   * Estimate value from features
   */
  private async estimateValueFromFeatures(features: number[]): Promise<number> {
    if (!this.critic || !this.tf) return 0;
    const tf = this.tf;

    const input = tf.tensor2d([features]);
    const value = this.critic.predict(input) as tf.Tensor;
    const val = await value.data();
    
    input.dispose();
    value.dispose();
    
    return val[0];
  }

  /**
   * Get model summary
   */
  getSummary(): string {
    return `A3C (n-steps: ${this.nSteps}, entropy: ${this.entropyCoeff})`;
  }
}
