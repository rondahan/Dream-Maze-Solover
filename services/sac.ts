import type * as tf from '@tensorflow/tfjs';
import { loadTensorFlow } from '../utils/tfLoader';
import { MazeState, Position } from '../types';
import { MAZE_SIZE } from '../constants';

/**
 * SAC (Soft Actor-Critic)
 * Modern off-policy algorithm with entropy regularization
 * Uses maximum entropy RL for better exploration
 */
export class SAC {
  private actor: tf.LayersModel | null = null;
  private critic1: tf.LayersModel | null = null;
  private critic2: tf.LayersModel | null = null;
  private targetCritic1: tf.LayersModel | null = null;
  private targetCritic2: tf.LayersModel | null = null;
  private optimizer: tf.Optimizer | null = null;
  private isInitialized = false;
  private tf: typeof import('@tensorflow/tfjs') | null = null;
  private alpha: number = 0.2; // Temperature parameter (entropy coefficient)
  private tau: number = 0.005; // Soft update coefficient
  private gamma: number = 0.99; // Discount factor

  constructor(alpha: number = 0.2, tau: number = 0.005) {
    this.alpha = alpha;
    this.tau = tau;
  }

  /**
   * Initialize SAC networks
   */
  async initialize() {
    if (this.isInitialized) return;

    this.tf = await loadTensorFlow();
    const tf = this.tf;

    this.optimizer = tf.train.adam(0.0003);

    // Actor Network: state -> action probabilities
    this.actor = tf.sequential({
      layers: [
        tf.layers.dense({
          inputShape: [8],
          units: 256,
          activation: 'relu',
          name: 'actor_dense1'
        }),
        tf.layers.dense({
          units: 256,
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

    // Twin Q-networks (for better value estimation)
    const createCritic = () => tf.sequential({
      layers: [
        tf.layers.dense({
          inputShape: [8],
          units: 256,
          activation: 'relu',
          name: 'critic_dense1'
        }),
        tf.layers.dense({
          units: 256,
          activation: 'relu',
          name: 'critic_dense2'
        }),
        tf.layers.dense({
          units: 4, // Q-values for 4 actions
          activation: 'linear',
          name: 'critic_output'
        })
      ]
    });

    this.critic1 = createCritic();
    this.critic2 = createCritic();
    this.targetCritic1 = createCritic();
    this.targetCritic2 = createCritic();

    // Initialize target networks
    this.targetCritic1.setWeights(this.critic1.getWeights().map(w => w.clone()));
    this.targetCritic2.setWeights(this.critic2.getWeights().map(w => w.clone()));

    this.isInitialized = true;
    console.log('SAC initialized');
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
   * Select action using softmax sampling with entropy
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

    // Softmax sampling with temperature
    const scores = actionScores.map(a => a.score);
    const maxScore = Math.max(...scores);
    const expScores = scores.map(s => Math.exp((s - maxScore) / this.alpha));
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
   * Train SAC
   */
  async train(
    states: number[][],
    actions: number[],
    rewards: number[],
    nextStates: number[],
    dones: boolean[]
  ) {
    if (!this.actor || !this.critic1 || !this.critic2 || !this.targetCritic1 || !this.targetCritic2 || !this.optimizer || !this.tf) {
      return;
    }
    const tf = this.tf;

    if (states.length === 0) return;

    const batchSize = states.length;
    const statesTensor = tf.tensor2d(states);
    const nextStatesTensor = tf.tensor2d(nextStates);
    const actionsTensor = tf.oneHot(tf.tensor1d(actions, 'int32'), 4);
    const rewardsTensor = tf.tensor1d(rewards);
    const donesTensor = tf.tensor1d(dones.map(d => d ? 1 : 0));

    // Get next action probabilities from actor
    const nextActionProbs = this.actor.apply(nextStatesTensor) as tf.Tensor;
    
    // Compute target Q-values (min of twin critics)
    const targetQ1 = this.targetCritic1.apply(nextStatesTensor) as tf.Tensor;
    const targetQ2 = this.targetCritic2.apply(nextStatesTensor) as tf.Tensor;
    const targetQ = tf.minimum(targetQ1, targetQ2);
    
    // Compute entropy bonus
    const logProbs = tf.log(tf.add(nextActionProbs, 1e-8));
    const entropy = tf.neg(tf.sum(tf.mul(nextActionProbs, logProbs), 1, true));
    
    // Compute target: r + gamma * (min(Q1, Q2) + alpha * entropy) * (1 - done)
    const target = tf.add(
      rewardsTensor,
      tf.mul(
        tf.mul(this.gamma, tf.add(tf.sum(tf.mul(targetQ, nextActionProbs), 1), tf.mul(this.alpha, entropy))),
        tf.sub(1, donesTensor)
      )
    );

    // Critic losses
    const critic1Loss = () => {
      const q1 = this.critic1!.apply(statesTensor) as tf.Tensor;
      const q1Selected = tf.sum(tf.mul(q1, actionsTensor), 1);
      return tf.losses.meanSquaredError(target, q1Selected);
    };

    const critic2Loss = () => {
      const q2 = this.critic2!.apply(statesTensor) as tf.Tensor;
      const q2Selected = tf.sum(tf.mul(q2, actionsTensor), 1);
      return tf.losses.meanSquaredError(target, q2Selected);
    };

    // Actor loss: maximize Q + entropy
    const actorLoss = () => {
      const actionProbs = this.actor!.apply(statesTensor) as tf.Tensor;
      const q1 = this.critic1!.apply(statesTensor) as tf.Tensor;
      const qValues = tf.sum(tf.mul(q1, actionProbs), 1);
      
      const logProbs = tf.log(tf.add(actionProbs, 1e-8));
      const entropy = tf.neg(tf.sum(tf.mul(actionProbs, logProbs), 1));
      
      return tf.neg(tf.mean(tf.add(qValues, tf.mul(this.alpha, entropy))));
    };

    // Train networks
    await this.optimizer!.minimize(critic1Loss);
    await this.optimizer!.minimize(critic2Loss);
    await this.optimizer!.minimize(actorLoss);

    // Soft update target networks
    this.softUpdate(this.targetCritic1, this.critic1);
    this.softUpdate(this.targetCritic2, this.critic2);

    statesTensor.dispose();
    nextStatesTensor.dispose();
    actionsTensor.dispose();
    rewardsTensor.dispose();
    donesTensor.dispose();
    nextActionProbs.dispose();
    targetQ1.dispose();
    targetQ2.dispose();
    targetQ.dispose();
  }

  /**
   * Soft update target network
   */
  private softUpdate(target: tf.LayersModel, source: tf.LayersModel) {
    const targetWeights = target.getWeights();
    const sourceWeights = source.getWeights();
    
    const newWeights = targetWeights.map((targetW, i) => {
      const sourceW = sourceWeights[i];
      return targetW.mul(1 - this.tau).add(sourceW.mul(this.tau));
    });
    
    target.setWeights(newWeights);
  }

  /**
   * Get model summary
   */
  getSummary(): string {
    return `SAC (α=${this.alpha}, τ=${this.tau})`;
  }
}
