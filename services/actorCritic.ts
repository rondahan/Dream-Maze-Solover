import type * as tf from '@tensorflow/tfjs';
import { loadTensorFlow } from '../utils/tfLoader';
import { MazeState, Position } from '../types';
import { MAZE_SIZE } from '../constants';

export type ExplorationType = 'epsilon-greedy' | 'softmax';

/**
 * Actor-Critic Network
 * Actor: Policy network (selects actions)
 * Critic: Value network (estimates state values)
 * Supports both epsilon-greedy and softmax exploration
 */
export class ActorCritic {
  private actor: tf.LayersModel | null = null;
  private critic: tf.LayersModel | null = null;
  private optimizer: tf.Optimizer | null = null;
  private isInitialized = false;
  private tf: typeof import('@tensorflow/tfjs') | null = null;
  private explorationType: ExplorationType = 'softmax';
  private temperature: number = 1.0; // For softmax

  constructor(explorationType: ExplorationType = 'softmax', temperature: number = 1.0) {
    this.explorationType = explorationType;
    this.temperature = temperature;
  }

  /**
   * Set exploration type
   */
  setExplorationType(type: ExplorationType, temperature: number = 1.0) {
    this.explorationType = type;
    this.temperature = temperature;
  }

  /**
   * Initialize Actor-Critic networks
   */
  async initialize() {
    if (this.isInitialized) return;

    // Lazy load TensorFlow.js
    this.tf = await loadTensorFlow();
    const tf = this.tf;

    // Create optimizer
    this.optimizer = tf.train.adam(0.001);

    // Actor Network: state -> action probabilities
    this.actor = tf.sequential({
      layers: [
        tf.layers.dense({
          inputShape: [8],
          units: 64,
          activation: 'relu',
          name: 'actor_dense1'
        }),
        tf.layers.dense({
          units: 32,
          activation: 'relu',
          name: 'actor_dense2'
        }),
        tf.layers.dense({
          units: 4,
          activation: 'softmax', // Probabilities
          name: 'actor_output'
        })
      ]
    });

    // Critic Network: state -> value estimate
    this.critic = tf.sequential({
      layers: [
        tf.layers.dense({
          inputShape: [8],
          units: 64,
          activation: 'relu',
          name: 'critic_dense1'
        }),
        tf.layers.dense({
          units: 32,
          activation: 'relu',
          name: 'critic_dense2'
        }),
        tf.layers.dense({
          units: 1, // Single value estimate
          activation: 'linear',
          name: 'critic_output'
        })
      ]
    });

    this.isInitialized = true;
    console.log('Actor-Critic initialized');
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
   * Get action probabilities from actor
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
   * Get state value from critic
   */
  async getStateValue(
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
    
    return Array.from(val)[0];
  }

  /**
   * Sample from softmax distribution
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
   * Select action using exploration strategy
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

    let selectedAction: number;

    if (this.explorationType === 'epsilon-greedy') {
      // Epsilon-greedy: explore randomly with probability epsilon
      if (Math.random() < epsilon) {
        const randomIndex = Math.floor(Math.random() * actionScores.length);
        selectedAction = actionScores[randomIndex].action;
      } else {
        // Exploit: choose highest probability
        actionScores.sort((a, b) => b.prob - a.prob);
        selectedAction = actionScores[0].action;
      }
    } else {
      // Softmax: sample from distribution
      const possibleProbs = actionScores.map(a => a.prob);
      const possibleActions = actionScores.map(a => a.action);
      const sampledIdx = this.sampleFromSoftmax(possibleProbs, this.temperature);
      selectedAction = possibleActions[sampledIdx];
    }

    const selected = actionScores.find(a => a.action === selectedAction)!;
    
    return {
      pos: selected.move,
      actionIndex: selectedAction
    };
  }

  /**
   * Train Actor-Critic using advantage
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
    if (!this.actor || !this.critic || !this.isInitialized || !this.tf || !this.optimizer) {
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

    const statesTensor = tf.tensor2d(states);
    const nextStatesTensor = tf.tensor2d(nextStates);
    const actionsTensor = tf.tensor1d(actions, 'int32');
    const rewardsTensor = tf.tensor1d(rewards);
    const donesTensor = tf.tensor1d(dones);

    // Compute values
    const values = this.critic.predict(statesTensor) as tf.Tensor;
    const nextValues = this.critic.predict(nextStatesTensor) as tf.Tensor;

    // Compute targets: r + γ * V(next_state) if not done, else r
    const valuesData = await values.data();
    const nextValuesData = await nextValues.data();
    const valuesArray = Array.from(valuesData);
    const nextValuesArray = Array.from(nextValuesData);

    const targets: number[] = [];
    for (let i = 0; i < experiences.length; i++) {
      const target = rewards[i] + (1 - dones[i]) * gamma * nextValuesArray[i];
      targets.push(target);
    }

    const targetsTensor = tf.tensor1d(targets);

    // Compute advantages: target - value
    const advantages = tf.sub(targetsTensor, values);

    // Critic loss: MSE between targets and values
    const criticLoss = () => {
      const predictions = this.critic!.predict(statesTensor) as tf.Tensor;
      return tf.losses.meanSquaredError(targetsTensor, predictions).asScalar();
    };

    // Actor loss: policy gradient with advantage
    const actorLoss = () => {
      const actionProbs = this.actor!.predict(statesTensor) as tf.Tensor;
      const selectedProbs = tf.gather(actionProbs, actionsTensor, 1);
      const logProbs = tf.log(tf.add(selectedProbs, 1e-8));
      const weightedLogProbs = tf.mul(logProbs, advantages);
      return tf.neg(tf.mean(weightedLogProbs)).asScalar();
    };

    // Train both networks
    this.optimizer.minimize(criticLoss);
    this.optimizer.minimize(actorLoss);

    // Cleanup
    statesTensor.dispose();
    nextStatesTensor.dispose();
    actionsTensor.dispose();
    rewardsTensor.dispose();
    donesTensor.dispose();
    values.dispose();
    nextValues.dispose();
    targetsTensor.dispose();
    advantages.dispose();
  }

  /**
   * Get network weights (for transfer learning)
   */
  getWeights(): { actor: tf.Tensor[]; critic: tf.Tensor[] } {
    return {
      actor: this.actor ? this.actor.getWeights() : [],
      critic: this.critic ? this.critic.getWeights() : []
    };
  }

  /**
   * Dispose resources
   */
  dispose() {
    if (this.actor) {
      this.actor.dispose();
      this.actor = null;
    }
    if (this.critic) {
      this.critic.dispose();
      this.critic = null;
    }
    this.isInitialized = false;
  }
}
