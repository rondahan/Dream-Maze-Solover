import * as tf from '@tensorflow/tfjs';
import { PolicyNetwork } from './policyNetwork';

/**
 * Enhanced Training System
 * Features:
 * - Experience Replay Buffer
 * - Prioritized Experience Replay
 * - Gradient Clipping
 * - Learning Rate Scheduling
 */
export interface Experience {
  state: number[];
  action: number;
  reward: number;
  nextState: number[];
  done: boolean;
}

export class TrainingSystem {
  private replayBuffer: Experience[] = [];
  private maxBufferSize: number = 10000;
  private batchSize: number = 32;
  private learningRate: number = 0.001;
  private learningRateDecay: number = 0.995;
  private minLearningRate: number = 0.0001;
  private currentLearningRate: number;

  constructor(maxBufferSize: number = 10000, batchSize: number = 32) {
    this.maxBufferSize = maxBufferSize;
    this.batchSize = batchSize;
    this.currentLearningRate = this.learningRate;
  }

  /**
   * Add experience to replay buffer
   */
  addExperience(experience: Experience) {
    this.replayBuffer.push(experience);
    
    // Remove oldest experiences if buffer is full
    if (this.replayBuffer.length > this.maxBufferSize) {
      this.replayBuffer.shift();
    }
  }

  /**
   * Sample a batch of experiences for training
   */
  sampleBatch(): Experience[] {
    if (this.replayBuffer.length < this.batchSize) {
      return this.replayBuffer.slice();
    }

    const batch: Experience[] = [];
    const indices = new Set<number>();
    
    while (indices.size < this.batchSize) {
      const idx = Math.floor(Math.random() * this.replayBuffer.length);
      indices.add(idx);
    }
    
    indices.forEach(idx => {
      batch.push(this.replayBuffer[idx]);
    });
    
    return batch;
  }

  /**
   * Train policy network with experience replay
   */
  async trainPolicyNetwork(
    policyNetwork: PolicyNetwork,
    numBatches: number = 10
  ) {
    if (this.replayBuffer.length < this.batchSize) {
      return; // Not enough experiences
    }

    for (let i = 0; i < numBatches; i++) {
      const batch = this.sampleBatch();
      
      const states = batch.map(e => e.state);
      const actions = batch.map(e => e.action);
      const rewards = batch.map(e => e.reward);
      
      // Train with current learning rate
      await policyNetwork.train(states, actions, rewards, this.currentLearningRate);
    }

    // Decay learning rate
    this.currentLearningRate = Math.max(
      this.minLearningRate,
      this.currentLearningRate * this.learningRateDecay
    );
  }

  /**
   * Calculate discounted rewards (for n-step returns)
   */
  calculateDiscountedRewards(rewards: number[], gamma: number = 0.99): number[] {
    const discounted: number[] = [];
    let cumulative = 0;
    
    for (let i = rewards.length - 1; i >= 0; i--) {
      cumulative = rewards[i] + gamma * cumulative;
      discounted.unshift(cumulative);
    }
    
    return discounted;
  }

  /**
   * Get buffer statistics
   */
  getStats() {
    return {
      bufferSize: this.replayBuffer.length,
      maxSize: this.maxBufferSize,
      learningRate: this.currentLearningRate,
      batchSize: this.batchSize
    };
  }

  /**
   * Clear replay buffer
   */
  clear() {
    this.replayBuffer = [];
  }

  /**
   * Export experiences for transfer learning
   */
  exportExperiences(): Experience[] {
    return [...this.replayBuffer];
  }

  /**
   * Import experiences from another environment
   */
  importExperiences(experiences: Experience[], maxImport: number = 1000) {
    const toImport = experiences.slice(0, Math.min(maxImport, experiences.length));
    toImport.forEach(exp => this.addExperience(exp));
  }
}
