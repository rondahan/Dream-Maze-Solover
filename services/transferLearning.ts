import type * as tf from '@tensorflow/tfjs';
import { loadTensorFlow } from '../utils/tfLoader';
import { PolicyNetwork } from './policyNetwork';
import { VAE } from './vae';
import { MDNRNN } from './mdnRnn';
import { TrainingSystem, Experience } from './trainingSystem';

/**
 * Transfer Learning System
 * Allows sharing learned knowledge between different environments
 */
export class TransferLearning {
  /**
   * Transfer policy network weights from source to target
   * Option 1: Full transfer (copy all weights)
   * Option 2: Partial transfer (copy only some layers)
   * Option 3: Fine-tuning (copy and then retrain)
   */
  static async transferPolicyWeights(
    sourceNetwork: PolicyNetwork,
    targetNetwork: PolicyNetwork,
    transferMode: 'full' | 'partial' | 'fine_tune' = 'partial'
  ) {
    // Get source model weights
    const sourceWeights = sourceNetwork.getWeights();
    
    if (!sourceWeights || sourceWeights.length === 0) {
      console.warn('Source network has no weights to transfer');
      return;
    }

    // Get target model
    const targetModel = (targetNetwork as any).model;
    if (!targetModel) {
      await targetNetwork.initialize();
    }

    if (transferMode === 'full') {
      // Copy all weights
      targetModel.setWeights(sourceWeights);
    } else if (transferMode === 'partial') {
      // Copy only hidden layers, keep output layer random
      const targetWeights = targetModel.getWeights();
      const layersToTransfer = Math.min(sourceWeights.length - 1, targetWeights.length - 1);
      
      for (let i = 0; i < layersToTransfer; i++) {
        if (sourceWeights[i].shape.join(',') === targetWeights[i].shape.join(',')) {
          targetWeights[i] = sourceWeights[i].clone();
        }
      }
      
      targetModel.setWeights(targetWeights);
    } else if (transferMode === 'fine_tune') {
      // Copy weights and reduce learning rate for fine-tuning
      targetModel.setWeights(sourceWeights);
      // Fine-tuning would happen during subsequent training with lower learning rate
    }
  }

  /**
   * Transfer VAE encoder/decoder weights
   */
  static async transferVAEWeights(
    sourceVAE: VAE,
    targetVAE: VAE,
    transferEncoder: boolean = true,
    transferDecoder: boolean = true
  ) {
    await sourceVAE.initialize();
    await targetVAE.initialize();

    const sourceEncoder = (sourceVAE as any).encoder;
    const sourceDecoder = (sourceVAE as any).decoder;
    const targetEncoder = (targetVAE as any).encoder;
    const targetDecoder = (targetVAE as any).decoder;

    if (transferEncoder && sourceEncoder && targetEncoder) {
      try {
        const encoderWeights = sourceEncoder.getWeights();
        targetEncoder.setWeights(encoderWeights);
      } catch (error) {
        console.warn('Could not transfer encoder weights:', error);
      }
    }

    if (transferDecoder && sourceDecoder && targetDecoder) {
      try {
        const decoderWeights = sourceDecoder.getWeights();
        targetDecoder.setWeights(decoderWeights);
      } catch (error) {
        console.warn('Could not transfer decoder weights:', error);
      }
    }
  }

  /**
   * Transfer MDN-RNN weights
   */
  static async transferRNNWeights(
    sourceRNN: MDNRNN,
    targetRNN: MDNRNN
  ) {
    await sourceRNN.initialize();
    await targetRNN.initialize();

    const sourceModel = (sourceRNN as any).model;
    const targetModel = (targetRNN as any).model;

    if (sourceModel && targetModel) {
      try {
        const rnnWeights = sourceModel.getWeights();
        targetModel.setWeights(rnnWeights);
      } catch (error) {
        console.warn('Could not transfer RNN weights:', error);
      }
    }
  }

  /**
   * Transfer experiences from one training system to another
   */
  static transferExperiences(
    sourceTraining: TrainingSystem,
    targetTraining: TrainingSystem,
    maxExperiences: number = 1000
  ) {
    const experiences = sourceTraining.exportExperiences();
    targetTraining.importExperiences(experiences, maxExperiences);
  }

  /**
   * Create a shared knowledge base that can be used across environments
   */
  static createKnowledgeBase() {
    return {
      policyWeights: null as tf.Tensor[][] | null,
      vaeEncoderWeights: null as tf.Tensor[][] | null,
      vaeDecoderWeights: null as tf.Tensor[][] | null,
      rnnWeights: null as tf.Tensor[][] | null,
      experiences: [] as Experience[],
      
      async savePolicy(network: PolicyNetwork) {
        const weights = network.getWeights();
        this.policyWeights = weights.map(w => w.clone());
      },
      
      async saveVAE(vae: VAE) {
        await vae.initialize();
        const encoder = (vae as any).encoder;
        const decoder = (vae as any).decoder;
        if (encoder) this.vaeEncoderWeights = encoder.getWeights().map(w => w.clone());
        if (decoder) this.vaeDecoderWeights = decoder.getWeights().map(w => w.clone());
      },
      
      async saveRNN(rnn: MDNRNN) {
        await rnn.initialize();
        const model = (rnn as any).model;
        if (model) this.rnnWeights = model.getWeights().map(w => w.clone());
      },
      
      async loadPolicy(network: PolicyNetwork) {
        if (!this.policyWeights) return;
        await network.initialize();
        const model = (network as any).model;
        if (model) model.setWeights(this.policyWeights.map(w => w.clone()));
      },
      
      async loadVAE(vae: VAE) {
        await vae.initialize();
        const encoder = (vae as any).encoder;
        const decoder = (vae as any).decoder;
        if (this.vaeEncoderWeights && encoder) encoder.setWeights(this.vaeEncoderWeights.map(w => w.clone()));
        if (this.vaeDecoderWeights && decoder) decoder.setWeights(this.vaeDecoderWeights.map(w => w.clone()));
      },
      
      async loadRNN(rnn: MDNRNN) {
        await rnn.initialize();
        const model = (rnn as any).model;
        if (this.rnnWeights && model) model.setWeights(this.rnnWeights.map(w => w.clone()));
      },
      
      dispose() {
        this.policyWeights?.forEach(w => w.dispose());
        this.vaeEncoderWeights?.forEach(w => w.dispose());
        this.vaeDecoderWeights?.forEach(w => w.dispose());
        this.rnnWeights?.forEach(w => w.dispose());
        this.policyWeights = null;
        this.vaeEncoderWeights = null;
        this.vaeDecoderWeights = null;
        this.rnnWeights = null;
        this.experiences = [];
      }
    };
  }
}
