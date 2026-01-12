import * as tf from '@tensorflow/tfjs';
import { MazeState, LatentVector } from '../types';
import { MAZE_SIZE, LATENT_DIM } from '../constants';

/**
 * Variational Autoencoder (VAE) for Vision Component
 * Compresses maze state into latent representation
 * Architecture: Encoder (CNN) -> Latent Space -> Decoder (CNN)
 */
export class VAE {
  private encoder: tf.LayersModel | null = null;
  private decoder: tf.LayersModel | null = null;
  private isInitialized = false;

  constructor() {}

  /**
   * Initialize VAE models
   * Encoder: Maze grid -> Latent vector (z_mean, z_log_var)
   * Decoder: Latent vector -> Reconstructed grid
   */
  async initialize() {
    if (this.isInitialized) return;

    // Encoder: Converts maze grid to latent space
    const encoderInput = tf.input({ shape: [MAZE_SIZE, MAZE_SIZE, 1] });
    
    // CNN layers for encoding - Original size for better encoding quality
    let x = tf.layers.conv2d({
      filters: 16,
      kernelSize: 3,
      strides: 2,
      padding: 'same',
      activation: 'relu'
    }).apply(encoderInput) as tf.SymbolicTensor;
    
    x = tf.layers.conv2d({
      filters: 32,
      kernelSize: 3,
      strides: 2,
      padding: 'same',
      activation: 'relu'
    }).apply(x) as tf.SymbolicTensor;
    
    x = tf.layers.conv2d({
      filters: 64,
      kernelSize: 3,
      strides: 2,
      padding: 'same',
      activation: 'relu'
    }).apply(x) as tf.SymbolicTensor;
    
    x = tf.layers.flatten().apply(x) as tf.SymbolicTensor;
    x = tf.layers.dense({ units: 128, activation: 'relu' }).apply(x) as tf.SymbolicTensor;
    
    // Latent space: mean and log variance
    const zMean = tf.layers.dense({ units: LATENT_DIM, name: 'z_mean' }).apply(x) as tf.SymbolicTensor;
    const zLogVar = tf.layers.dense({ units: LATENT_DIM, name: 'z_log_var' }).apply(x) as tf.SymbolicTensor;
    
    // Sampling layer - use a custom layer approach
    // Note: Lambda layers with random operations can be tricky, so we'll handle sampling in encode()
    const z = zMean; // We'll do actual sampling in the encode method
    
    this.encoder = tf.model({ inputs: encoderInput, outputs: [zMean, zLogVar] });

    // Decoder: Reconstructs grid from latent vector
    const decoderInput = tf.input({ shape: [LATENT_DIM] });
    let d = tf.layers.dense({ units: 128, activation: 'relu' }).apply(decoderInput) as tf.SymbolicTensor;
    d = tf.layers.dense({ units: 4 * 4 * 64, activation: 'relu' }).apply(d) as tf.SymbolicTensor;
    d = tf.layers.reshape({ targetShape: [4, 4, 64] }).apply(d) as tf.SymbolicTensor;
    
    // Transposed convolutions for upsampling
    d = tf.layers.conv2dTranspose({
      filters: 32,
      kernelSize: 3,
      strides: 2,
      padding: 'same',
      activation: 'relu'
    }).apply(d) as tf.SymbolicTensor;
    
    d = tf.layers.conv2dTranspose({
      filters: 16,
      kernelSize: 3,
      strides: 2,
      padding: 'same',
      activation: 'relu'
    }).apply(d) as tf.SymbolicTensor;
    
    d = tf.layers.conv2dTranspose({
      filters: 1,
      kernelSize: 3,
      strides: 2,
      padding: 'same',
      activation: 'sigmoid'
    }).apply(d) as tf.SymbolicTensor;
    
    // Use slice instead of cropping2d (which may not be available)
    // The output should be close to MAZE_SIZE, we'll handle any size mismatch in decode()
    const decoderOutput = d;
    
    this.decoder = tf.model({ inputs: decoderInput, outputs: decoderOutput });

    this.isInitialized = true;
    console.log('VAE initialized');
  }

  /**
   * Convert maze grid to tensor
   */
  private mazeToTensor(mazeState: MazeState): tf.Tensor {
    const { grid } = mazeState;
    const data: number[] = [];
    
    for (let y = 0; y < MAZE_SIZE; y++) {
      for (let x = 0; x < MAZE_SIZE; x++) {
        // Encode: WALL = 0, PATH/START/GOAL = 1
        const value = grid[y][x] === 'WALL' ? 0 : 1;
        data.push(value);
      }
    }
    
    return tf.tensor4d(data, [1, MAZE_SIZE, MAZE_SIZE, 1]);
  }

  /**
   * Encode maze state to latent vector
   */
  async encode(mazeState: MazeState): Promise<LatentVector> {
    if (!this.encoder || !this.isInitialized) {
      await this.initialize();
    }

    const mazeTensor = this.mazeToTensor(mazeState);
    const [zMean, zLogVar] = this.encoder.predict(mazeTensor) as tf.Tensor[];
    
    // Reparameterization trick: z = mean + std * epsilon
    const meanData = await zMean.data();
    const logVarData = await zLogVar.data();
    const meanArray = Array.from(meanData);
    const logVarArray = Array.from(logVarData);
    
    // Sample from latent distribution
    const latentVector = meanArray.map((mean, i) => {
      const std = Math.exp(logVarArray[i] * 0.5);
      const epsilon = (Math.random() * 2 - 1) * 0.1; // Small random noise
      return mean + std * epsilon;
    });
    
    // Get reconstruction for visualization
    const reconstruction = await this.decode(latentVector);
    
    // Cleanup
    mazeTensor.dispose();
    zMean.dispose();
    zLogVar.dispose();
    
    return {
      vector: latentVector,
      reconstruction: reconstruction
    };
  }

  /**
   * Decode latent vector to reconstructed grid
   */
  async decode(latentVector: number[]): Promise<number[][]> {
    if (!this.decoder || !this.isInitialized) {
      await this.initialize();
    }

    const latentTensor = tf.tensor2d([latentVector]);
    const reconstruction = this.decoder.predict(latentTensor) as tf.Tensor;
    
    const reconData = await reconstruction.data();
    const reconArray = Array.from(reconData);
    
    // Reshape to 5x5 for visualization (downsampled)
    const visSize = 5;
    const result: number[][] = [];
    const step = Math.floor(MAZE_SIZE / visSize);
    
    for (let y = 0; y < visSize; y++) {
      const row: number[] = [];
      for (let x = 0; x < visSize; x++) {
        const idx = (y * step) * MAZE_SIZE + (x * step);
        row.push(reconArray[idx] || 0);
      }
      result.push(row);
    }
    
    latentTensor.dispose();
    reconstruction.dispose();
    
    return result;
  }

  /**
   * Train VAE on maze states
   */
  async train(mazeStates: MazeState[], epochs: number = 5) {
    if (!this.encoder || !this.decoder || !this.isInitialized) {
      await this.initialize();
    }

    const optimizer = tf.train.adam(0.001);
    
    for (let epoch = 0; epoch < epochs; epoch++) {
      for (const mazeState of mazeStates) {
        const mazeTensor = this.mazeToTensor(mazeState);
        const [zMean, zLogVar] = this.encoder!.predict(mazeTensor) as tf.Tensor[];
        
        // Sample z
        const meanData = await zMean.data();
        const logVarData = await zLogVar.data();
        const zArray = Array.from(meanData).map((mean, i) => {
          const std = Math.exp(Array.from(logVarData)[i] * 0.5);
          const epsilon = (Math.random() * 2 - 1) * 0.1;
          return mean + std * epsilon;
        });
        const z = tf.tensor2d([zArray]);
        
        // Decode
        const reconstruction = this.decoder!.predict(z) as tf.Tensor;
        
        // VAE Loss: Reconstruction + KL Divergence
        const reconstructionLoss = tf.losses.meanSquaredError(
          mazeTensor,
          reconstruction
        );
        
        // KL divergence: -0.5 * sum(1 + log_var - mean^2 - exp(log_var))
        const klLoss = tf.mul(
          -0.5,
          tf.sum(
            tf.add(
              1,
              tf.sub(
                zLogVar,
                tf.add(
                  tf.square(zMean),
                  tf.exp(zLogVar)
                )
              )
            )
          )
        );
        
        const totalLoss = tf.add(reconstructionLoss, klLoss);
        
        optimizer.minimize(() => {
          return totalLoss.mean() as tf.Scalar;
        });
        
        // Cleanup
        mazeTensor.dispose();
        zMean.dispose();
        zLogVar.dispose();
        z.dispose();
        reconstruction.dispose();
        reconstructionLoss.dispose();
        klLoss.dispose();
        totalLoss.dispose();
        
        // Force cleanup
        await tf.nextFrame();
      }
    }
  }

  /**
   * Dispose resources
   */
  dispose() {
    if (this.encoder) {
      this.encoder.dispose();
      this.encoder = null;
    }
    if (this.decoder) {
      this.decoder.dispose();
      this.decoder = null;
    }
    this.isInitialized = false;
  }
}
