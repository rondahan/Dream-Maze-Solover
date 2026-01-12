/**
 * Lazy loader for TensorFlow.js
 * Only loads TensorFlow when actually needed, improving initial page load time
 */

let tfPromise: Promise<typeof import('@tensorflow/tfjs')> | null = null;
let tfModule: typeof import('@tensorflow/tfjs') | null = null;

/**
 * Lazy load TensorFlow.js
 * Returns a promise that resolves when TensorFlow is ready
 */
export async function loadTensorFlow(): Promise<typeof import('@tensorflow/tfjs')> {
  // If already loaded, return immediately
  if (tfModule) {
    return tfModule;
  }

  // If already loading, return the existing promise
  if (tfPromise) {
    return tfPromise;
  }

  // Start loading TensorFlow.js
  tfPromise = import('@tensorflow/tfjs').then((module) => {
    tfModule = module;
    console.log('TensorFlow.js loaded successfully');
    return module;
  }).catch((error) => {
    console.error('Failed to load TensorFlow.js:', error);
    tfPromise = null; // Reset so we can retry
    throw error;
  });

  return tfPromise;
}

/**
 * Check if TensorFlow.js is already loaded
 */
export function isTensorFlowLoaded(): boolean {
  return tfModule !== null;
}

/**
 * Get the TensorFlow module if already loaded (synchronous)
 * Returns null if not loaded yet
 */
export function getTensorFlow(): typeof import('@tensorflow/tfjs') | null {
  return tfModule;
}
