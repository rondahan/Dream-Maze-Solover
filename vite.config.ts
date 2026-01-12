import path from 'path';
import { defineConfig, loadEnv } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig(({ mode }) => {
    const env = loadEnv(mode, process.cwd(), '');
    return {
      server: {
        port: 3000,
        host: '0.0.0.0',
      },
      plugins: [react()],
      resolve: {
        alias: {
          '@': path.resolve(__dirname, '.'),
        }
      },
      optimizeDeps: {
        // Don't pre-bundle TensorFlow - let it load lazily
        exclude: ['@tensorflow/tfjs']
      },
      build: {
        commonjsOptions: {
          include: [/node_modules/]
        },
        rollupOptions: {
          output: {
            // Code splitting for better loading performance
            manualChunks: {
              'tensorflow': ['@tensorflow/tfjs']
            }
          }
        }
      },
      // Vite automatically exposes env variables prefixed with VITE_ to import.meta.env
      // No need for manual define - just use VITE_GEMINI_API_KEY in .env
    };
});
