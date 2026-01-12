# 🛠️ Technical Stack

Complete technical documentation for the Dreaming Maze Solver project.

---

## Technology Overview

The project is built using modern web technologies with a focus on:
- **Client-side machine learning** using TensorFlow.js
- **React** for interactive UI components
- **TypeScript** for type safety
- **Vite** for fast development and building

---

## Core Technologies

### Frontend Framework

**React 19.2.3**
- Modern React with hooks
- Functional components
- State management with useState and useRef
- Effect management with useEffect

**Why React:**
- Excellent for interactive UIs
- Component-based architecture
- Great ecosystem and community
- Perfect for real-time visualizations

### Type System

**TypeScript 5.8.2**
- Full type safety
- Interface definitions
- Type inference
- Compile-time error checking

**Benefits:**
- Catches errors early
- Better IDE support
- Self-documenting code
- Easier refactoring

### Build Tool

**Vite 6.2.0**
- Fast development server
- Hot module replacement (HMR)
- Optimized production builds
- ES modules support

**Features:**
- Instant server start
- Lightning-fast HMR
- Optimized bundling
- Tree-shaking

### Machine Learning

**TensorFlow.js 4.22.0**
- Browser-based ML
- Neural network implementation
- GPU acceleration support
- Model training and inference

**Used For:**
- Policy Network (Controller)
- VAE (Vision)
- MDN-RNN (Memory)
- Training and inference

### Styling

**Tailwind CSS (via CDN)**
- Utility-first CSS framework
- Responsive design
- Dark theme support
- Custom styling

**Benefits:**
- Rapid UI development
- Consistent design system
- Responsive by default
- Small bundle size

---

## Project Structure

### Components (`/components`)

React UI components for visualization:

**MazeBoard.tsx**
- Renders the maze grid
- Displays agent, goal, and path
- Handles cell rendering
- Updates in real-time

**LatentVisualizer.tsx**
- Shows latent vector values
- Displays VAE reconstruction
- Visual representation of Vision component
- Real-time updates

**DreamState.tsx**
- Shows dream predictions
- Displays internal monologue
- Confidence visualization
- Updates every 5 steps

### Services (`/services`)

Machine learning services:

**policyNetwork.ts**
- Policy Network implementation
- Action selection logic
- Training with REINFORCE
- Experience collection

**vae.ts**
- Variational Autoencoder
- Encoder/decoder networks
- Latent space encoding
- Reconstruction generation

**mdnRnn.ts**
- MDN-RNN implementation
- LSTM layers
- Sequence prediction
- Dream generation

**trainingSystem.ts**
- Experience replay buffer
- Batch sampling
- Learning rate scheduling
- Training orchestration

**transferLearning.ts**
- Knowledge base management
- Weight transfer
- Multi-environment learning
- Knowledge sharing

**geminiService.ts**
- State prediction
- Internal monologue generation
- Dream description
- Context-aware responses

### Utilities (`/utils`)

Helper functions:

**mazeGenerator.ts**
- Maze generation algorithms
- Recursive backtracking
- Path finding
- Grid manipulation

**environmentGenerator.ts**
- Environment type generation
- Multiple environment types
- Configuration management
- State initialization

### Configuration Files

**constants.ts**
- Application constants
- Maze configuration
- ML parameters
- UI settings

**types.ts**
- TypeScript type definitions
- Interfaces for data structures
- Type exports
- Type safety

**vite.config.ts**
- Vite configuration
- Build settings
- Plugin configuration
- Development server settings

**tsconfig.json**
- TypeScript configuration
- Compiler options
- Module resolution
- Type checking settings

---

## Machine Learning Implementation

### Neural Networks

All neural networks are implemented using TensorFlow.js:

**Architecture:**
- Layers API for building networks
- Sequential and Functional models
- Custom layer configurations
- Weight management

**Training:**
- Gradient computation
- Optimizer configuration (Adam)
- Loss functions
- Batch processing

**Inference:**
- Forward passes
- State management
- Prediction generation
- Real-time processing

### TensorFlow.js Features Used

**Core API:**
- `tf.tensor()` - Data structures
- `tf.model()` - Model creation
- `tf.layers.*` - Layer definitions
- `tf.train.*` - Optimizers

**Operations:**
- Matrix operations
- Convolutional operations
- Activation functions
- Loss computations

**Memory Management:**
- Tensor disposal
- Memory cleanup
- Efficient computation
- GPU utilization

---

## Development Workflow

### Local Development

1. **Start Dev Server:**
   ```bash
   npm run dev
   ```

2. **Make Changes:**
   - Edit source files
   - Changes hot-reload automatically
   - See updates in browser instantly

3. **Test:**
   - Use browser DevTools
   - Check console for errors
   - Test functionality

### Building

1. **Production Build:**
   ```bash
   npm run build
   ```

2. **Output:**
   - Optimized JavaScript
   - Minified code
   - Tree-shaken dependencies
   - Ready for deployment

### Deployment

**Static Hosting:**
- Deploy `dist/` folder
- No server required
- Works with any static host

**Recommended Platforms:**
- Vercel
- Netlify
- GitHub Pages
- Any static hosting

---

## Dependencies

### Production Dependencies

```json
{
  "@tensorflow/tfjs": "^4.22.0",  // Machine learning
  "react": "^19.2.3",              // UI framework
  "react-dom": "^19.2.3",         // React DOM
  "recharts": "^3.6.0"            // Charts (if used)
}
```

### Development Dependencies

```json
{
  "@types/node": "^22.14.0",      // Node types
  "@vitejs/plugin-react": "^5.0.0", // Vite React plugin
  "typescript": "~5.8.2",          // TypeScript
  "vite": "^6.2.0"                 // Build tool
}
```

---

## Performance Considerations

### Optimization Strategies

**Code Splitting:**
- Lazy loading components
- Dynamic imports
- Route-based splitting

**Bundle Size:**
- Tree-shaking unused code
- Minification
- Compression
- CDN for large libraries

**Runtime Performance:**
- Efficient React rendering
- TensorFlow.js optimization
- Memory management
- GPU acceleration

### Browser Compatibility

**Supported Browsers:**
- Chrome 90+
- Firefox 88+
- Safari 14+
- Edge 90+

**Features Required:**
- ES6+ support
- WebGL (for TensorFlow.js GPU)
- Modern JavaScript APIs

---

## Security

### Client-Side Only

- No server required
- No API keys needed
- No data transmission
- All processing local

### Best Practices

- TypeScript for type safety
- Input validation
- Error handling
- Secure coding practices

---

## Future Enhancements

### Potential Improvements

**Performance:**
- Web Workers for ML computation
- WebAssembly for faster execution
- Model quantization
- Caching strategies

**Features:**
- More environment types
- Advanced visualizations
- Export/import models
- Comparison tools

**Development:**
- Unit tests
- Integration tests
- CI/CD pipeline
- Automated deployment

---

## Resources

### Documentation

- [React Documentation](https://react.dev/)
- [TensorFlow.js Guide](https://www.tensorflow.org/js)
- [TypeScript Handbook](https://www.typescriptlang.org/docs/)
- [Vite Documentation](https://vitejs.dev/)

### Learning Resources

- [World Models Paper](https://arxiv.org/abs/1803.10122)
- [Reinforcement Learning](https://spinningup.openai.com/)
- [Neural Networks](https://neuralnetworksanddeeplearning.com/)

---

[← Back to README](../README.md)
