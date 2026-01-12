# 🚀 Getting Started

Complete setup and installation guide for the Dreaming Maze Solver.

---

## Prerequisites

### Required Software

- **Node.js 18+** 
  - Download from [nodejs.org](https://nodejs.org/)
  - Verify installation: `node --version`
  - Should show v18.0.0 or higher

- **npm** (comes with Node.js)
  - Verify installation: `npm --version`
  - Should show 9.0.0 or higher

- **Modern Web Browser**
  - Chrome (recommended)
  - Firefox
  - Safari
  - Edge

### System Requirements

- **Operating System:** Windows, macOS, or Linux
- **RAM:** 4GB minimum, 8GB recommended
- **Disk Space:** ~500MB for dependencies
- **Internet:** Required for initial npm install

---

## Installation

### Step 1: Clone or Download the Project

**Option A: Clone from Git**
```bash
git clone <repository-url>
cd "draeming maze"
```

**Option B: Download ZIP**
- Download the project ZIP file
- Extract to your desired location
- Open terminal in the project folder

### Step 2: Install Dependencies

Navigate to the project directory and run:

```bash
npm install
```

This will:
- Install all required packages
- Set up node_modules directory
- Download TensorFlow.js and React dependencies
- May take 2-5 minutes depending on internet speed

**Expected Output:**
```
added 15271 packages in 2m
```

### Step 3: Verify Installation

Check that all dependencies are installed:

```bash
npm list --depth=0
```

You should see:
- `@tensorflow/tfjs`
- `react`
- `react-dom`
- `vite`
- Other dependencies

---

## Running the Application

### Development Mode

**Start the development server:**

```bash
npm run dev
```

**Expected Output:**
```
  VITE v6.2.0  ready in 500 ms

  ➜  Local:   http://localhost:5173/
  ➜  Network: use --host to expose
```

**Open in Browser:**
- Navigate to `http://localhost:5173`
- The application will load automatically
- Hot reload is enabled (changes update automatically)

### Building for Production

**Create production build:**

```bash
npm run build
```

**Expected Output:**
```
vite v6.2.0 building for production...
✓ 1234 modules transformed.
dist/index.html                   0.45 kB
dist/assets/index-abc123.js       1.23 MB
```

**Output Location:**
- Built files are in the `dist/` directory
- Ready for deployment to any static hosting

### Preview Production Build

**Test the production build locally:**

```bash
npm run preview
```

This serves the production build locally for testing.

---

## Project Structure

```
draeming-maze/
├── components/              # React UI components
│   ├── MazeBoard.tsx       # Maze visualization
│   ├── LatentVisualizer.tsx # VAE visualization
│   └── DreamState.tsx      # Dream prediction display
├── services/               # ML services
│   ├── policyNetwork.ts    # Policy Network (Controller)
│   ├── vae.ts              # Variational Autoencoder (Vision)
│   ├── mdnRnn.ts           # MDN-RNN (Memory)
│   ├── trainingSystem.ts   # Experience replay system
│   ├── transferLearning.ts # Knowledge transfer
│   └── geminiService.ts    # State prediction & monologue
├── utils/                  # Utility functions
│   ├── mazeGenerator.ts    # Maze generation algorithms
│   └── environmentGenerator.ts # Environment type generators
├── docs/                   # Documentation
│   ├── architecture.md     # ML architecture details
│   ├── learning-algorithms.md # Learning algorithms
│   ├── user-guide.md       # User guide
│   ├── technical-stack.md  # Technical details
│   └── getting-started.md  # This file
├── types.ts                # TypeScript type definitions
├── constants.ts            # Configuration constants
├── App.tsx                 # Main application component
├── package.json            # Dependencies and scripts
├── tsconfig.json           # TypeScript configuration
├── vite.config.ts          # Vite configuration
└── README.md               # Main documentation
```

---

## Available Scripts

### Development

```bash
npm run dev
```
- Starts development server
- Enables hot module replacement
- Opens at http://localhost:5173

### Build

```bash
npm run build
```
- Creates production build
- Optimizes and minifies code
- Outputs to `dist/` directory

### Preview

```bash
npm run preview
```
- Serves production build locally
- Tests built application
- Useful for pre-deployment testing

---

## Configuration

### Constants

Edit `constants.ts` to adjust:

- **MAZE_SIZE:** Size of the maze (default: 15)
- **CELL_SIZE:** Size of each cell in pixels (default: 30)
- **LATENT_DIM:** Latent vector dimensions (default: 8)
- **TICK_RATE:** Base simulation speed in ms (default: 300)

### Environment Variables

No environment variables are required. The application runs completely client-side.

---

## Troubleshooting

### Installation Issues

**Problem: npm install fails**

**Solutions:**
- Clear npm cache: `npm cache clean --force`
- Delete `node_modules` and `package-lock.json`
- Run `npm install` again
- Check Node.js version: `node --version` (should be 18+)

**Problem: Permission errors**

**Solutions:**
- Use `sudo` (Linux/Mac): `sudo npm install`
- Or fix npm permissions: `npm config set prefix ~/.npm-global`

### Runtime Issues

**Problem: Port already in use**

**Solutions:**
- Change port in `vite.config.ts`
- Or kill process using port: `lsof -ti:5173 | xargs kill`

**Problem: Application won't load**

**Solutions:**
- Check browser console for errors (F12)
- Verify all dependencies installed: `npm list`
- Clear browser cache
- Try different browser

**Problem: Slow performance**

**Solutions:**
- Reduce MAZE_SIZE in constants.ts
- Lower speed in UI
- Close other browser tabs
- Check system resources

### Build Issues

**Problem: Build fails**

**Solutions:**
- Check for TypeScript errors: `npm run build`
- Verify all imports are correct
- Check `tsconfig.json` configuration
- Clear `dist/` folder and rebuild

---

## Next Steps

After installation:

1. **Read the [User Guide](./user-guide.md)** to learn how to use the application
2. **Explore the [Architecture](./architecture.md)** to understand the ML components
3. **Check [Learning Algorithms](./learning-algorithms.md)** to understand how it learns
4. **Review [Technical Stack](./technical-stack.md)** for development details

---

## Development Tips

### Hot Reload

Changes to code automatically reload in the browser during development.

### TypeScript

The project uses TypeScript for type safety. Check types with:
```bash
npx tsc --noEmit
```

### Debugging

- Use browser DevTools (F12) for debugging
- Check Console for errors and logs
- Use React DevTools extension for component inspection

### Performance

- Use production build for performance testing
- Monitor browser performance tab
- Check TensorFlow.js memory usage

---

[← Back to README](../README.md)
