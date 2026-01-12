
import React, { useState, useEffect, useCallback, useRef } from 'react';
import { MazeState, CellType, Position, LatentVector, DreamPrediction, AgentLog } from './types';
import { MAZE_SIZE, TICK_RATE, LATENT_DIM } from './constants';
import { generateMaze } from './utils/mazeGenerator';
import { EnvironmentGenerator, EnvironmentType } from './utils/environmentGenerator';
import { getAgentInternalMonologue, predictNextState } from './services/geminiService';
import { PolicyNetwork } from './services/policyNetwork';
import { VAE } from './services/vae';
import { MDNRNN } from './services/mdnRnn';
import { TrainingSystem } from './services/trainingSystem';
import { TransferLearning } from './services/transferLearning';
import MazeBoard from './components/MazeBoard';
import LatentVisualizer from './components/LatentVisualizer';
import DreamState from './components/DreamState';

const App: React.FC = () => {
  const [maze, setMaze] = useState<MazeState | null>(null);
  const [isRunning, setIsRunning] = useState(false);
  const [isInitializing, setIsInitializing] = useState(true);
  const [curiosityMultiplier, setCuriosityMultiplier] = useState(5);
  const [speed, setSpeed] = useState(1); // Speed multiplier: 1 = normal, higher = faster
  const [latent, setLatent] = useState<LatentVector>({ 
    vector: Array(LATENT_DIM).fill(0).map(() => Math.random()), 
    reconstruction: Array(5).fill(0).map(() => Array(5).fill(0).map(() => Math.random()))
  });
  const [dream, setDream] = useState<DreamPrediction | null>(null);
  const [monologue, setMonologue] = useState<string>("Initializing world model architecture...");
  const [logs, setLogs] = useState<AgentLog[]>([]);
  const [stats, setStats] = useState({ 
    steps: 0, 
    reward: 0, 
    goalReached: false, 
    curiosityScore: 0,
    currentMode: 'initializing' as 'explore' | 'exploit' | 'initializing',
    epsilon: 1.0,
    useNeuralNetwork: true, // Toggle between NN and Epsilon-Greedy
    trainingExperiences: 0, // Number of experiences in training buffer
    environmentType: EnvironmentType.MAZE as EnvironmentType,
    useVAE: true, // Use real VAE instead of random
    useMDNRNN: true // Use real MDN-RNN instead of simple prediction
  });

  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const policyNetworkRef = useRef<PolicyNetwork | null>(null);
  const vaeRef = useRef<VAE | null>(null);
  const mdnRnnRef = useRef<MDNRNN | null>(null);
  const trainingSystemRef = useRef<TrainingSystem | null>(null);
  const knowledgeBaseRef = useRef<ReturnType<typeof TransferLearning.createKnowledgeBase> | null>(null);
  const trainingBufferRef = useRef<{
    states: number[][];
    actions: number[];
    rewards: number[];
  }>({ states: [], actions: [], rewards: [] });
  const lastActionRef = useRef<'up' | 'down' | 'left' | 'right' | null>(null);

  const addLog = useCallback((message: string, type: AgentLog['type']) => {
    const newLog: AgentLog = {
      id: Math.random().toString(36).substr(2, 9),
      timestamp: Date.now(),
      message,
      type
    };
    setLogs(prev => [newLog, ...prev].slice(0, 10));
  }, []);

  const initMaze = useCallback((envType?: EnvironmentType) => {
    const type = envType || stats.environmentType;
    const { grid, start, goal } = EnvironmentGenerator.generate(type, MAZE_SIZE);
    setMaze({
      grid,
      agentPos: start,
      goalPos: goal,
      history: [start]
    });
    setStats(prev => ({ 
      steps: 0, 
      reward: 0, 
      goalReached: false, 
      curiosityScore: 0,
      currentMode: 'initializing',
      epsilon: 1.0,
      useNeuralNetwork: prev.useNeuralNetwork,
      trainingExperiences: 0,
      environmentType: type,
      useVAE: prev.useVAE,
      useMDNRNN: prev.useMDNRNN
    }));
    trainingBufferRef.current = { states: [], actions: [], rewards: [] };
    if (mdnRnnRef.current) {
      mdnRnnRef.current.resetStates();
    }
    addLog(`System initialized. ${type} environment generated.`, "action");
    setIsRunning(false);
  }, [addLog, stats.environmentType]);

  const getNextAction = useCallback(async (
    m: MazeState, 
    currentSteps: number, 
    curiosityWeight: number
  ): Promise<{ pos: Position, mode: 'explore' | 'exploit', epsilon: number, actionIndex?: number }> => {
    const { agentPos, goalPos, grid, history } = m;
    
    // Epsilon-Greedy Strategy: Decays from 1.0 to 0.15 over 100 steps
    const epsilon = Math.max(0.15, 1 - currentSteps / 100);

    const dirs = [
      { x: 0, y: 1 }, { x: 0, y: -1 }, { x: 1, y: 0 }, { x: -1, y: 0 }
    ];

    const possibleMoves = dirs
      .map(dir => ({ x: agentPos.x + dir.x, y: agentPos.y + dir.y }))
      .filter(p => p.x >= 0 && p.x < MAZE_SIZE && p.y >= 0 && p.y < MAZE_SIZE && grid[p.y][p.x] !== CellType.WALL);

    if (possibleMoves.length === 0) return { pos: agentPos, mode: 'exploit', epsilon };

    // Use Neural Network if enabled and initialized
    if (stats.useNeuralNetwork && policyNetworkRef.current) {
      try {
        const result = await policyNetworkRef.current.selectAction(
          m,
          possibleMoves,
          currentSteps,
          curiosityWeight,
          epsilon
        );
        
        // Determine mode based on whether it's exploring or exploiting
        const isNewArea = !history.some(hp => hp.x === result.pos.x && hp.y === result.pos.y);
        const mode = isNewArea ? 'explore' : 'exploit';
        
        return { 
          pos: result.pos, 
          mode, 
          epsilon,
          actionIndex: result.actionIndex
        };
      } catch (error) {
        console.error('Neural network error, falling back to epsilon-greedy:', error);
        // Fall through to epsilon-greedy
      }
    }

    // Fallback: Epsilon-Greedy Strategy
    const isExploring = Math.random() < epsilon;

    if (isExploring) {
      // EXPLORATION: Prioritize unvisited neighbors with a weight tied to the curiosity multiplier
      const unvisited = possibleMoves.filter(p => !history.some(hp => hp.x === p.x && hp.y === p.y));
      
      if (unvisited.length > 0 && curiosityWeight > 0) {
        // If curiosity is high, we almost always take unvisited
        const chosen = unvisited[Math.floor(Math.random() * unvisited.length)];
        return { pos: chosen, mode: 'explore', epsilon };
      }
      
      const chosen = possibleMoves[Math.floor(Math.random() * possibleMoves.length)];
      return { pos: chosen, mode: 'explore', epsilon };
    } else {
      // EXPLOITATION: Greedy distance to goal
      let bestMove = possibleMoves[0];
      let minDist = Infinity;
      for (const p of possibleMoves) {
        const dist = Math.abs(p.x - goalPos.x) + Math.abs(p.y - goalPos.y);
        if (dist < minDist) {
          minDist = dist;
          bestMove = p;
        }
      }
      return { pos: bestMove, mode: 'exploit', epsilon };
    }
  }, [stats.useNeuralNetwork]);

  const step = useCallback(async () => {
    if (!maze || stats.goalReached) return;

    // Get state features for neural network training
    const stateFeatures = policyNetworkRef.current ? (() => {
      const { agentPos, goalPos, history } = maze;
      const distance = Math.abs(agentPos.x - goalPos.x) + Math.abs(agentPos.y - goalPos.y);
      const normalizedAgentX = agentPos.x / MAZE_SIZE;
      const normalizedAgentY = agentPos.y / MAZE_SIZE;
      const normalizedGoalX = goalPos.x / MAZE_SIZE;
      const normalizedGoalY = goalPos.y / MAZE_SIZE;
      const normalizedDistance = distance / (MAZE_SIZE * 2);
      const visitedRatio = history.length / (MAZE_SIZE * MAZE_SIZE);
      const epsilon = Math.max(0.15, 1 - stats.steps / 100);
      const normalizedCuriosity = curiosityMultiplier / 25;
      return [
        normalizedAgentX, normalizedAgentY, normalizedGoalX, normalizedGoalY,
        normalizedDistance, visitedRatio, normalizedCuriosity, epsilon
      ];
    })() : null;

    const { pos: nextPos, mode, epsilon, actionIndex } = await getNextAction(maze, stats.steps, curiosityMultiplier);
    const reachedGoal = nextPos.x === maze.goalPos.x && nextPos.y === maze.goalPos.y;
    const isNewArea = !maze.history.some(p => p.x === nextPos.x && p.y === nextPos.y);

    // Curiosity Reward Logic using the adjustable multiplier
    const stepCost = -1;
    const goalBonus = reachedGoal ? 100 : 0;
    const curiosityBonus = isNewArea ? curiosityMultiplier : 0;
    const totalRewardIncrement = stepCost + goalBonus + curiosityBonus;

    // Store experience for training (if using neural network)
    if (stats.useNeuralNetwork && stateFeatures && actionIndex !== undefined && actionIndex >= 0) {
      trainingBufferRef.current.states.push(stateFeatures);
      trainingBufferRef.current.actions.push(actionIndex);
      trainingBufferRef.current.rewards.push(totalRewardIncrement);
      
      // Also add to training system for experience replay
      if (trainingSystemRef.current) {
        const nextStateFeatures = stateFeatures; // Simplified - in real implementation would compute next state
        trainingSystemRef.current.addExperience({
          state: stateFeatures,
          action: actionIndex,
          reward: totalRewardIncrement,
          nextState: nextStateFeatures,
          done: reachedGoal
        });
      }
      
      setStats(prev => ({ ...prev, trainingExperiences: trainingBufferRef.current.states.length }));
    }

    // V-M loop: Use VAE if enabled, otherwise use random
    // Optimize: Only encode every few steps to reduce computation
    let newLatent: LatentVector;
    if (stats.useVAE && vaeRef.current && maze && stats.steps % 3 === 0) {
      try {
        newLatent = await vaeRef.current.encode(maze);
      } catch (error) {
        console.error('VAE encoding error:', error);
        // Fallback to random
        newLatent = {
          vector: Array(LATENT_DIM).fill(0).map(() => Math.random()),
          reconstruction: Array(5).fill(0).map(() => Array(5).fill(0).map(() => Math.random()))
        };
      }
    } else {
      // Reuse previous latent or generate random
      newLatent = latent || {
        vector: Array(LATENT_DIM).fill(0).map(() => Math.random()),
        reconstruction: Array(5).fill(0).map(() => Array(5).fill(0).map(() => Math.random()))
      };
    }
    setLatent(newLatent);

    // Determine action direction for MDN-RNN
    const dx = nextPos.x - maze.agentPos.x;
    const dy = nextPos.y - maze.agentPos.y;
    let action: 'up' | 'down' | 'left' | 'right' = 'right';
    if (dy > 0) action = 'down';
    else if (dy < 0) action = 'up';
    else if (dx < 0) action = 'left';
    else if (dx > 0) action = 'right';
    lastActionRef.current = action;

    // Occasional "Dream" analysis - optimize: reduce frequency and make non-blocking
    if (stats.steps % 10 === 0) {
      // Run dream analysis asynchronously without blocking the main step
      Promise.resolve().then(async () => {
        let d: DreamPrediction;
        if (stats.useMDNRNN && mdnRnnRef.current && lastActionRef.current) {
          try {
            // Predict sequence using MDN-RNN
            const actions: ('up' | 'down' | 'left' | 'right')[] = [
              lastActionRef.current, lastActionRef.current, lastActionRef.current, 
              lastActionRef.current, lastActionRef.current
            ];
            d = await mdnRnnRef.current.predictSequence(newLatent.vector, actions, maze);
          } catch (error) {
            console.error('MDN-RNN prediction error:', error);
            // Fallback to simple prediction
            d = await predictNextState(maze);
          }
        } else {
          d = await predictNextState(maze);
        }
        setDream(d);
        // Only update monologue occasionally to reduce API calls
        if (stats.steps % 20 === 0) {
          try {
            const m = await getAgentInternalMonologue(maze);
            setMonologue(m);
          } catch (error) {
            console.error('Monologue error:', error);
          }
        }
        if (mode === 'explore') {
          addLog(`Internal State: Novelty seeking (ε=${epsilon.toFixed(2)}, Curiosity x${curiosityMultiplier}).`, "dream");
        }
      }).catch(error => {
        console.error('Dream analysis error:', error);
      });
    }

    if (isNewArea && !reachedGoal) {
      addLog(`Sensory Input: Novel area detected. Reward: +${curiosityMultiplier}`, "vision");
    }

    setMaze(prev => {
      if (!prev) return null;
      // Optimize: Limit history size to prevent memory growth
      const newHistory = [...prev.history, nextPos];
      const maxHistorySize = MAZE_SIZE * MAZE_SIZE; // Limit to maze area
      const trimmedHistory = newHistory.length > maxHistorySize 
        ? newHistory.slice(-maxHistorySize)
        : newHistory;
      
      return {
        ...prev,
        agentPos: nextPos,
        history: trimmedHistory
      };
    });

    setStats(prev => ({
      ...prev,
      steps: prev.steps + 1,
      reward: prev.reward + totalRewardIncrement,
      curiosityScore: prev.curiosityScore + curiosityBonus,
      goalReached: reachedGoal,
      currentMode: mode,
      epsilon: epsilon
    }));

    if (reachedGoal) {
      addLog("Goal state achieved. Terminating episode.", "vision");
      setIsRunning(false);
      
      // Train neural network at end of episode
      if (stats.useNeuralNetwork && policyNetworkRef.current && trainingBufferRef.current.states.length > 0) {
        try {
          // Use training system if available
          if (trainingSystemRef.current) {
            await trainingSystemRef.current.trainPolicyNetwork(policyNetworkRef.current, 10);
            addLog(`Neural network trained with experience replay (${trainingSystemRef.current.getStats().bufferSize} experiences).`, "action");
          } else {
            await policyNetworkRef.current.train(
              trainingBufferRef.current.states,
              trainingBufferRef.current.actions,
              trainingBufferRef.current.rewards
            );
            addLog(`Neural network trained on ${trainingBufferRef.current.states.length} experiences.`, "action");
          }
          // Clear buffer
          trainingBufferRef.current = { states: [], actions: [], rewards: [] };
          setStats(prev => ({ ...prev, trainingExperiences: 0 }));
        } catch (error) {
          console.error('Training error:', error);
        }
      }
    }

    // Periodic training (every 20 steps)
    if (stats.useNeuralNetwork && policyNetworkRef.current && 
        trainingBufferRef.current.states.length >= 20 && 
        stats.steps % 20 === 0) {
      try {
        if (trainingSystemRef.current) {
          await trainingSystemRef.current.trainPolicyNetwork(policyNetworkRef.current, 3);
        } else {
          await policyNetworkRef.current.train(
            trainingBufferRef.current.states.slice(-20),
            trainingBufferRef.current.actions.slice(-20),
            trainingBufferRef.current.rewards.slice(-20)
          );
        }
        // Keep buffer size manageable
        if (trainingBufferRef.current.states.length > 100) {
          trainingBufferRef.current = {
            states: trainingBufferRef.current.states.slice(-50),
            actions: trainingBufferRef.current.actions.slice(-50),
            rewards: trainingBufferRef.current.rewards.slice(-50)
          };
        }
      } catch (error) {
        console.error('Periodic training error:', error);
      }
    }
  }, [maze, stats.goalReached, stats.steps, stats.useNeuralNetwork, getNextAction, curiosityMultiplier, addLog]);

  useEffect(() => {
    if (isRunning && !stats.goalReached) {
      // Calculate dynamic tick rate: lower speed multiplier = faster (shorter delay)
      // Speed 1 = 300ms, Speed 5 = 60ms, Speed 10 = 30ms
      const dynamicTickRate = Math.max(10, TICK_RATE / speed);
      timerRef.current = setInterval(step, dynamicTickRate);
    } else {
      if (timerRef.current) clearInterval(timerRef.current);
    }
    return () => {
      if (timerRef.current) clearInterval(timerRef.current);
    };
  }, [isRunning, step, stats.goalReached, speed]);

  // Initialize all neural networks asynchronously to avoid blocking UI
  useEffect(() => {
    let cancelled = false;
    
    const initNetworks = async () => {
      setIsInitializing(true);
      
      try {
        // Initialize Policy Network
        policyNetworkRef.current = new PolicyNetwork();
        await policyNetworkRef.current.initialize();
        if (!cancelled) {
          addLog("Neural Network Policy initialized. Ready for learning.", "action");
        }
        
        // Allow UI to update between initializations
        await new Promise(resolve => setTimeout(resolve, 50));
        
        // Initialize VAE
        if (stats.useVAE && !cancelled) {
          vaeRef.current = new VAE();
          await vaeRef.current.initialize();
          if (!cancelled) {
            addLog("VAE (Vision) initialized. Encoding maze states.", "vision");
          }
          await new Promise(resolve => setTimeout(resolve, 50));
        }
        
        // Initialize MDN-RNN
        if (stats.useMDNRNN && !cancelled) {
          mdnRnnRef.current = new MDNRNN();
          await mdnRnnRef.current.initialize();
          if (!cancelled) {
            addLog("MDN-RNN (Memory) initialized. Ready for dreaming.", "dream");
          }
          await new Promise(resolve => setTimeout(resolve, 50));
        }
        
        // Initialize Training System
        if (!cancelled) {
          trainingSystemRef.current = new TrainingSystem(10000, 32);
          addLog("Training System initialized. Experience replay enabled.", "action");
        }
        
        // Initialize Knowledge Base for transfer learning
        if (!cancelled) {
          knowledgeBaseRef.current = TransferLearning.createKnowledgeBase();
        }
        
        if (!cancelled) {
          setIsInitializing(false);
        }
      } catch (error) {
        console.error('Network initialization error:', error);
        if (!cancelled) {
          setIsInitializing(false);
        }
      }
    };
    
    initNetworks();

    return () => {
      cancelled = true;
      if (policyNetworkRef.current) {
        policyNetworkRef.current.dispose();
      }
      if (vaeRef.current) {
        vaeRef.current.dispose();
      }
      if (mdnRnnRef.current) {
        mdnRnnRef.current.dispose();
      }
      if (knowledgeBaseRef.current) {
        knowledgeBaseRef.current.dispose();
      }
    };
  }, [addLog, stats.useVAE, stats.useMDNRNN]);

  useEffect(() => {
    initMaze();
  }, [initMaze]);

  return (
    <div className="min-h-screen bg-slate-950 text-slate-200 p-6 md:p-12 font-sans flex flex-col items-center">
      {/* Header */}
      <div className="w-full max-w-6xl flex flex-col md:flex-row justify-between items-start md:items-center gap-6 mb-8">
        <div>
          <h1 className="text-4xl font-black tracking-tighter text-transparent bg-clip-text bg-gradient-to-r from-blue-400 via-purple-500 to-emerald-400">
            DREAMING MAZE SOLVER
          </h1>
          <p className="text-slate-500 mt-1 font-mono text-xs uppercase tracking-[0.2em]">
            Autonomous World Model Architecture (V-M-C)
          </p>
        </div>
        
        <div className="flex flex-wrap gap-4 items-center">
          {/* Curiosity Slider */}
          <div className="bg-slate-900 border border-slate-700 rounded-full px-4 py-2 flex items-center gap-4">
            <span className="text-[10px] font-bold text-slate-500 uppercase">Curiosity Multiplier</span>
            <input 
              type="range" 
              min="0" 
              max="25" 
              step="1"
              value={curiosityMultiplier}
              onChange={(e) => setCuriosityMultiplier(parseInt(e.target.value))}
              className="w-32 accent-emerald-500 cursor-pointer"
            />
            <span className="text-xs font-mono text-emerald-400 w-4 text-center">{curiosityMultiplier}</span>
          </div>

          {/* Speed Control */}
          <div className="bg-slate-900 border border-slate-700 rounded-full px-4 py-2 flex items-center gap-4">
            <span className="text-[10px] font-bold text-slate-500 uppercase">Speed</span>
            <input 
              type="range" 
              min="1" 
              max="10" 
              step="0.5"
              value={speed}
              onChange={(e) => setSpeed(parseFloat(e.target.value))}
              className="w-32 accent-blue-500 cursor-pointer"
            />
            <span className="text-xs font-mono text-blue-400 w-8 text-center">{speed.toFixed(1)}x</span>
          </div>

          <button 
            onClick={() => setIsRunning(!isRunning)}
            disabled={stats.goalReached || isInitializing}
            className={`px-8 py-3 rounded-full font-bold transition-all duration-300 flex items-center gap-3 ${
              isRunning 
                ? 'bg-red-500/10 text-red-400 border border-red-500/50 hover:bg-red-500/20' 
                : isInitializing
                ? 'bg-slate-700 text-slate-400 cursor-not-allowed'
                : 'bg-blue-600 text-white hover:bg-blue-500 shadow-[0_0_20px_rgba(37,99,235,0.4)]'
            }`}
          >
            {isRunning ? <i className="fa-solid fa-pause"></i> : <i className="fa-solid fa-play"></i>}
            {isInitializing ? 'INITIALIZING...' : isRunning ? 'HALT PROCESS' : 'START SIMULATION'}
          </button>
          
          <button 
            onClick={initMaze}
            className="px-6 py-3 bg-slate-800 border border-slate-700 rounded-full font-bold hover:bg-slate-700 transition-all flex items-center gap-3"
          >
            <i className="fa-solid fa-rotate-right"></i>
            NEW WORLD
          </button>

          {/* Neural Network Toggle */}
          <button
            onClick={() => {
              setStats(prev => ({ 
                ...prev, 
                useNeuralNetwork: !prev.useNeuralNetwork,
                trainingExperiences: 0
              }));
              trainingBufferRef.current = { states: [], actions: [], rewards: [] };
            }}
            className={`px-6 py-3 rounded-full font-bold transition-all flex items-center gap-3 ${
              stats.useNeuralNetwork
                ? 'bg-purple-600 text-white hover:bg-purple-500 shadow-[0_0_20px_rgba(147,51,234,0.4)]'
                : 'bg-slate-800 border border-slate-700 hover:bg-slate-700'
            }`}
            title={stats.useNeuralNetwork ? "Using Neural Network Policy" : "Using Epsilon-Greedy Policy"}
          >
            <i className={`fa-solid ${stats.useNeuralNetwork ? 'fa-brain' : 'fa-code'}`}></i>
            {stats.useNeuralNetwork ? 'NN ON' : 'NN OFF'}
          </button>

          {/* Environment Type Selector */}
          <select
            value={stats.environmentType}
            onChange={(e) => {
              const newType = e.target.value as EnvironmentType;
              setStats(prev => ({ ...prev, environmentType: newType }));
              initMaze(newType);
            }}
            className="px-4 py-3 bg-slate-800 border border-slate-700 rounded-full font-bold text-sm hover:bg-slate-700 transition-all"
          >
            <option value={EnvironmentType.MAZE}>Maze</option>
            <option value={EnvironmentType.OPEN_FIELD}>Open Field</option>
            <option value={EnvironmentType.CORRIDOR}>Corridor</option>
            <option value={EnvironmentType.SPIRAL}>Spiral</option>
            <option value={EnvironmentType.GRID}>Grid</option>
          </select>

          {/* Transfer Learning Button */}
          <button
            onClick={async () => {
              if (knowledgeBaseRef.current && policyNetworkRef.current && vaeRef.current && mdnRnnRef.current) {
                await knowledgeBaseRef.current.savePolicy(policyNetworkRef.current);
                await knowledgeBaseRef.current.saveVAE(vaeRef.current);
                await knowledgeBaseRef.current.saveRNN(mdnRnnRef.current);
                addLog("Knowledge saved to shared base. Ready for transfer.", "action");
              }
            }}
            className="px-6 py-3 bg-emerald-600 text-white rounded-full font-bold hover:bg-emerald-500 transition-all flex items-center gap-3"
            title="Save current knowledge for transfer learning"
          >
            <i className="fa-solid fa-download"></i>
            SAVE KNOWLEDGE
          </button>
        </div>
      </div>

      <div className="w-full max-w-6xl grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Left Column: Vision & Memory Panels */}
        <div className="lg:col-span-3 space-y-6 flex flex-col">
          <LatentVisualizer latent={latent} />
          <div className="flex-1">
            <DreamState dream={dream} monologue={monologue} />
          </div>
        </div>

        {/* Center Column: The Maze */}
        <div className="lg:col-span-6 flex flex-col items-center">
          {isInitializing ? (
            <div className="flex flex-col items-center justify-center h-96 bg-slate-900 border-2 border-slate-700 rounded-lg">
              <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-blue-500 mb-4"></div>
              <p className="text-slate-400 text-sm">Initializing neural networks...</p>
            </div>
          ) : maze ? (
            <MazeBoard maze={maze} />
          ) : null}
          
          <div className="w-full grid grid-cols-2 md:grid-cols-4 gap-4 mt-6">
            <div className="bg-slate-900/50 p-4 border border-slate-800 rounded-xl text-center">
              <span className="text-[10px] text-slate-500 block mb-1">STEPS</span>
              <span className="text-2xl font-mono text-blue-400">{stats.steps}</span>
            </div>
            <div className="bg-slate-900/50 p-4 border border-slate-800 rounded-xl text-center">
              <span className="text-[10px] text-slate-500 block mb-1">EPSILON (ε)</span>
              <span className="text-2xl font-mono text-amber-400">{stats.epsilon.toFixed(2)}</span>
            </div>
            <div className="bg-slate-900/50 p-4 border border-slate-800 rounded-xl text-center">
              <span className="text-[10px] text-slate-500 block mb-1">REWARD</span>
              <span className={`text-2xl font-mono ${stats.reward >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
                {stats.reward}
              </span>
            </div>
            <div className="bg-slate-900/50 p-4 border border-slate-800 rounded-xl text-center">
              <span className="text-[10px] text-slate-500 block mb-1">CURIOSITY</span>
              <span className="text-2xl font-mono text-emerald-400">+{stats.curiosityScore}</span>
            </div>
          </div>
          
          {/* Neural Network Training Info */}
          {stats.useNeuralNetwork && (
            <div className="mt-4 bg-purple-900/20 border border-purple-500/30 rounded-xl p-3">
              <div className="flex items-center justify-between mb-2">
                <span className="text-[10px] text-purple-400 font-bold uppercase">Neural Network</span>
                <span className="text-[10px] text-purple-300 font-mono">
                  {stats.trainingExperiences} experiences
                </span>
              </div>
              <div className="text-[10px] text-purple-400/70 italic">
                Learning from experience... Policy improves over time.
              </div>
            </div>
          )}
        </div>

        {/* Right Column: Logs & Controller */}
        <div className="lg:col-span-3 flex flex-col space-y-6">
          <div className="bg-slate-900 p-4 border border-slate-700 rounded-xl flex-1 flex flex-col">
            <h3 className="text-xs font-bold text-slate-500 uppercase tracking-widest mb-4 flex items-center gap-2">
              <i className="fa-solid fa-terminal text-blue-400"></i> Controller (C) Logs
            </h3>
            <div className="flex-1 space-y-3 overflow-y-auto max-h-[400px] pr-2 custom-scrollbar">
              {logs.map((log) => (
                <div key={log.id} className="text-[11px] font-mono leading-tight border-l-2 border-slate-800 pl-3 py-1">
                  <span className="text-slate-600 block mb-1">
                    [{new Date(log.timestamp).toLocaleTimeString()}] 
                    <span className={`ml-2 uppercase ${
                      log.type === 'dream' ? 'text-purple-500' : 
                      log.type === 'vision' ? 'text-emerald-500' : 'text-blue-500'
                    }`}>{log.type}</span>
                  </span>
                  <span className="text-slate-300">{log.message}</span>
                </div>
              ))}
              {logs.length === 0 && <p className="text-slate-600 text-center mt-20 italic">Awaiting connection...</p>}
            </div>
          </div>

          <div className="bg-gradient-to-br from-blue-900/20 to-purple-900/20 p-4 border border-blue-500/30 rounded-xl">
             <div className="flex items-center justify-between mb-3">
               <div className="flex items-center gap-3">
                 <div className={`w-3 h-3 rounded-full animate-pulse shadow-[0_0_8px_currentColor] ${stats.currentMode === 'explore' ? 'text-purple-500 bg-purple-500' : 'text-blue-500 bg-blue-500'}`}></div>
                 <span className="text-xs font-bold text-slate-300 uppercase tracking-widest">Strategy Mode</span>
               </div>
               <span className={`text-[10px] font-mono font-bold px-2 py-0.5 rounded border ${
                 stats.currentMode === 'explore' ? 'text-purple-400 border-purple-500/30 bg-purple-500/10' : 'text-blue-400 border-blue-500/30 bg-blue-500/10'
               }`}>
                 {stats.currentMode.toUpperCase()}
               </span>
             </div>
             <p className="text-[10px] text-slate-400 leading-relaxed italic">
               {stats.currentMode === 'explore' 
                 ? `Seeking environment novelty and mapping latent transitions. Current curiosity bias: x${curiosityMultiplier}.`
                 : "Executing optimized policy based on accumulated internal rewards. Minimizing distance to goal."}
             </p>
          </div>
        </div>
      </div>

      {/* Background decoration */}
      <div className="fixed top-0 left-0 w-full h-full pointer-events-none -z-10 opacity-30">
        <div className="absolute top-[10%] left-[5%] w-64 h-64 bg-blue-600/10 rounded-full blur-[120px]"></div>
        <div className="absolute bottom-[10%] right-[5%] w-96 h-96 bg-purple-600/10 rounded-full blur-[150px]"></div>
      </div>
      
      <style>{`
        .custom-scrollbar::-webkit-scrollbar {
          width: 4px;
        }
        .custom-scrollbar::-webkit-scrollbar-track {
          background: rgba(15, 23, 42, 0.1);
        }
        .custom-scrollbar::-webkit-scrollbar-thumb {
          background: rgba(71, 85, 105, 0.5);
          border-radius: 10px;
        }
      `}</style>
    </div>
  );
};

export default App;
