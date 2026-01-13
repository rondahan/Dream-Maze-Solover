
import React, { useState, useEffect, useCallback, useRef } from 'react';
import { MazeState, CellType, Position, LatentVector, DreamPrediction, AgentLog } from './types';
import { MAZE_SIZE, TICK_RATE, LATENT_DIM } from './constants';
import { generateMaze } from './utils/mazeGenerator';
import { EnvironmentGenerator, EnvironmentType } from './utils/environmentGenerator';
import { getAgentInternalMonologue, predictNextState } from './services/geminiService';
import { PolicyNetwork } from './services/policyNetwork';
import { DQN } from './services/dqn';
import { ActorCritic, ExplorationType } from './services/actorCritic';
import { PPO } from './services/ppo';
import { A3C } from './services/a3c';
import { SAC } from './services/sac';
import { AStar } from './services/astar';
import { VAE } from './services/vae';
import { MDNRNN } from './services/mdnRnn';
import { TrainingSystem } from './services/trainingSystem';
import { TransferLearning } from './services/transferLearning';
import MazeBoard from './components/MazeBoard';
import LatentVisualizer from './components/LatentVisualizer';
import DreamState from './components/DreamState';

export type AlgorithmType = 'REINFORCE' | 'DQN' | 'ActorCritic' | 'PPO' | 'A3C' | 'SAC' | 'AStar' | 'EpsilonGreedy';

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
    useMDNRNN: true, // Use real MDN-RNN instead of simple prediction
    algorithm: 'REINFORCE' as AlgorithmType, // Selected algorithm
    useOptimalExploration: true, // Use optimal exploration for each algorithm (default)
    explorationType: 'softmax' as ExplorationType // For Actor-Critic manual override
  });

  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const policyNetworkRef = useRef<PolicyNetwork | null>(null);
  const dqnRef = useRef<DQN | null>(null);
  const actorCriticRef = useRef<ActorCritic | null>(null);
  const ppoRef = useRef<PPO | null>(null);
  const a3cRef = useRef<A3C | null>(null);
  const sacRef = useRef<SAC | null>(null);
  const astarRef = useRef<AStar | null>(null);
  const vaeRef = useRef<VAE | null>(null);
  const mdnRnnRef = useRef<MDNRNN | null>(null);
  const trainingSystemRef = useRef<TrainingSystem | null>(null);
  const knowledgeBaseRef = useRef<ReturnType<typeof TransferLearning.createKnowledgeBase> | null>(null);
  const trainingBufferRef = useRef<{
    states: number[][];
    actions: number[];
    rewards: number[];
    nextStates?: number[];
    dones?: boolean[];
    oldProbs?: number[]; // For PPO
  }>({ states: [], actions: [], rewards: [] });
  const lastActionRef = useRef<'up' | 'down' | 'left' | 'right' | null>(null);

  const addLog = useCallback((message: string, type: AgentLog['type']) => {
    const newLog: AgentLog = {
      id: Math.random().toString(36).substr(2, 9),
      timestamp: Date.now(),
      message,
      type
    };
    // Increase log buffer to show more information
    setLogs(prev => [newLog, ...prev].slice(0, 50));
    // Also log to console for debugging
    console.log(`[${type.toUpperCase()}] ${message}`);
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
      useMDNRNN: prev.useMDNRNN,
      algorithm: prev.algorithm,
      useOptimalExploration: prev.useOptimalExploration,
      explorationType: prev.explorationType
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

    // Use selected algorithm if neural network is enabled
    if (stats.useNeuralNetwork && stats.algorithm !== 'EpsilonGreedy') {
      try {
        let result: { pos: Position; actionIndex: number } | null = null;

        switch (stats.algorithm) {
          case 'REINFORCE':
            if (policyNetworkRef.current) {
              result = await policyNetworkRef.current.selectAction(
                m, possibleMoves, currentSteps, curiosityWeight, epsilon
              );
            }
            break;

          case 'DQN':
            if (dqnRef.current) {
              result = await dqnRef.current.selectAction(
                m, possibleMoves, currentSteps, curiosityWeight, epsilon
              );
            }
            break;

          case 'ActorCritic':
            if (actorCriticRef.current) {
              // Set exploration type based on settings
              if (stats.useOptimalExploration) {
                actorCriticRef.current.setExplorationType('softmax', 1.0);
              } else {
                actorCriticRef.current.setExplorationType(stats.explorationType, 1.0);
              }
              result = await actorCriticRef.current.selectAction(
                m, possibleMoves, currentSteps, curiosityWeight, epsilon
              );
            }
            break;

          case 'PPO':
            if (ppoRef.current) {
              // PPO always uses softmax (ignores epsilon parameter)
              result = await ppoRef.current.selectAction(
                m, possibleMoves, currentSteps, curiosityWeight, epsilon
              );
            }
            break;

          case 'A3C':
            if (a3cRef.current) {
              result = await a3cRef.current.selectAction(
                m, possibleMoves, currentSteps, curiosityWeight, epsilon
              );
            }
            break;

          case 'SAC':
            if (sacRef.current) {
              result = await sacRef.current.selectAction(
                m, possibleMoves, currentSteps, curiosityWeight, epsilon
              );
            }
            break;

          case 'AStar':
            if (astarRef.current) {
              result = await astarRef.current.selectAction(
                m, possibleMoves, currentSteps, curiosityWeight, epsilon
              );
            }
            break;
        }

        if (result) {
          const isNewArea = !history.some(hp => hp.x === result!.pos.x && hp.y === result!.pos.y);
          const mode = isNewArea ? 'explore' : 'exploit';
          
          return { 
            pos: result.pos, 
            mode, 
            epsilon,
            actionIndex: result.actionIndex
          };
        }
      } catch (error) {
        console.error(`Algorithm ${stats.algorithm} error, falling back to epsilon-greedy:`, error);
        // Fall through to epsilon-greedy
      }
    }

    // Fallback: Epsilon-Greedy Strategy (simple greedy)
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
  }, [stats.useNeuralNetwork, stats.algorithm, stats.useOptimalExploration, stats.explorationType]);

  const step = useCallback(async () => {
    if (!maze || stats.goalReached) return;

    // Log current state
    const currentDistance = Math.abs(maze.agentPos.x - maze.goalPos.x) + Math.abs(maze.agentPos.y - maze.goalPos.y);
    const visitedCount = maze.history.length;
    const totalCells = MAZE_SIZE * MAZE_SIZE;
    const visitedPercent = ((visitedCount / totalCells) * 100).toFixed(1);
    
    // Log state info every 5 steps to avoid spam
    if (stats.steps % 5 === 0) {
      addLog(`State: Agent[${maze.agentPos.x},${maze.agentPos.y}] → Goal[${maze.goalPos.x},${maze.goalPos.y}] | Distance: ${currentDistance} | Visited: ${visitedCount}/${totalCells} (${visitedPercent}%)`, "action");
    }

    // Get state features for neural network training (same for all algorithms)
    const stateFeatures = stats.useNeuralNetwork ? (() => {
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

    // Determine action direction for logging
    const dx = nextPos.x - maze.agentPos.x;
    const dy = nextPos.y - maze.agentPos.y;
    let actionDir: string = 'stay';
    if (dy > 0) actionDir = 'DOWN';
    else if (dy < 0) actionDir = 'UP';
    else if (dx < 0) actionDir = 'LEFT';
    else if (dx > 0) actionDir = 'RIGHT';

    // Log action selection details
    if (stats.useNeuralNetwork && stats.algorithm !== 'EpsilonGreedy') {
      addLog(`Action: ${actionDir} | Mode: ${mode.toUpperCase()} | Algorithm: ${stats.algorithm} | ε=${epsilon.toFixed(2)}`, "action");
    } else {
      addLog(`Action: ${actionDir} | Mode: ${mode.toUpperCase()} | Strategy: Epsilon-Greedy | ε=${epsilon.toFixed(2)}`, "action");
    }

    // Curiosity Reward Logic using the adjustable multiplier
    const stepCost = -1;
    const goalBonus = reachedGoal ? 100 : 0;
    const curiosityBonus = isNewArea ? curiosityMultiplier : 0;
    const totalRewardIncrement = stepCost + goalBonus + curiosityBonus;

    // Log reward breakdown
    if (reachedGoal) {
      addLog(`🎯 GOAL REACHED! Reward: ${stepCost} (step) + ${goalBonus} (goal) = ${totalRewardIncrement}`, "action");
    } else if (isNewArea) {
      addLog(`Reward: ${stepCost} (step) + ${curiosityBonus} (curiosity) = ${totalRewardIncrement}`, "action");
    } else {
      addLog(`Reward: ${stepCost} (step cost)`, "action");
    }

    // Store experience for training (if using neural network)
    if (stats.useNeuralNetwork && stateFeatures && actionIndex !== undefined && actionIndex >= 0) {
      trainingBufferRef.current.states.push(stateFeatures);
      trainingBufferRef.current.actions.push(actionIndex);
      trainingBufferRef.current.rewards.push(totalRewardIncrement);
      
      // Compute next state features for algorithms that need it (DQN, Actor-Critic, PPO)
      if (stats.algorithm === 'DQN' || stats.algorithm === 'ActorCritic' || stats.algorithm === 'PPO' || stats.algorithm === 'A3C' || stats.algorithm === 'SAC') {
        const nextStateFeatures = stateFeatures; // Simplified - would compute actual next state
        if (!trainingBufferRef.current.nextStates) {
          trainingBufferRef.current.nextStates = [];
          trainingBufferRef.current.dones = [];
        }
        trainingBufferRef.current.nextStates.push(nextStateFeatures);
        trainingBufferRef.current.dones!.push(reachedGoal);
        
        // For PPO, also store old probability
        if (stats.algorithm === 'PPO') {
          if (!trainingBufferRef.current.oldProbs) {
            trainingBufferRef.current.oldProbs = [];
          }
          // Get old probability (simplified - would get from previous prediction)
          trainingBufferRef.current.oldProbs.push(0.25); // Default uniform
        }
      }
      
      // Also add to training system for experience replay
      if (trainingSystemRef.current) {
        const nextStateFeatures = stateFeatures; // Simplified
        trainingSystemRef.current.addExperience({
          state: stateFeatures,
          action: actionIndex,
          reward: totalRewardIncrement,
          nextState: nextStateFeatures,
          done: reachedGoal
        });
      }
      
      const newBufferSize = trainingBufferRef.current.states.length;
      setStats(prev => ({ ...prev, trainingExperiences: newBufferSize }));
      
      // Log buffer status every 10 experiences
      if (newBufferSize % 10 === 0) {
        addLog(`Training buffer: ${newBufferSize} experiences collected`, "action");
      }
    }

    // V-M loop: Use VAE if enabled, otherwise use random
    // Optimize: Only encode every few steps to reduce computation
    let newLatent: LatentVector;
    if (stats.useVAE && vaeRef.current && maze && stats.steps % 3 === 0) {
      try {
        newLatent = await vaeRef.current.encode(maze);
        const latentStr = newLatent.vector.slice(0, 3).map(v => v.toFixed(2)).join(', ');
        addLog(`VAE: Encoded state → latent[${latentStr}...] (dim=${LATENT_DIM})`, "vision");
      } catch (error) {
        console.error('VAE encoding error:', error);
        addLog(`VAE: Encoding failed, using fallback`, "error");
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
      if (stats.steps % 3 !== 0 && stats.useVAE) {
        addLog(`VAE: Reusing previous latent (encoding every 3 steps)`, "vision");
      }
    }
    setLatent(newLatent);

    // Determine action direction for MDN-RNN (already calculated above)
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
            addLog(`MDN-RNN: Predicting future sequence (5 steps ahead)`, "dream");
            // Predict sequence using MDN-RNN
            const actions: ('up' | 'down' | 'left' | 'right')[] = [
              lastActionRef.current, lastActionRef.current, lastActionRef.current, 
              lastActionRef.current, lastActionRef.current
            ];
            d = await mdnRnnRef.current.predictSequence(newLatent.vector, actions, maze);
            const predictedEnd = d.steps.length > 0 ? d.steps[d.steps.length - 1] : null;
            if (predictedEnd) {
              addLog(`MDN-RNN: Dream sequence generated (${d.steps.length} steps, confidence: ${(d.confidence * 100).toFixed(1)}%, end: [${predictedEnd.x},${predictedEnd.y}])`, "dream");
            } else {
              addLog(`MDN-RNN: Dream sequence generated (${d.steps.length} steps, confidence: ${(d.confidence * 100).toFixed(1)}%)`, "dream");
            }
          } catch (error) {
            console.error('MDN-RNN prediction error:', error);
            addLog(`MDN-RNN: Prediction failed, using fallback (non-critical - system continues normally)`, "error");
            // Fallback to simple prediction
            d = await predictNextState(maze);
          }
        } else {
          d = await predictNextState(maze);
          addLog(`Dream: Using simple prediction (MDN-RNN disabled)`, "dream");
        }
        setDream(d);
        // Only update monologue occasionally to reduce API calls
        if (stats.steps % 20 === 0) {
          try {
            addLog(`Generating internal monologue...`, "dream");
            const m = await getAgentInternalMonologue(maze);
            setMonologue(m);
            addLog(`Monologue updated`, "dream");
          } catch (error) {
            console.error('Monologue error:', error);
            addLog(`Monologue generation failed`, "error");
          }
        }
        if (mode === 'explore') {
          addLog(`Internal State: Novelty seeking (ε=${epsilon.toFixed(2)}, Curiosity x${curiosityMultiplier}).`, "dream");
        }
      }).catch(error => {
        console.error('Dream analysis error:', error);
        addLog(`Dream analysis error: ${error}`, "error");
      });
    }

    if (isNewArea && !reachedGoal) {
      const distanceToGoal = Math.abs(nextPos.x - maze.goalPos.x) + Math.abs(nextPos.y - maze.goalPos.y);
      addLog(`Sensory Input: Novel area detected at [${nextPos.x},${nextPos.y}]. Distance to goal: ${distanceToGoal}. Reward: +${curiosityMultiplier}`, "vision");
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

    // Log mode changes
    const prevMode = stats.currentMode;
    setStats(prev => ({
      ...prev,
      steps: prev.steps + 1,
      reward: prev.reward + totalRewardIncrement,
      curiosityScore: prev.curiosityScore + curiosityBonus,
      goalReached: reachedGoal,
      currentMode: mode,
      epsilon: epsilon
    }));
    
    // Log mode transition
    if (prevMode !== mode && prevMode !== 'initializing') {
      addLog(`Mode transition: ${prevMode.toUpperCase()} → ${mode.toUpperCase()}`, "action");
    }

    if (reachedGoal) {
      addLog("Goal state achieved. Terminating episode.", "vision");
      setIsRunning(false);
      
      // Train neural network at end of episode
      if (stats.useNeuralNetwork && trainingBufferRef.current.states.length > 0) {
        try {
          const bufferSize = trainingBufferRef.current.states.length;
          const totalReward = trainingBufferRef.current.rewards.reduce((a, b) => a + b, 0);
          addLog(`Training ${stats.algorithm} network with ${bufferSize} experiences (total reward: ${totalReward.toFixed(1)})...`, "action");
          
          switch (stats.algorithm) {
            case 'REINFORCE':
              if (policyNetworkRef.current) {
                if (trainingSystemRef.current) {
                  const statsBefore = trainingSystemRef.current.getStats();
                  await trainingSystemRef.current.trainPolicyNetwork(policyNetworkRef.current, 10);
                  const statsAfter = trainingSystemRef.current.getStats();
                  addLog(`REINFORCE: Trained with experience replay (buffer: ${statsAfter.bufferSize} experiences, ${statsAfter.trainingSteps} training steps)`, "action");
                } else {
                  await policyNetworkRef.current.train(
                    trainingBufferRef.current.states,
                    trainingBufferRef.current.actions,
                    trainingBufferRef.current.rewards
                  );
                  addLog(`REINFORCE: Trained on ${bufferSize} experiences (episode end)`, "action");
                }
              }
              break;

            case 'DQN':
              if (dqnRef.current && trainingBufferRef.current.nextStates && trainingBufferRef.current.dones) {
                const experiences = trainingBufferRef.current.states.map((state, i) => ({
                  state,
                  action: trainingBufferRef.current.actions[i],
                  reward: trainingBufferRef.current.rewards[i],
                  nextState: trainingBufferRef.current.nextStates![i],
                  done: trainingBufferRef.current.dones![i]
                }));
                await dqnRef.current.train(experiences);
                addLog(`DQN: Trained on ${bufferSize} experiences (Q-learning update)`, "action");
              }
              break;

            case 'ActorCritic':
              if (actorCriticRef.current && trainingBufferRef.current.nextStates && trainingBufferRef.current.dones) {
                const experiences = trainingBufferRef.current.states.map((state, i) => ({
                  state,
                  action: trainingBufferRef.current.actions[i],
                  reward: trainingBufferRef.current.rewards[i],
                  nextState: trainingBufferRef.current.nextStates![i],
                  done: trainingBufferRef.current.dones![i]
                }));
                await actorCriticRef.current.train(experiences);
                addLog(`Actor-Critic: Trained on ${bufferSize} experiences (policy + value update)`, "action");
              }
              break;

            case 'PPO':
              if (ppoRef.current && trainingBufferRef.current.nextStates && trainingBufferRef.current.dones && trainingBufferRef.current.oldProbs) {
                const experiences = trainingBufferRef.current.states.map((state, i) => ({
                  state,
                  action: trainingBufferRef.current.actions[i],
                  reward: trainingBufferRef.current.rewards[i],
                  nextState: trainingBufferRef.current.nextStates![i],
                  done: trainingBufferRef.current.dones![i],
                  oldProb: trainingBufferRef.current.oldProbs![i]
                }));
                await ppoRef.current.train(experiences);
                addLog(`PPO: Trained on ${bufferSize} experiences (clipped objective)`, "action");
              }
              break;

            case 'A3C':
              if (a3cRef.current && trainingBufferRef.current.nextStates && trainingBufferRef.current.dones) {
                await a3cRef.current.train(
                  trainingBufferRef.current.states,
                  trainingBufferRef.current.actions,
                  trainingBufferRef.current.rewards,
                  trainingBufferRef.current.nextStates,
                  trainingBufferRef.current.dones
                );
                addLog(`A3C: Trained on ${bufferSize} experiences (n-step returns)`, "action");
              }
              break;

            case 'SAC':
              if (sacRef.current && trainingBufferRef.current.nextStates && trainingBufferRef.current.dones) {
                await sacRef.current.train(
                  trainingBufferRef.current.states,
                  trainingBufferRef.current.actions,
                  trainingBufferRef.current.rewards,
                  trainingBufferRef.current.nextStates,
                  trainingBufferRef.current.dones
                );
                addLog(`SAC: Trained on ${bufferSize} experiences (soft actor-critic)`, "action");
              }
              break;

            case 'AStar':
              // A* doesn't learn, but we keep the interface consistent
              addLog(`A*: No training needed (deterministic pathfinding)`, "action");
              break;
          }
          
          // Clear buffer
          trainingBufferRef.current = { states: [], actions: [], rewards: [] };
          setStats(prev => ({ ...prev, trainingExperiences: 0 }));
          addLog(`Training complete. Buffer cleared.`, "action");
        } catch (error) {
          console.error('Training error:', error);
          addLog(`Training error: ${error}`, "error");
        }
      }
    }

    // Periodic training (every 20 steps)
    if (stats.useNeuralNetwork && trainingBufferRef.current.states.length >= 20 && stats.steps % 20 === 0) {
      try {
        const recentStates = trainingBufferRef.current.states.slice(-20);
        const recentActions = trainingBufferRef.current.actions.slice(-20);
        const recentRewards = trainingBufferRef.current.rewards.slice(-20);
        const recentRewardSum = recentRewards.reduce((a, b) => a + b, 0);
        
        addLog(`Periodic training: ${stats.algorithm} on last 20 steps (reward: ${recentRewardSum.toFixed(1)})`, "action");
        
        switch (stats.algorithm) {
          case 'REINFORCE':
            if (policyNetworkRef.current) {
              if (trainingSystemRef.current) {
                await trainingSystemRef.current.trainPolicyNetwork(policyNetworkRef.current, 3);
                addLog(`REINFORCE: Periodic training complete (3 epochs)`, "action");
              } else {
                await policyNetworkRef.current.train(recentStates, recentActions, recentRewards);
                addLog(`REINFORCE: Periodic training complete`, "action");
              }
            }
            break;

          case 'DQN':
          case 'ActorCritic':
          case 'PPO':
          case 'A3C':
          case 'SAC':
            // These algorithms train better with full episodes, skip periodic training
            addLog(`${stats.algorithm}: Skipping periodic training (trains at episode end)`, "action");
            break;

          case 'AStar':
            // A* doesn't learn
            addLog(`A*: No training needed (deterministic pathfinding)`, "action");
            break;
        }
        
        // Keep buffer size manageable
        if (trainingBufferRef.current.states.length > 100) {
          const keepSize = 50;
          const removed = trainingBufferRef.current.states.length - keepSize;
          trainingBufferRef.current = {
            states: trainingBufferRef.current.states.slice(-keepSize),
            actions: trainingBufferRef.current.actions.slice(-keepSize),
            rewards: trainingBufferRef.current.rewards.slice(-keepSize),
            nextStates: trainingBufferRef.current.nextStates?.slice(-keepSize),
            dones: trainingBufferRef.current.dones?.slice(-keepSize),
            oldProbs: trainingBufferRef.current.oldProbs?.slice(-keepSize)
          };
          addLog(`Buffer trimmed: removed ${removed} old experiences (kept ${keepSize})`, "action");
        }
      } catch (error) {
        console.error('Periodic training error:', error);
        addLog(`Periodic training error: ${error}`, "error");
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
  // Use requestIdleCallback for better performance and lazy loading
  useEffect(() => {
    let cancelled = false;
    
    const initNetworks = async () => {
      setIsInitializing(true);
      
      try {
        // Initialize lightweight systems first (no TensorFlow)
        if (!cancelled) {
          trainingSystemRef.current = new TrainingSystem(10000, 32);
          addLog("Training System initialized. Experience replay enabled.", "action");
        }
        
        if (!cancelled) {
          knowledgeBaseRef.current = TransferLearning.createKnowledgeBase();
        }
        
        // Mark UI as ready immediately - don't wait for TensorFlow
        // This allows the user to see the interface and start interacting
        if (!cancelled) {
          setIsInitializing(false);
          addLog("UI ready. Loading neural networks in background...", "action");
        }
        
        // Initialize algorithms lazily - only load when needed
        // This ensures fast page load - algorithms load on-demand
        const initAlgorithm = async (algorithm: AlgorithmType) => {
          try {
            switch (algorithm) {
              case 'REINFORCE':
                if (!policyNetworkRef.current) {
                  policyNetworkRef.current = new PolicyNetwork();
                  await policyNetworkRef.current.initialize();
                  if (!cancelled) {
                    addLog("REINFORCE (Policy Network) initialized.", "action");
                  }
                }
                break;
              case 'DQN':
                if (!dqnRef.current) {
                  dqnRef.current = new DQN();
                  await dqnRef.current.initialize();
                  if (!cancelled) {
                    addLog("DQN initialized.", "action");
                  }
                }
                break;
              case 'ActorCritic':
                if (!actorCriticRef.current) {
                  const explorationType = stats.useOptimalExploration ? 'softmax' : stats.explorationType;
                  actorCriticRef.current = new ActorCritic(explorationType, 1.0);
                  await actorCriticRef.current.initialize();
                  if (!cancelled) {
                    addLog(`Actor-Critic initialized (${explorationType}).`, "action");
                  }
                }
                break;
              case 'PPO':
                if (!ppoRef.current) {
                  ppoRef.current = new PPO(1.0, 0.2);
                  await ppoRef.current.initialize();
                  if (!cancelled) {
                    addLog("PPO initialized (softmax).", "action");
                  }
                }
                break;
              case 'A3C':
                if (!a3cRef.current) {
                  a3cRef.current = new A3C(5, 0.01);
                  await a3cRef.current.initialize();
                  if (!cancelled) {
                    addLog("A3C initialized.", "action");
                  }
                }
                break;
              case 'SAC':
                if (!sacRef.current) {
                  sacRef.current = new SAC(0.2, 0.005);
                  await sacRef.current.initialize();
                  if (!cancelled) {
                    addLog("SAC initialized.", "action");
                  }
                }
                break;
              case 'AStar':
                if (!astarRef.current) {
                  astarRef.current = new AStar();
                  await astarRef.current.initialize();
                  if (!cancelled) {
                    addLog("A* Pathfinding initialized.", "action");
                  }
                }
                break;
            }
          } catch (error) {
            console.warn(`${algorithm} initialization failed:`, error);
            if (!cancelled) {
              addLog(`${algorithm} initialization failed.`, "action");
            }
          }
        };
        
        // Load default algorithm (REINFORCE) after a short delay
        setTimeout(() => {
          if (!cancelled) {
            initAlgorithm('REINFORCE');
          }
        }, 200);
        
        // Initialize VAE lazily - only if enabled
        // Load in background after UI is ready, with timeout protection
        if (stats.useVAE && !cancelled) {
          const initVAE = async () => {
            try {
              vaeRef.current = new VAE();
              // Add timeout protection
              const initPromise = vaeRef.current.initialize();
              const timeoutPromise = new Promise((_, reject) => 
                setTimeout(() => reject(new Error('VAE initialization timeout')), 15000)
              );
              
              await Promise.race([initPromise, timeoutPromise]);
              
              if (!cancelled) {
                addLog("VAE (Vision) initialized. Encoding maze states.", "vision");
              }
            } catch (error) {
              console.warn('VAE initialization failed or timed out:', error);
              if (!cancelled) {
                addLog("VAE initialization skipped (will use fallback).", "vision");
              }
            }
          };
          
          // Load after a longer delay to let Policy Network load first
          setTimeout(() => {
            if (!cancelled) {
              initVAE();
            }
          }, 1000);
        }
        
        // Initialize MDN-RNN lazily - only if enabled
        // Load in background after VAE, with timeout protection
        if (stats.useMDNRNN && !cancelled) {
          const initMDNRNN = async () => {
            try {
              mdnRnnRef.current = new MDNRNN();
              // Add timeout protection
              const initPromise = mdnRnnRef.current.initialize();
              const timeoutPromise = new Promise((_, reject) => 
                setTimeout(() => reject(new Error('MDN-RNN initialization timeout')), 20000)
              );
              
              await Promise.race([initPromise, timeoutPromise]);
              
              if (!cancelled) {
                addLog("MDN-RNN (Memory) initialized. Ready for dreaming.", "dream");
              }
            } catch (error) {
              console.warn('MDN-RNN initialization failed or timed out:', error);
              if (!cancelled) {
                addLog("MDN-RNN initialization skipped (will use fallback).", "dream");
              }
            }
          };
          
          // Load after VAE starts loading
          setTimeout(() => {
            if (!cancelled) {
              initMDNRNN();
            }
          }, 2000);
        }
      } catch (error) {
        console.error('Network initialization error:', error);
        if (!cancelled) {
          setIsInitializing(false);
        }
      }
    };
    
    // Use requestIdleCallback for non-blocking initialization
    // But with a shorter timeout so UI appears faster
    if ('requestIdleCallback' in window) {
      (window as any).requestIdleCallback(() => {
        initNetworks();
      }, { timeout: 100 });
    } else {
      // Fallback for browsers without requestIdleCallback
      setTimeout(initNetworks, 50);
    }

    return () => {
      cancelled = true;
      if (policyNetworkRef.current) {
        policyNetworkRef.current.dispose();
      }
      if (dqnRef.current) {
        dqnRef.current.dispose();
      }
      if (actorCriticRef.current) {
        actorCriticRef.current.dispose();
      }
      if (ppoRef.current) {
        ppoRef.current.dispose();
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
  }, [addLog, stats.useVAE, stats.useMDNRNN, stats.algorithm, stats.useOptimalExploration, stats.explorationType]);

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
                trainingExperiences: 0,
                algorithm: !prev.useNeuralNetwork ? 'REINFORCE' : 'EpsilonGreedy'
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

          {/* Algorithm Selector - Only show when NN is ON */}
          {stats.useNeuralNetwork && (
            <select
              value={stats.algorithm}
              onChange={async (e) => {
                const newAlgorithm = e.target.value as AlgorithmType;
                setStats(prev => ({ 
                  ...prev, 
                  algorithm: newAlgorithm,
                  trainingExperiences: 0
                }));
                trainingBufferRef.current = { states: [], actions: [], rewards: [] };
                
                // Lazy load the selected algorithm
                if (newAlgorithm !== 'EpsilonGreedy') {
                  try {
                    switch (newAlgorithm) {
                      case 'REINFORCE':
                        if (!policyNetworkRef.current) {
                          policyNetworkRef.current = new PolicyNetwork();
                          await policyNetworkRef.current.initialize();
                          addLog("REINFORCE loaded.", "action");
                        }
                        break;
                      case 'DQN':
                        if (!dqnRef.current) {
                          dqnRef.current = new DQN();
                          await dqnRef.current.initialize();
                          addLog("DQN loaded.", "action");
                        }
                        break;
                      case 'ActorCritic':
                        if (!actorCriticRef.current) {
                          const explorationType = stats.useOptimalExploration ? 'softmax' : stats.explorationType;
                          actorCriticRef.current = new ActorCritic(explorationType, 1.0);
                          await actorCriticRef.current.initialize();
                          addLog(`Actor-Critic loaded (${explorationType}).`, "action");
                        }
                        break;
                      case 'PPO':
                        if (!ppoRef.current) {
                          ppoRef.current = new PPO(1.0, 0.2);
                          await ppoRef.current.initialize();
                          addLog("PPO loaded (softmax).", "action");
                        }
                        break;
                      case 'A3C':
                        if (!a3cRef.current) {
                          a3cRef.current = new A3C(5, 0.01);
                          await a3cRef.current.initialize();
                          addLog("A3C loaded.", "action");
                        }
                        break;
                      case 'SAC':
                        if (!sacRef.current) {
                          sacRef.current = new SAC(0.2, 0.005);
                          await sacRef.current.initialize();
                          addLog("SAC loaded.", "action");
                        }
                        break;
                      case 'AStar':
                        if (!astarRef.current) {
                          astarRef.current = new AStar();
                          await astarRef.current.initialize();
                          addLog("A* Pathfinding loaded.", "action");
                        }
                        break;
                    }
                  } catch (error) {
                    console.error(`Failed to load ${newAlgorithm}:`, error);
                    addLog(`Failed to load ${newAlgorithm}.`, "error");
                  }
                }
              }}
              className="px-4 py-3 bg-slate-800 border border-slate-700 rounded-full font-bold text-sm hover:bg-slate-700 transition-all"
            >
              <option value="REINFORCE">REINFORCE</option>
              <option value="DQN">DQN</option>
              <option value="ActorCritic">Actor-Critic</option>
              <option value="PPO">PPO</option>
              <option value="A3C">A3C</option>
              <option value="SAC">SAC</option>
              <option value="AStar">A* Pathfinding</option>
            </select>
          )}

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
              addLog("Attempting to save knowledge...", "action");
              
              // Check what's available
              const missing: string[] = [];
              if (!knowledgeBaseRef.current) missing.push("Knowledge Base");
              if (!policyNetworkRef.current) missing.push("Policy Network");
              if (!vaeRef.current) missing.push("VAE");
              if (!mdnRnnRef.current) missing.push("MDN-RNN");
              
              if (missing.length > 0) {
                addLog(`Cannot save: Missing components: ${missing.join(", ")}. Make sure NN is ON and networks are initialized.`, "error");
                return;
              }
              
              if (!stats.useNeuralNetwork) {
                addLog("Cannot save: Neural Network is OFF. Turn NN ON to enable knowledge saving.", "error");
                return;
              }
              
              try {
                addLog("Saving Policy Network weights...", "action");
                await knowledgeBaseRef.current!.savePolicy(policyNetworkRef.current!);
                addLog("✓ Policy Network saved", "action");
                
                addLog("Saving VAE weights...", "action");
                await knowledgeBaseRef.current!.saveVAE(vaeRef.current!);
                addLog("✓ VAE (Vision) saved", "action");
                
                addLog("Saving MDN-RNN weights...", "action");
                await knowledgeBaseRef.current!.saveRNN(mdnRnnRef.current!);
                addLog("✓ MDN-RNN (Memory) saved", "action");
                
                addLog("✅ Knowledge saved successfully! Ready for transfer learning to new environments.", "action");
              } catch (error) {
                console.error('Save knowledge error:', error);
                addLog(`❌ Error saving knowledge: ${error}`, "error");
              }
            }}
            className="px-6 py-3 bg-emerald-600 text-white rounded-full font-bold hover:bg-emerald-500 transition-all flex items-center gap-3 disabled:opacity-50 disabled:cursor-not-allowed"
            title="Save current knowledge for transfer learning (requires NN ON and all networks initialized)"
            disabled={!stats.useNeuralNetwork || !knowledgeBaseRef.current}
          >
            <i className="fa-solid fa-download"></i>
            SAVE KNOWLEDGE
          </button>
        </div>
      </div>

      {/* Algorithm Settings Panel - Only show when NN is ON */}
      {stats.useNeuralNetwork && stats.algorithm !== 'EpsilonGreedy' && (
        <div className="w-full max-w-6xl mb-6 bg-slate-900/50 border border-slate-700 rounded-xl p-4">
          <div className="flex flex-wrap items-center gap-4">
            <div className="flex items-center gap-2">
              <span className="text-xs font-bold text-slate-400 uppercase">Algorithm:</span>
              <span className="text-sm font-mono text-purple-400">{stats.algorithm}</span>
            </div>

            {/* Exploration Type Toggle - Only for Actor-Critic */}
            {stats.algorithm === 'ActorCritic' && (
              <>
                <div className="flex items-center gap-2">
                  <span className="text-xs font-bold text-slate-400 uppercase">Exploration:</span>
                  <button
                    onClick={() => {
                      setStats(prev => ({ 
                        ...prev, 
                        useOptimalExploration: !prev.useOptimalExploration
                      }));
                      if (actorCriticRef.current) {
                        const explorationType = !stats.useOptimalExploration ? 'softmax' : stats.explorationType;
                        actorCriticRef.current.setExplorationType(explorationType, 1.0);
                      }
                    }}
                    className={`px-3 py-1 rounded-full text-xs font-bold transition-all ${
                      stats.useOptimalExploration
                        ? 'bg-emerald-600 text-white'
                        : 'bg-slate-700 text-slate-300'
                    }`}
                    title={stats.useOptimalExploration ? "Using optimal exploration (softmax)" : "Using manual exploration"}
                  >
                    {stats.useOptimalExploration ? 'Optimal (Softmax)' : 'Manual'}
                  </button>
                </div>

                {!stats.useOptimalExploration && (
                  <select
                    value={stats.explorationType}
                    onChange={(e) => {
                      const newType = e.target.value as ExplorationType;
                      setStats(prev => ({ ...prev, explorationType: newType }));
                      if (actorCriticRef.current) {
                        actorCriticRef.current.setExplorationType(newType, 1.0);
                      }
                    }}
                    className="px-3 py-1 bg-slate-800 border border-slate-700 rounded-full font-bold text-xs hover:bg-slate-700 transition-all"
                  >
                    <option value="softmax">Softmax</option>
                    <option value="epsilon-greedy">Epsilon-Greedy</option>
                  </select>
                )}
              </>
            )}

            {/* Info for other algorithms */}
            {stats.algorithm === 'REINFORCE' && (
              <span className="text-xs text-slate-500 italic">Using Epsilon-Greedy exploration</span>
            )}
            {stats.algorithm === 'DQN' && (
              <span className="text-xs text-slate-500 italic">Using Epsilon-Greedy exploration</span>
            )}
            {stats.algorithm === 'PPO' && (
              <span className="text-xs text-slate-500 italic">Using Softmax exploration (required)</span>
            )}
            {stats.algorithm === 'A3C' && (
              <span className="text-xs text-slate-500 italic">Using Softmax exploration with n-step returns</span>
            )}
            {stats.algorithm === 'SAC' && (
              <span className="text-xs text-slate-500 italic">Using Softmax exploration with entropy regularization</span>
            )}
            {stats.algorithm === 'AStar' && (
              <span className="text-xs text-slate-500 italic">Heuristic pathfinding (deterministic)</span>
            )}
          </div>
        </div>
      )}

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
                <span className="text-[10px] text-purple-400 font-bold uppercase">
                  {stats.algorithm} Network
                </span>
                <span className="text-[10px] text-purple-300 font-mono">
                  {stats.trainingExperiences} experiences
                </span>
              </div>
              <div className="text-[10px] text-purple-400/70 italic">
                Learning from experience... {stats.algorithm} improves over time.
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
            <div className="flex-1 space-y-2 overflow-y-auto max-h-[500px] pr-2 custom-scrollbar">
              {logs.map((log) => (
                <div key={log.id} className="text-[10px] font-mono leading-relaxed border-l-2 pl-3 py-1.5 hover:bg-slate-800/50 transition-colors ${
                  log.type === 'dream' ? 'border-purple-500/30' : 
                  log.type === 'vision' ? 'border-emerald-500/30' : 
                  log.type === 'error' ? 'border-red-500/30' : 'border-blue-500/30'
                }">
                  <span className="text-slate-500 block mb-0.5 text-[9px]">
                    [{new Date(log.timestamp).toLocaleTimeString()}] 
                    <span className={`ml-2 uppercase font-bold ${
                      log.type === 'dream' ? 'text-purple-400' : 
                      log.type === 'vision' ? 'text-emerald-400' : 
                      log.type === 'error' ? 'text-red-400' : 'text-blue-400'
                    }`}>{log.type}</span>
                  </span>
                  <span className={`text-[11px] ${
                    log.type === 'error' ? 'text-red-300' : 'text-slate-200'
                  }`}>{log.message}</span>
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
