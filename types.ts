
export enum CellType {
  WALL = 'WALL',
  PATH = 'PATH',
  START = 'START',
  GOAL = 'GOAL',
  AGENT = 'AGENT'
}

export interface Position {
  x: number;
  y: number;
}

export interface MazeState {
  grid: CellType[][];
  agentPos: Position;
  goalPos: Position;
  history: Position[];
}

export interface LatentVector {
  vector: number[];
  reconstruction: number[][]; // Visual representation of what 'V' sees
}

export interface DreamPrediction {
  steps: Position[];
  confidence: number;
  description: string;
}

export interface AgentLog {
  id: string;
  timestamp: number;
  message: string;
  type: 'action' | 'dream' | 'vision' | 'error';
}
