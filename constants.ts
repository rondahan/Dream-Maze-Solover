
// Fix: Import CellType from types to resolve usage in COLORS constant
import { CellType } from './types';

export const MAZE_SIZE = 15;
export const CELL_SIZE = 30;
export const LATENT_DIM = 8;
export const TICK_RATE = 300; // ms

export const COLORS = {
  [CellType.WALL]: '#1e293b', // slate-800
  [CellType.PATH]: '#0f172a', // slate-900
  [CellType.START]: '#10b981', // emerald-500
  [CellType.GOAL]: '#f59e0b', // amber-500
  [CellType.AGENT]: '#3b82f6', // blue-500
};