
import { CellType, Position } from '../types';
import { MAZE_SIZE } from '../constants';

export const generateMaze = (size: number): { grid: CellType[][]; start: Position; goal: Position } => {
  const grid: CellType[][] = Array(size).fill(0).map(() => Array(size).fill(CellType.WALL));

  const start: Position = { x: 1, y: 1 };
  const goal: Position = { x: size - 2, y: size - 2 };

  const stack: Position[] = [start];
  grid[start.y][start.x] = CellType.PATH;

  const getNeighbors = (p: Position) => {
    const neighbors: Position[] = [];
    const dirs = [[0, 2], [0, -2], [2, 0], [-2, 0]];
    for (const [dx, dy] of dirs) {
      const nx = p.x + dx;
      const ny = p.y + dy;
      if (nx > 0 && nx < size - 1 && ny > 0 && ny < size - 1 && grid[ny][nx] === CellType.WALL) {
        neighbors.push({ x: nx, y: ny });
      }
    }
    return neighbors;
  };

  while (stack.length > 0) {
    const current = stack[stack.length - 1];
    const neighbors = getNeighbors(current);

    if (neighbors.length > 0) {
      const next = neighbors[Math.floor(Math.random() * neighbors.length)];
      grid[next.y][next.x] = CellType.PATH;
      grid[current.y + (next.y - current.y) / 2][current.x + (next.x - current.x) / 2] = CellType.PATH;
      stack.push(next);
    } else {
      stack.pop();
    }
  }

  // Ensure goal is accessible
  grid[goal.y][goal.x] = CellType.GOAL;
  grid[goal.y - 1][goal.x] = CellType.PATH; // Basic path guarantee
  
  return { grid, start, goal };
};
