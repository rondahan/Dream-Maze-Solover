import { CellType, Position } from '../types';
import { MAZE_SIZE } from '../constants';

export enum EnvironmentType {
  MAZE = 'maze',
  OPEN_FIELD = 'open_field',
  CORRIDOR = 'corridor',
  SPIRAL = 'spiral',
  GRID = 'grid'
}

/**
 * Generate different types of environments
 */
export class EnvironmentGenerator {
  /**
   * Generate standard maze (recursive backtracking)
   */
  static generateMaze(size: number): { grid: CellType[][]; start: Position; goal: Position } {
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

    grid[goal.y][goal.x] = CellType.GOAL;
    grid[goal.y - 1][goal.x] = CellType.PATH;
    
    return { grid, start, goal };
  }

  /**
   * Generate open field (mostly paths, few walls)
   */
  static generateOpenField(size: number): { grid: CellType[][]; start: Position; goal: Position } {
    const grid: CellType[][] = Array(size).fill(0).map(() => Array(size).fill(CellType.PATH));
    
    // Add some random walls (20% density)
    for (let y = 1; y < size - 1; y++) {
      for (let x = 1; x < size - 1; x++) {
        if (Math.random() < 0.2) {
          grid[y][x] = CellType.WALL;
        }
      }
    }
    
    const start: Position = { x: 1, y: 1 };
    const goal: Position = { x: size - 2, y: size - 2 };
    
    grid[start.y][start.x] = CellType.PATH;
    grid[goal.y][goal.x] = CellType.GOAL;
    
    return { grid, start, goal };
  }

  /**
   * Generate corridor maze (long winding paths)
   */
  static generateCorridor(size: number): { grid: CellType[][]; start: Position; goal: Position } {
    const grid: CellType[][] = Array(size).fill(0).map(() => Array(size).fill(CellType.WALL));
    
    // Create a winding corridor
    let x = 1, y = 1;
    const path: Position[] = [{ x, y }];
    grid[y][x] = CellType.PATH;
    
    const dirs = [[0, 1], [1, 0], [0, -1], [-1, 0]];
    let dirIdx = 0;
    
    while (x < size - 2 || y < size - 2) {
      const [dx, dy] = dirs[dirIdx];
      const nx = x + dx;
      const ny = y + dy;
      
      if (nx > 0 && nx < size - 1 && ny > 0 && ny < size - 1) {
        x = nx;
        y = ny;
        grid[y][x] = CellType.PATH;
        path.push({ x, y });
      } else {
        dirIdx = (dirIdx + 1) % dirs.length;
      }
      
      // Occasionally change direction
      if (Math.random() < 0.3) {
        dirIdx = (dirIdx + 1) % dirs.length;
      }
    }
    
    const start: Position = { x: 1, y: 1 };
    const goal: Position = { x: size - 2, y: size - 2 };
    grid[goal.y][goal.x] = CellType.GOAL;
    
    return { grid, start, goal };
  }

  /**
   * Generate spiral maze
   */
  static generateSpiral(size: number): { grid: CellType[][]; start: Position; goal: Position } {
    const grid: CellType[][] = Array(size).fill(0).map(() => Array(size).fill(CellType.WALL));
    
    const center = Math.floor(size / 2);
    let radius = 1;
    let angle = 0;
    
    while (radius < size / 2) {
      const x = Math.floor(center + radius * Math.cos(angle));
      const y = Math.floor(center + radius * Math.sin(angle));
      
      if (x > 0 && x < size - 1 && y > 0 && y < size - 1) {
        grid[y][x] = CellType.PATH;
      }
      
      angle += 0.1;
      if (angle > Math.PI * 2) {
        angle = 0;
        radius += 0.5;
      }
    }
    
    // Connect center to start
    for (let i = 1; i <= center; i++) {
      grid[center][i] = CellType.PATH;
    }
    
    const start: Position = { x: 1, y: center };
    const goal: Position = { x: size - 2, y: size - 2 };
    grid[start.y][start.x] = CellType.PATH;
    grid[goal.y][goal.x] = CellType.GOAL;
    
    return { grid, start, goal };
  }

  /**
   * Generate grid pattern (checkerboard-like paths)
   */
  static generateGrid(size: number): { grid: CellType[][]; start: Position; goal: Position } {
    const grid: CellType[][] = Array(size).fill(0).map(() => Array(size).fill(CellType.WALL));
    
    // Create grid pattern
    for (let y = 1; y < size - 1; y += 2) {
      for (let x = 1; x < size - 1; x += 2) {
        grid[y][x] = CellType.PATH;
        if (x + 1 < size - 1) grid[y][x + 1] = CellType.PATH;
        if (y + 1 < size - 1) grid[y + 1][x] = CellType.PATH;
      }
    }
    
    const start: Position = { x: 1, y: 1 };
    const goal: Position = { x: size - 2, y: size - 2 };
    grid[start.y][start.x] = CellType.PATH;
    grid[goal.y][goal.x] = CellType.GOAL;
    
    return { grid, start, goal };
  }

  /**
   * Generate environment by type
   */
  static generate(type: EnvironmentType, size: number = MAZE_SIZE): { grid: CellType[][]; start: Position; goal: Position } {
    switch (type) {
      case EnvironmentType.MAZE:
        return this.generateMaze(size);
      case EnvironmentType.OPEN_FIELD:
        return this.generateOpenField(size);
      case EnvironmentType.CORRIDOR:
        return this.generateCorridor(size);
      case EnvironmentType.SPIRAL:
        return this.generateSpiral(size);
      case EnvironmentType.GRID:
        return this.generateGrid(size);
      default:
        return this.generateMaze(size);
    }
  }
}
