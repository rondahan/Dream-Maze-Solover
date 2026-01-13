import { MazeState, Position, CellType } from '../types';
import { MAZE_SIZE } from '../constants';

/**
 * A* Pathfinding Algorithm
 * Classic heuristic search algorithm for finding optimal paths
 * Uses Manhattan distance as heuristic
 */
export class AStar {
  private isInitialized = false;

  constructor() {}

  /**
   * Initialize A* (no neural network needed)
   */
  async initialize() {
    this.isInitialized = true;
    console.log('A* initialized');
  }

  /**
   * Heuristic function: Manhattan distance
   */
  private heuristic(pos: Position, goal: Position): number {
    return Math.abs(pos.x - goal.x) + Math.abs(pos.y - goal.y);
  }

  /**
   * Get valid neighbors
   */
  private getNeighbors(pos: Position, grid: CellType[][]): Position[] {
    const neighbors: Position[] = [];
    const directions = [
      { x: 0, y: 1 },   // up
      { x: 0, y: -1 },  // down
      { x: 1, y: 0 },   // right
      { x: -1, y: 0 }   // left
    ];

    for (const dir of directions) {
      const newPos = { x: pos.x + dir.x, y: pos.y + dir.y };
      if (
        newPos.x >= 0 && newPos.x < MAZE_SIZE &&
        newPos.y >= 0 && newPos.y < MAZE_SIZE &&
        grid[newPos.y][newPos.x] !== CellType.WALL
      ) {
        neighbors.push(newPos);
      }
    }

    return neighbors;
  }

  /**
   * Find path using A* algorithm
   */
  private findPath(start: Position, goal: Position, grid: CellType[][]): Position[] | null {
    interface Node {
      pos: Position;
      g: number; // Cost from start
      h: number; // Heuristic to goal
      f: number; // Total cost (g + h)
      parent: Node | null;
    }

    const openSet: Node[] = [];
    const closedSet = new Set<string>();

    const startNode: Node = {
      pos: start,
      g: 0,
      h: this.heuristic(start, goal),
      f: this.heuristic(start, goal),
      parent: null
    };

    openSet.push(startNode);

    while (openSet.length > 0) {
      // Find node with lowest f score
      let currentIndex = 0;
      for (let i = 1; i < openSet.length; i++) {
        if (openSet[i].f < openSet[currentIndex].f) {
          currentIndex = i;
        }
      }

      const current = openSet.splice(currentIndex, 1)[0];
      const currentKey = `${current.pos.x},${current.pos.y}`;
      closedSet.add(currentKey);

      // Check if goal reached
      if (current.pos.x === goal.x && current.pos.y === goal.y) {
        // Reconstruct path
        const path: Position[] = [];
        let node: Node | null = current;
        while (node) {
          path.unshift(node.pos);
          node = node.parent;
        }
        return path;
      }

      // Check neighbors
      const neighbors = this.getNeighbors(current.pos, grid);
      for (const neighbor of neighbors) {
        const neighborKey = `${neighbor.x},${neighbor.y}`;
        if (closedSet.has(neighborKey)) continue;

        const g = current.g + 1;
        const h = this.heuristic(neighbor, goal);
        const f = g + h;

        // Check if already in open set
        const existingNode = openSet.find(n => n.pos.x === neighbor.x && n.pos.y === neighbor.y);
        if (existingNode) {
          if (g < existingNode.g) {
            existingNode.g = g;
            existingNode.f = f;
            existingNode.parent = current;
          }
        } else {
          openSet.push({
            pos: neighbor,
            g,
            h,
            f,
            parent: current
          });
        }
      }
    }

    return null; // No path found
  }

  /**
   * Select next action based on A* path
   */
  async selectAction(
    mazeState: MazeState,
    possibleMoves: Position[],
    currentSteps: number,
    curiosityWeight: number,
    epsilon: number
  ): Promise<{ pos: Position; actionIndex: number }> {
    if (possibleMoves.length === 0) {
      return { pos: mazeState.agentPos, actionIndex: -1 };
    }

    // Epsilon-greedy: sometimes explore randomly
    if (Math.random() < epsilon) {
      const randomIndex = Math.floor(Math.random() * possibleMoves.length);
      return { pos: possibleMoves[randomIndex], actionIndex: randomIndex };
    }

    // Use A* to find optimal path
    const path = this.findPath(mazeState.agentPos, mazeState.goalPos, mazeState.grid);
    
    if (path && path.length > 1) {
      // Next step in optimal path
      const nextStep = path[1];
      
      // Check if next step is in possible moves
      const moveIndex = possibleMoves.findIndex(
        m => m.x === nextStep.x && m.y === nextStep.y
      );
      
      if (moveIndex >= 0) {
        // Find action index
        const dirs = [
          { x: 0, y: 1 },   // up
          { x: 0, y: -1 },  // down
          { x: 1, y: 0 },   // right
          { x: -1, y: 0 }   // left
        ];
        
        const dx = nextStep.x - mazeState.agentPos.x;
        const dy = nextStep.y - mazeState.agentPos.y;
        
        const actionIndex = dirs.findIndex(d => d.x === dx && d.y === dy);
        return { pos: nextStep, actionIndex: actionIndex >= 0 ? actionIndex : 0 };
      }
    }

    // Fallback: greedy distance
    let bestMove = possibleMoves[0];
    let minDist = Infinity;
    for (const move of possibleMoves) {
      const dist = Math.abs(move.x - mazeState.goalPos.x) + Math.abs(move.y - mazeState.goalPos.y);
      if (dist < minDist) {
        minDist = dist;
        bestMove = move;
      }
    }

    const dirs = [
      { x: 0, y: 1 },   // up
      { x: 0, y: -1 },  // down
      { x: 1, y: 0 },   // right
      { x: -1, y: 0 }   // left
    ];
    
    const dx = bestMove.x - mazeState.agentPos.x;
    const dy = bestMove.y - mazeState.agentPos.y;
    const actionIndex = dirs.findIndex(d => d.x === dx && d.y === dy);
    
    return { pos: bestMove, actionIndex: actionIndex >= 0 ? actionIndex : 0 };
  }

  /**
   * Train (A* doesn't learn, but we keep the interface consistent)
   */
  async train(
    states: number[][],
    actions: number[],
    rewards: number[],
    nextStates: number[],
    dones: boolean[]
  ) {
    // A* doesn't learn, so this is a no-op
    // But we keep the interface for consistency
  }

  /**
   * Get model summary
   */
  getSummary(): string {
    return 'A* Pathfinding (Heuristic Search)';
  }
}
