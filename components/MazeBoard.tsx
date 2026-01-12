
import React, { useMemo } from 'react';
import { MazeState, CellType } from '../types';
import { COLORS, CELL_SIZE } from '../constants';

interface MazeBoardProps {
  maze: MazeState;
}

const MazeBoard: React.FC<MazeBoardProps> = ({ maze }) => {
  // Optimize history lookup: Convert array to Set for O(1) lookup instead of O(n)
  const historySet = useMemo(() => {
    const set = new Set<string>();
    maze.history.forEach(p => {
      set.add(`${p.x},${p.y}`);
    });
    return set;
  }, [maze.history]);

  return (
    <div 
      className="relative bg-slate-900 border-2 border-slate-700 rounded-lg overflow-hidden shadow-2xl"
      style={{
        display: 'grid',
        gridTemplateColumns: `repeat(${maze.grid.length}, ${CELL_SIZE}px)`,
        gridTemplateRows: `repeat(${maze.grid.length}, ${CELL_SIZE}px)`,
      }}
    >
      {maze.grid.map((row, y) => 
        row.map((cell, x) => {
          const isAgent = maze.agentPos.x === x && maze.agentPos.y === y;
          // Fast O(1) lookup instead of O(n) array.some()
          const isHistory = historySet.has(`${x},${y}`);
          
          return (
            <div
              key={`${x}-${y}`}
              className="relative"
              style={{
                width: CELL_SIZE,
                height: CELL_SIZE,
                backgroundColor: cell === CellType.WALL ? COLORS[CellType.WALL] : COLORS[CellType.PATH],
                border: '0.1px solid rgba(255,255,255,0.02)'
              }}
            >
              {cell === CellType.GOAL && (
                <div className="absolute inset-0 flex items-center justify-center">
                  <i className="fa-solid fa-trophy text-amber-500 animate-pulse text-xs"></i>
                </div>
              )}
              {isHistory && !isAgent && cell !== CellType.GOAL && (
                <div className="absolute inset-0 bg-blue-500 opacity-20"></div>
              )}
              {isAgent && (
                <div 
                  className="absolute inset-1 bg-blue-500 rounded-sm shadow-[0_0_10px_rgba(59,130,246,0.8)] z-10 flex items-center justify-center"
                >
                  <i className="fa-solid fa-robot text-[10px] text-white"></i>
                </div>
              )}
            </div>
          );
        })
      )}
    </div>
  );
};

export default React.memo(MazeBoard);
