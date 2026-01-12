
import React from 'react';
import { DreamPrediction } from '../types';

interface Props {
  dream: DreamPrediction | null;
  monologue: string;
}

const DreamState: React.FC<Props> = React.memo(({ dream, monologue }) => {
  return (
    <div className="bg-slate-900 p-4 border border-slate-700 rounded-xl h-full flex flex-col">
      <h3 className="text-xs font-bold text-slate-500 uppercase tracking-widest mb-4 flex items-center gap-2">
        <i className="fa-solid fa-cloud text-purple-400"></i> Memory (M) Dreams
      </h3>
      
      <div className="flex-1 bg-slate-950 rounded-lg p-3 border border-purple-500/20 mb-4 font-mono text-sm leading-relaxed relative overflow-hidden">
        <div className="absolute top-0 right-0 p-2 opacity-10">
          <i className="fa-solid fa-brain text-4xl text-purple-500"></i>
        </div>
        <p className="text-purple-300">
          <span className="text-purple-500">{'>>> [AGENT_MONOLOGUE]:'}</span><br/>
          {monologue}
        </p>
      </div>

      <div className="space-y-3">
        <div className="flex justify-between items-center">
          <span className="text-[10px] text-slate-500">PREDICTION CONFIDENCE</span>
          <span className="text-[10px] text-purple-400 font-mono">{(dream?.confidence || 0 * 100).toFixed(1)}%</span>
        </div>
        <div className="h-1.5 w-full bg-slate-800 rounded-full overflow-hidden">
          <div 
            className="h-full bg-purple-500 transition-all duration-700" 
            style={{ width: `${(dream?.confidence || 0) * 100}%` }}
          ></div>
        </div>
        <p className="text-[10px] text-slate-400 italic">
          {dream?.description || "Simulating next 5 actions in latent space..."}
        </p>
      </div>
    </div>
  );
});

DreamState.displayName = 'DreamState';

export default DreamState;
