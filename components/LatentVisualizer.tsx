
import React from 'react';
import { LatentVector } from '../types';

interface Props {
  latent: LatentVector;
}

const LatentVisualizer: React.FC<Props> = ({ latent }) => {
  return (
    <div className="bg-slate-900 p-4 border border-slate-700 rounded-xl h-full">
      <h3 className="text-xs font-bold text-slate-500 uppercase tracking-widest mb-4 flex items-center gap-2">
        <i className="fa-solid fa-eye text-emerald-400"></i> Vision Latent (z)
      </h3>
      <div className="grid grid-cols-4 gap-2">
        {latent.vector.map((val, i) => (
          <div key={i} className="space-y-1">
            <div className="h-16 bg-slate-800 rounded relative overflow-hidden flex flex-col justify-end">
              <div 
                className="w-full bg-emerald-500 transition-all duration-500" 
                style={{ height: `${val * 100}%` }}
              ></div>
            </div>
            <div className="text-[10px] text-center font-mono text-emerald-400">{(val * 10).toFixed(1)}</div>
          </div>
        ))}
      </div>
      
      <div className="mt-6 border-t border-slate-800 pt-4">
        <h4 className="text-[10px] font-semibold text-slate-500 mb-2 uppercase italic">VAE Reconstruction (Partial)</h4>
        <div className="grid grid-cols-5 gap-1">
          {latent.reconstruction.flat().map((v, i) => (
            <div 
              key={i} 
              className="h-4 w-4 rounded-sm"
              style={{ backgroundColor: `rgba(16, 185, 129, ${v})` }}
            ></div>
          ))}
        </div>
      </div>
    </div>
  );
};

export default LatentVisualizer;
