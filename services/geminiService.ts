
import { MazeState, DreamPrediction } from "../types";

export const getAgentInternalMonologue = async (mazeState: MazeState): Promise<string> => {
  // Simulated internal monologue based on agent state
  const { agentPos, goalPos } = mazeState;
  const distance = Math.abs(agentPos.x - goalPos.x) + Math.abs(agentPos.y - goalPos.y);
  
  const simulatedMonologues = [
    "Navigating through latent space... path emerging.",
    "Sensory input processed. Optimal route calculated.",
    "World model active. Traversing maze structure.",
    "Memory component engaged. Route prediction stable.",
    "Vision module scanning. Goal trajectory identified.",
    "Latent representation updated. Exploring novel states.",
    "Reward signal detected. Adjusting policy parameters.",
    "Dream sequence generated. Future states predicted.",
    "Curiosity drive active. Seeking unexplored regions.",
    "Exploration-exploitation balance optimized."
  ];
  
  // Add context-aware monologues based on distance
  if (distance < 5) {
    const nearGoalMonologues = [
      "Goal proximity detected. Final approach sequence initiated.",
      "Target within range. Optimizing final trajectory.",
      "Convergence imminent. Reward maximization active."
    ];
    return nearGoalMonologues[Math.floor(Math.random() * nearGoalMonologues.length)];
  }
  
  return simulatedMonologues[Math.floor(Math.random() * simulatedMonologues.length)];
};

export const predictNextState = async (mazeState: MazeState): Promise<DreamPrediction> => {
  // We simulate a controller/dream prediction
  const { agentPos, goalPos } = mazeState;
  
  // Logic to simulate a dream path towards goal
  const steps: any[] = [];
  let currX = agentPos.x;
  let currY = agentPos.y;
  
  for (let i = 0; i < 5; i++) {
    if (currX < goalPos.x) currX++;
    else if (currX > goalPos.x) currX--;
    if (currY < goalPos.y) currY++;
    else if (currY > goalPos.y) currY--;
    steps.push({ x: currX, y: currY });
  }

  return {
    steps,
    confidence: 0.85 + Math.random() * 0.1,
    description: "Memory component (M) predicts high probability route through the lower corridor."
  };
};
