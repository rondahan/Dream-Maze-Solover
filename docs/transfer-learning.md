# 🔄 Transfer Learning & Knowledge Saving

Complete guide to the **SAVE KNOWLEDGE** feature and transfer learning capabilities.

---

## Overview

The **SAVE KNOWLEDGE** button enables you to save the learned knowledge (neural network weights) from your current agent and transfer it to new environments. This is part of the **Transfer Learning** system that allows the agent to start with knowledge from previous learning sessions.

---

## What is Transfer Learning?

Transfer Learning allows an AI agent to:

- **Reuse knowledge** learned in one environment in a different environment
- **Start with better initial weights** instead of random initialization
- **Accelerate learning** in new environments by building on previous experience
- **Share knowledge** between different environment types (Maze → Open Field → Corridor, etc.)

---

## SAVE KNOWLEDGE Button

### Location

Top control bar, next to the Environment Selector.

### What It Does

The **SAVE KNOWLEDGE** button saves the current state of all neural network components:

1. **Policy Network** - Decision-making weights
2. **VAE (Vision)** - Encoder and decoder weights for visual processing
3. **MDN-RNN (Memory)** - Recurrent network weights for prediction

These weights are stored in a shared knowledge base that can be loaded into new agent instances.

### Requirements

To use **SAVE KNOWLEDGE**, you need:

- ✅ **NN ON** - Neural Network must be enabled
- ✅ **Networks Initialized** - All networks (Policy, VAE, MDN-RNN) must be loaded
- ✅ **Knowledge Base Ready** - The knowledge base system must be initialized

### How to Use

1. **Start Learning**
   - Turn **NN ON** (if not already on)
   - Start the simulation and let the agent learn
   - Wait for networks to initialize (check logs)

2. **Let Agent Learn**
   - Allow the agent to complete at least one episode
   - The more the agent learns, the more valuable the saved knowledge

3. **Save Knowledge**
   - Click **"SAVE KNOWLEDGE"** button
   - Check the logs for confirmation messages:
     - `✓ Policy Network saved`
     - `✓ VAE (Vision) saved`
     - `✓ MDN-RNN (Memory) saved`
     - `✅ Knowledge saved successfully!`

4. **Use Saved Knowledge**
   - Switch to a new environment type
   - The saved knowledge can be loaded (if transfer learning is implemented)
   - Agent starts with better initial knowledge

### Error Messages

If the button doesn't work, you'll see error messages in the logs:

- **"Cannot save: Missing components"** - Some networks aren't initialized
- **"Cannot save: Neural Network is OFF"** - Turn NN ON first
- **"Error saving knowledge"** - Technical error occurred

### Troubleshooting

**Button is disabled:**
- Make sure **NN ON** is enabled
- Wait for networks to initialize (check logs)
- Refresh the page if networks fail to load

**Save fails:**
- Check browser console for detailed errors
- Ensure TensorFlow.js loaded successfully
- Try saving after agent completes one episode

---

## When to Use Transfer Learning

### Recommended Use Cases

1. **Environment Switching**
   - Learn in one environment type (e.g., Maze)
   - Save knowledge
   - Switch to another type (e.g., Open Field)
   - Agent starts with relevant knowledge

2. **Preserving Progress**
   - Save before resetting or refreshing
   - Save after successful learning episodes
   - Build up knowledge over multiple sessions

3. **Comparative Learning**
   - Save knowledge from one algorithm
   - Compare with different algorithms
   - Analyze transfer effectiveness

### When NOT to Use

- **First-time learning** - No knowledge to transfer yet
- **Simple experiments** - Not necessary for basic testing
- **Algorithm comparison** - May want fresh starts for fair comparison

---

## Technical Details

### What Gets Saved

The knowledge base stores:

```typescript
{
  policyWeights: Tensor[][],      // Policy Network weights
  vaeEncoderWeights: Tensor[][],  // VAE encoder weights
  vaeDecoderWeights: Tensor[][], // VAE decoder weights
  rnnWeights: Tensor[][]          // MDN-RNN weights
}
```

### Storage Location

- Knowledge is stored **in memory** (not persisted to disk)
- Knowledge is **lost on page refresh**
- For persistence, you would need to implement browser storage (localStorage/IndexedDB)

### Transfer Modes

The system supports different transfer modes:

- **Full Transfer** - Copy all weights exactly
- **Partial Transfer** - Copy only some layers (e.g., hidden layers, not output)
- **Fine-tuning** - Copy weights and continue training with lower learning rate

---

## Future Enhancements

Potential improvements:

- **Persistent Storage** - Save to browser storage or export/import files
- **Multiple Knowledge Bases** - Save different knowledge sets
- **Automatic Transfer** - Auto-load knowledge when switching environments
- **Knowledge Visualization** - Visualize what knowledge is saved
- **Transfer Metrics** - Measure how well knowledge transfers

---

## Related Documentation

- **[User Guide](./user-guide.md)** - Complete interface guide
- **[Architecture](./architecture.md)** - V-M-C architecture details
- **[Learning Algorithms](./learning-algorithms.md)** - How the agent learns
- **[Logs Reference](./logs-reference.md)** - Understanding log messages

---

**Note:** Transfer Learning is an advanced feature. For basic usage, you don't need to use it. The agent learns effectively without it, but transfer learning can accelerate learning in new environments.
