# 🧠 Universal TensorFlow.js Double-DQN SDK

> **Philosophy:** *"আমি শিখি, তুমি শেখাও"* (I Learn, You Teach)  
> A lightweight, game-agnostic Reinforcement Learning SDK powered by TensorFlow.js. Works in both **Node.js** and **Browser**.

---

## 🚀 Quick Import

```javascript
import { DQNAgent } from './lib/dqn/index.js';

const agent = new DQNAgent({
  stateSize: 4,
  actionSize: 2,
  layers: [
    { units: 64, activation: 'relu' },
    { units: 32, activation: 'relu' }
  ]
});

await agent.init();
const action = agent.act(state);
agent.remember(state, action, reward, nextState, done);
await agent.trainBatch();
```

---

## 📖 Full Documentation

For comprehensive guides, real-world examples (GridWorld, Continuous Runner, Expert DQfD), configuration reference tables, and persistence setup, see:

👉 **[`docs/DQN_SDK_DOCUMENTATION.md`](../../docs/DQN_SDK_DOCUMENTATION.md)**
