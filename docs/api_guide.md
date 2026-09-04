# Alpha-DQN REST API Interaction Guide & Reference

This document provides complete interaction specifications for communicating with the **Alpha-DQN HTTP REST API Server**. 

It is designed for frontend developers, bot builders, browser automation scripts (e.g. Tampermonkey, Puppeteer), and external microservices.

---

## 1. Quick Start & Server Discovery

- **Local Base URL:** `http://127.0.0.1:8001`
- **Tailscale Network URL:** `http://100.64.148.113:8001`
- **Interactive Swagger UI (OpenAPI):** `http://100.64.148.113:8001/docs` (or `http://127.0.0.1:8001/docs`)
- **Health Check Endpoint:** `GET /health`

```bash
# Verify server status via Tailscale
curl -s http://100.64.148.113:8001/health
# Response: {"status":"ok","project":"Alpha-DQN","environment":"development","version":"0.1.0"}
```

---

## 2. API Endpoint Summary Table

| Method | Endpoint | Description | Typical Frequency |
| :--- | :--- | :--- | :--- |
| `POST` | `/api/dqn/init` | Initialize or reconfigure an agent session | Once per session startup |
| `POST` | `/api/dqn/act` | Request discrete action from state observation | Every frame or decision cycle (e.g. 60 FPS or every 5s) |
| `POST` | `/api/dqn/step` | Ingest environment transition, compute intrinsic reward, and train | After action execution (on environment feedback) |
| `POST` | `/api/dqn/train` | Explicitly trigger one mini-batch gradient update | Periodic or custom training schedule |
| `GET` | `/api/dqn/metrics` | Retrieve real-time telemetry (loss, curiosity, exploration) | Telemetry dashboard / polling (e.g. 1Hz) |
| `POST` | `/api/dqn/reset-episode` | Reset episodic counters (sticky actions, frontier trackers) | At episode completion (`done = true`) |
| `POST` | `/api/dqn/reset-agent` | Reset full agent weights and replay buffer to pristine state | On full environment/agent reset |
| `POST` | `/api/dqn/save` | Persist weights and metadata to JSON manifest | Periodic checkpointing |
| `POST` | `/api/dqn/load` | Restore weights and metadata from JSON manifest | Session resumption |

---

## 3. Detailed Endpoint Specifications & Payloads

### 1. Initialize Session (`POST /api/dqn/init`)

Initializes a new isolated agent instance or reconfigures an existing session.

#### cURL Request:
```bash
curl -X POST http://127.0.0.1:8001/api/dqn/init \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "spaceship_agent",
    "config": {
      "state_size": 6,
      "action_size": 5,
      "layers": [
        {"units": 64, "activation": "relu"},
        {"units": 64, "activation": "relu"}
      ],
      "dueling": true,
      "batch_size": 16,
      "learning_rate": 0.0005,
      "curiosity": { "enabled": true, "intrinsic_reward_scale": 0.15 },
      "adaptive_loss": { "enabled": true, "initial_beta": 0.5, "tau": 0.598 },
      "exploration": { "enabled": true, "sticky_min": 2, "sticky_max": 4 }
    }
  }'
```

#### JSON Response:
```json
{
  "session_id": "spaceship_agent",
  "status": "initialized",
  "state_size": 6,
  "action_size": 5,
  "curiosity_enabled": true,
  "adaptive_loss_enabled": true,
  "exploration_enabled": true
}
```

---

### 2. Action Inference (`POST /api/dqn/act`)

Queries the agent's policy to select an action given the current continuous state vector.

#### cURL Request:
```bash
curl -X POST http://127.0.0.1:8001/api/dqn/act \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "spaceship_agent",
    "state": [0.12, -0.25, 0.05, 0.0, -0.4, 0.3],
    "force_pure_neural": false
  }'
```

#### JSON Response:
```json
{
  "session_id": "spaceship_agent",
  "action": 3
}
```

> **Performance Note:** Average inference latency is **~4.2 ms**, comfortably fitting inside the 16.6 ms frame budget of 60 FPS real-time web games.

---

### 3. Record Transition & Train (`POST /api/dqn/step`)

Sends the transition experience tuple $(s, a, r, s', \text{done})$ to the agent. The server computes the intrinsic curiosity reward bonus, records the transition into the circular replay buffer, and automatically executes mini-batch training if `auto_train=true`.

#### cURL Request:
```bash
curl -X POST http://127.0.0.1:8001/api/dqn/step \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "spaceship_agent",
    "state": [0.12, -0.25, 0.05, 0.0, -0.4, 0.3],
    "action": 3,
    "reward": 0.5,
    "next_state": [0.12, -0.20, 0.05, 0.05, -0.38, 0.26],
    "done": false,
    "auto_train": true
  }'
```

#### JSON Response:
```json
{
  "session_id": "spaceship_agent",
  "intrinsic_reward": 0.434605,
  "effective_reward": 0.565191,
  "training_loss": 0.003633,
  "epsilon": 0.9984,
  "step_counter": 16,
  "train_step_count": 9
}
```

---

### 4. Telemetry & Metrics (`GET /api/dqn/metrics`)

Returns real-time telemetry from the Double-DQN agent, Curiosity Engine, Adaptive Loss module, and Directed Exploration policy.

#### cURL Request:
```bash
curl -s "http://127.0.0.1:8001/api/dqn/metrics?session_id=spaceship_agent"
```

#### JSON Response:
```json
{
  "session_id": "spaceship_agent",
  "step_counter": 50,
  "train_step_count": 35,
  "epsilon": 0.9702,
  "training_loss": 0.00284,
  "memory_size": 50,
  "memory_capacity": 500,
  "curiosity_enabled": true,
  "curiosity_metrics": {
    "unique_states_global": 38,
    "unique_states_episode": 0,
    "running_error_mean": 0.0142,
    "max_distance_episode": 0.0,
    "total_transition_steps": 140
  },
  "adaptive_loss_enabled": true,
  "loss_metrics": {
    "current_beta": 0.0118,
    "tau_optimism": 0.598,
    "running_td_mean": -0.0025,
    "running_td_var": 0.0089,
    "total_updates": 35
  },
  "exploration_enabled": true,
  "exploration_metrics": {
    "unique_states_tracked": 24,
    "total_exploration_steps": 50,
    "sticky_counter": 1,
    "action_distribution": [8.0, 12.0, 15.0, 9.0, 6.0]
  }
}
```

---

### 5. Save & Load Weights (`POST /api/dqn/save` and `POST /api/dqn/load`)

Persists or restores the agent's neural network weights to/from a cross-platform JSON manifest.

#### Save:
```bash
curl -X POST http://127.0.0.1:8001/api/dqn/save \
  -H "Content-Type: application/json" \
  -d '{"session_id": "spaceship_agent", "file_path": "models/spaceship_weights.json"}'
```

#### Load:
```bash
curl -X POST http://127.0.0.1:8001/api/dqn/load \
  -H "Content-Type: application/json" \
  -d '{"session_id": "spaceship_agent", "file_path": "models/spaceship_weights.json"}'
```

---

## 4. Frontend & Browser Integration Code Snippets

### JavaScript / Browser Console Integration
```javascript
const API_BASE = "http://127.0.0.1:8001/api/dqn";
const SESSION_ID = "browser_game_session";

// 1. Initialize Agent
async function initAgent() {
    await fetch(`${API_BASE}/init`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
            session_id: SESSION_ID,
            config: { state_size: 6, action_size: 5 }
        })
    });
}

// 2. Query Action in Game Loop
async function getAgentAction(stateArray) {
    const res = await fetch(`${API_BASE}/act`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ session_id: SESSION_ID, state: stateArray })
    });
    const data = await res.json();
    return data.action;
}

// 3. Step Transition
async function reportTransition(state, action, reward, nextState, isDone) {
    await fetch(`${API_BASE}/step`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
            session_id: SESSION_ID,
            state: state,
            action: action,
            reward: reward,
            next_state: nextState,
            done: isDone,
            auto_train: true
        })
    });
}
```

---

## 5. Error Handling & Troubleshooting

- **CORS Errors:** FastAPI is configured with `CORSMiddleware` (`allow_origins=["*"]`). If calling from strict browser domains (like Binance or secure pages), consider running via a Browser Extension, Userscript (Tampermonkey), or a reverse proxy.
- **Validation 422 Error:** Ensure the state vector length exactly matches `state_size` passed during initialization.
- **Port Conflict:** If port 8000 is occupied, Alpha-DQN defaults to `8001` via `.env`.
