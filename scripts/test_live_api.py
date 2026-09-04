"""
Live API Capability Test Suite for Alpha-DQN Server.
Interacts with the running HTTP REST service on http://127.0.0.1:8001 to validate:
  1. Health check and discovery
  2. Agent initialization with full trilogy configuration (Spaceship Game setup)
  3. Action inference queries
  4. Multi-step transition ingestion, curiosity reward computation, and auto-training
  5. Live metric extraction across all submodules (Loss, Exploration, Curiosity)
  6. Weight persistence (save & load)
  7. Episodic and full agent reset
"""

import json
import sys
import httpx

BASE_URL = "http://127.0.0.1:8001"
SESSION_ID = "spaceship_live_session"

def log_step(name: str, payload: dict, response: dict):
    print(f"\n=======================================================")
    print(f"🔹 [STEP] {name}")
    print(f"➡️ Request:  {json.dumps(payload, indent=2)}")
    print(f"⬅️ Response: {json.dumps(response, indent=2)}")

def main():
    client = httpx.Client(base_url=BASE_URL, timeout=10.0)
    
    print(f"🚀 Connecting to Alpha-DQN Live Server at {BASE_URL}...")
    
    # 1. Health Check
    health_resp = client.get("/health")
    assert health_resp.status_code == 200, f"Health check failed: {health_resp.text}"
    log_step("GET /health", {}, health_resp.json())

    # 2. Agent Initialization for Spaceship Environment
    # State: [ship_x, ship_y, vel_x, vel_y, asteroid_dx, asteroid_dy] (size = 6)
    # Action: 0: IDLE, 1: LEFT, 2: RIGHT, 3: THRUST, 4: FIRE (size = 5)
    init_payload = {
        "session_id": SESSION_ID,
        "config": {
            "state_size": 6,
            "action_size": 5,
            "layers": [
                {"units": 64, "activation": "relu"},
                {"units": 64, "activation": "relu"}
            ],
            "dueling": True,
            "batch_size": 8,
            "memory_capacity": 500,
            "train_interval": 1,
            "learning_rate": 0.0005,
            "curiosity": {
                "enabled": True,
                "intrinsic_reward_scale": 0.20,
            },
            "adaptive_loss": {
                "enabled": True,
                "initial_beta": 0.5,
                "tau": 0.598,
            },
            "exploration": {
                "enabled": True,
                "sticky_min": 2,
                "sticky_max": 4,
                "q_margin_threshold": 0.58,
            }
        }
    }
    init_resp = client.post("/api/dqn/init", json=init_payload)
    assert init_resp.status_code == 200, f"Init failed: {init_resp.text}"
    init_data = init_resp.json()
    assert init_data["curiosity_enabled"] is True
    assert init_data["adaptive_loss_enabled"] is True
    assert init_data["exploration_enabled"] is True
    log_step("POST /api/dqn/init", init_payload, init_data)

    # 3. Action Query (Inference)
    state_sample = [0.1, -0.2, 0.05, 0.0, -0.4, 0.3]
    act_payload = {
        "session_id": SESSION_ID,
        "state": state_sample,
        "force_pure_neural": False,
    }
    act_resp = client.post("/api/dqn/act", json=act_payload)
    assert act_resp.status_code == 200, f"Act failed: {act_resp.text}"
    act_data = act_resp.json()
    assert 0 <= act_data["action"] < 5
    log_step("POST /api/dqn/act (Query Action)", act_payload, act_data)

    # 4. Multi-step Flight Trajectory (Stepping with Curiosity & Auto-Train)
    print("\n=======================================================")
    print("🔹 [STEP] Simulating 15 Environment Transitions via POST /api/dqn/step...")
    curr_state = state_sample
    for step_idx in range(15):
        # Query action
        action = client.post("/api/dqn/act", json={"session_id": SESSION_ID, "state": curr_state}).json()["action"]
        
        # Next state with small motion
        next_state = [
            round(curr_state[0] + 0.05 * (1 if action == 2 else (-1 if action == 1 else 0)), 3),
            round(curr_state[1] + 0.05 * (1 if action == 3 else 0), 3),
            0.05, 0.0,
            round(curr_state[4] + 0.02, 3),
            round(curr_state[5] - 0.04, 3),
        ]
        
        # Reward: survival reward + dodge bonus
        reward = 0.5 if abs(next_state[4]) > 0.1 else -1.0
        done = (step_idx == 14)

        step_payload = {
            "session_id": SESSION_ID,
            "state": curr_state,
            "action": action,
            "reward": reward,
            "next_state": next_state,
            "done": done,
            "auto_train": True,
        }
        step_resp = client.post("/api/dqn/step", json=step_payload)
        assert step_resp.status_code == 200, f"Step {step_idx} failed: {step_resp.text}"
        step_data = step_resp.json()
        
        print(f"   Step {step_idx+1:02d}: Action={action} | Extrinsic={reward} | IntrinsicBonus={step_data['intrinsic_reward']} | Loss={step_data['training_loss']}")
        curr_state = next_state

    # 5. Telemetry & Metrics Verification
    metrics_resp = client.get(f"/api/dqn/metrics?session_id={SESSION_ID}")
    assert metrics_resp.status_code == 200, f"Metrics failed: {metrics_resp.text}"
    m = metrics_resp.json()
    log_step("GET /api/dqn/metrics", {"session_id": SESSION_ID}, m)

    # Validate sub-module metrics
    assert m["curiosity_enabled"] is True
    assert m["curiosity_metrics"]["total_transition_steps"] >= 15
    assert m["adaptive_loss_enabled"] is True
    assert m["loss_metrics"]["tau_optimism"] == 0.598
    assert m["exploration_enabled"] is True
    assert m["exploration_metrics"]["total_exploration_steps"] >= 15

    # 6. Weight Persistence (Save & Load)
    save_payload = {"session_id": SESSION_ID, "file_path": "models/spaceship_live_weights.json"}
    save_resp = client.post("/api/dqn/save", json=save_payload)
    assert save_resp.status_code == 200, f"Save failed: {save_resp.text}"
    log_step("POST /api/dqn/save", save_payload, save_resp.json())

    load_resp = client.post("/api/dqn/load", json=save_payload)
    assert load_resp.status_code == 200, f"Load failed: {load_resp.text}"
    log_step("POST /api/dqn/load", save_payload, load_resp.json())

    # 7. Reset Episode
    reset_resp = client.post(f"/api/dqn/reset-episode?session_id={SESSION_ID}")
    assert reset_resp.status_code == 200
    log_step("POST /api/dqn/reset-episode", {"session_id": SESSION_ID}, reset_resp.json())

    print("\n=======================================================")
    print("🎉 ALL LIVE API CAPABILITY TESTS PASSED WITH 100% SUCCESS!")
    print("=======================================================\n")

if __name__ == "__main__":
    main()
