"""
OpenEvolve Evaluator for Intrinsic Curiosity and Exploration Reward Optimization.

Evaluates candidate CuriosityRewardModule implementations against 4 core pillars:
1. Habituation / Boredom Decay: Intrinsic reward must decay toward 0 as states become familiar.
2. Noisy-TV Immunity: Pure stochastic noise transitions must receive significantly lower bonus than learnable novel states.
3. State-Space Coverage: Agent guided by curiosity must maximize unique state visits in a sparse maze.
4. Goal Reach Rate: Deep exploration efficiency on a sparse-reward navigation environment.

MAP-Elites Dimensions:
- state_coverage: Exploration breadth [0.0, 1.0]
- noisy_tv_immunity: Resistance to stochastic noise traps [0.0, 1.0]
"""

import importlib.util
import os
import sys
import time
import traceback
from typing import Any, Dict, Optional, Tuple

import numpy as np
from openevolve.evaluation_result import EvaluationResult


def set_global_seeds(seed: int = 42):
    """Enforce strict deterministic reproducibility across NumPy, Random, and PyTorch."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def load_candidate_module(program_path: str):
    """Safely and dynamically load candidate module from file path."""
    module_name = os.path.basename(program_path).replace(".py", "")
    spec = importlib.util.spec_from_file_location(module_name, program_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module specification from {program_path}")
    
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_habituation_decay(module_instance) -> Tuple[float, Dict[str, Any]]:
    """
    Pillar 1: Habituation / Boredom Decay Test.
    
    Repeatedly visits identical transitions for 50 steps.
    A healthy curiosity module must reduce its reward as familiarity increases.
    """
    state = np.array([0.5, -0.2, 0.8, -0.4], dtype=np.float64)
    action = 0
    next_state = np.array([0.6, -0.1, 0.7, -0.3], dtype=np.float64)

    # Initial reward on first visit
    r_initial = float(module_instance.compute_intrinsic_reward(state, action, next_state))

    # Repeatedly update with identical transition 50 times
    for _ in range(50):
        module_instance.update(
            np.array([state]),
            np.array([action]),
            np.array([next_state])
        )

    # Final reward after repeated visits
    r_final = float(module_instance.compute_intrinsic_reward(state, action, next_state))

    details = {
        "r_initial": r_initial,
        "r_final": r_final,
    }

    if r_initial <= 1e-7:
        # Module failed to produce any exploration bonus initially
        return 0.0, details

    # Ideal behavior: r_final should be close to 0
    # Decay score: 1.0 if r_final == 0, 0.0 if r_final >= r_initial
    decay_ratio = 1.0 - min(1.0, max(0.0, r_final / (r_initial + 1e-8)))
    return float(decay_ratio), details


def test_noisy_tv_immunity(module_instance) -> Tuple[float, Dict[str, Any]]:
    """
    Pillar 2: Noisy-TV Stochastic Noise Immunity Test.
    
    Compares intrinsic reward on structured deterministic transitions
    versus unpredictable pure white noise transitions.
    A naive predictor error module gets trapped by white noise.
    A smart curiosity formulation resists the noisy-TV trap.
    """
    rng = np.random.default_rng(123)
    
    # 1. Train on 30 structured, predictable transitions
    for step in range(30):
        s_struct = np.sin(np.array([step * 0.1, step * 0.2, step * 0.05, step * 0.3]))
        act = step % 2
        ns_struct = s_struct + 0.1 * (act + 1)
        module_instance.update(np.array([s_struct]), np.array([act]), np.array([ns_struct]))

    # Test novel structured state
    s_novel = np.sin(np.array([31 * 0.1, 31 * 0.2, 31 * 0.05, 31 * 0.3]))
    ns_novel = s_novel + 0.1
    r_novel = float(module_instance.compute_intrinsic_reward(s_novel, 0, ns_novel))

    # 2. Test pure stochastic white-noise transitions
    noise_rewards = []
    for _ in range(30):
        s_noise = rng.standard_normal(4)
        act = rng.integers(0, 2)
        ns_noise = rng.standard_normal(4)  # Completely unpredictable Gaussian noise
        
        # Train on noise
        module_instance.update(np.array([s_noise]), np.array([act]), np.array([ns_noise]))
        r_noise = float(module_instance.compute_intrinsic_reward(s_noise, act, ns_noise))
        noise_rewards.append(r_noise)

    avg_r_noise = float(np.mean(noise_rewards)) if noise_rewards else 1.0

    details = {
        "reward_structured_novel": r_novel,
        "reward_pure_noise": avg_r_noise,
    }

    # If noise reward is much higher than novel structured reward, immunity is low.
    # Score in [0, 1]: 1.0 if noise reward is 0, 0.5 if equal, <0.5 if noise is preferred.
    if (r_novel + avg_r_noise) <= 1e-8:
        immunity = 0.5
    else:
        diff_ratio = (r_novel - avg_r_noise) / (r_novel + avg_r_noise + 1e-8)
        immunity = 0.5 + (0.5 * diff_ratio)

    return float(max(0.0, min(1.0, immunity))), details


def test_state_space_coverage(module_instance) -> Tuple[float, Dict[str, Any]]:
    """
    Pillar 3: State-Space Coverage in Sparse Exploration (8x8 Grid = 64 states).
    
    A lightweight Q-learning agent navigates an 8x8 grid where external rewards are 0.
    The agent is solely guided by the intrinsic curiosity reward r_i.
    Measures what percentage of the state space the agent visits within 200 steps.
    """
    grid_size = 8
    total_states = grid_size * grid_size
    visited_states = set()

    # Tabular Q-table [64, 4 actions: 0=up, 1=right, 2=down, 3=left]
    q_table = np.zeros((total_states, 4), dtype=np.float64)
    pos = [0, 0]
    visited_states.add((pos[0], pos[1]))

    moves = [(-1, 0), (0, 1), (1, 0), (0, -1)]
    lr = 0.1
    gamma = 0.95
    epsilon = 0.15
    rng = np.random.default_rng(42)

    def state_to_vec(p):
        return np.array([p[0] / float(grid_size), p[1] / float(grid_size), 0.0, 0.0], dtype=np.float64)

    def state_idx(p):
        return p[0] * grid_size + p[1]

    for step in range(200):
        curr_idx = state_idx(pos)
        s_vec = state_to_vec(pos)

        # Epsilon-greedy action selection
        if rng.random() < epsilon:
            action = int(rng.integers(0, 4))
        else:
            action = int(np.argmax(q_table[curr_idx]))

        # Apply transition
        dy, dx = moves[action]
        next_pos = [
            max(0, min(grid_size - 1, pos[0] + dy)),
            max(0, min(grid_size - 1, pos[1] + dx)),
        ]
        next_idx = state_idx(next_pos)
        ns_vec = state_to_vec(next_pos)

        # Compute intrinsic curiosity bonus
        r_i = float(module_instance.compute_intrinsic_reward(s_vec, action, ns_vec))
        
        # Update curiosity module
        module_instance.update(np.array([s_vec]), np.array([action]), np.array([ns_vec]))

        # Q-learning Bellman update driven purely by curiosity
        best_next_q = np.max(q_table[next_idx])
        target_q = r_i + gamma * best_next_q
        q_table[curr_idx, action] += lr * (target_q - q_table[curr_idx, action])

        pos = next_pos
        visited_states.add((pos[0], pos[1]))

    coverage_ratio = len(visited_states) / float(total_states)
    details = {
        "unique_visited_states": len(visited_states),
        "total_states": total_states,
        "coverage_ratio": coverage_ratio,
    }
    return float(coverage_ratio), details


def test_goal_reach_rate(module_instance) -> Tuple[float, Dict[str, Any]]:
    """
    Pillar 4: Goal Reach & Learning Efficiency on Sparse 15-Step Chain.
    
    External reward is +10.0 ONLY at the final state 14, and 0.0 elsewhere.
    Combined reward: r_total = r_extrinsic + 0.5 * r_intrinsic.
    Tests if curiosity accelerates discovery of the distant sparse goal over 25 episodes.
    """
    chain_length = 15
    n_episodes = 25
    max_steps = 40
    goals_reached = 0

    q_table = np.zeros((chain_length, 2), dtype=np.float64)  # 0=left, 1=right
    rng = np.random.default_rng(999)
    lr = 0.15
    gamma = 0.95
    epsilon = 0.2

    def to_vec(s):
        return np.array([s / float(chain_length), 0.5, 0.0, 0.0], dtype=np.float64)

    for ep in range(n_episodes):
        state = 0
        for st in range(max_steps):
            if state == chain_length - 1:
                goals_reached += 1
                break

            s_vec = to_vec(state)
            if rng.random() < epsilon:
                action = int(rng.integers(0, 2))
            else:
                action = int(np.argmax(q_table[state]))

            # Move: 0=left, 1=right
            next_state = max(0, min(chain_length - 1, state + (1 if action == 1 else -1)))
            ns_vec = to_vec(next_state)

            # Extrinsic reward
            r_ext = 10.0 if next_state == (chain_length - 1) else 0.0
            
            # Intrinsic curiosity reward
            r_int = float(module_instance.compute_intrinsic_reward(s_vec, action, ns_vec))
            
            # Total reward
            r_total = r_ext + (0.5 * r_int)

            # Update curiosity module
            module_instance.update(np.array([s_vec]), np.array([action]), np.array([ns_vec]))

            # Q-update
            best_q = np.max(q_table[next_state])
            q_table[state, action] += lr * (r_total + gamma * best_q - q_table[state, action])

            state = next_state

    reach_rate = goals_reached / float(n_episodes)
    details = {
        "goals_reached": goals_reached,
        "total_episodes": n_episodes,
        "reach_rate": reach_rate,
    }
    return float(reach_rate), details


def evaluate_stage1(program_path: str) -> EvaluationResult:
    """
    Stage 1 Quick Cascade Filter:
    Checks Habituation Decay and Noisy-TV Immunity in < 50 milliseconds.
    If the candidate fails basic sanity checks or crashes, terminates early.
    """
    set_global_seeds(42)
    start_time = time.perf_counter()
    metrics = {
        "combined_score": 0.0,
        "habituation_decay": 0.0,
        "noisy_tv_immunity": 0.0,
        "state_coverage": 0.0,
        "goal_reach_rate": 0.0,
        "execution_time": 0.0,
    }
    artifacts = {
        "failure_stage": None,
        "stderr": None,
        "traceback": None,
        "suggestion": None,
    }

    try:
        candidate = load_candidate_module(program_path)
        
        # Instantiate candidate module
        if hasattr(candidate, "create_curiosity_module"):
            module = candidate.create_curiosity_module(state_dim=4, action_dim=2)
        elif hasattr(candidate, "CuriosityRewardModule"):
            module = candidate.CuriosityRewardModule(state_dim=4, action_dim=2)
        else:
            artifacts["failure_stage"] = "stage1_interface_contract"
            artifacts["stderr"] = "Missing 'create_curiosity_module' or 'CuriosityRewardModule' class."
            artifacts["suggestion"] = "Define 'create_curiosity_module(state_dim, action_dim)' returning an instance."
            return EvaluationResult(metrics=metrics, artifacts=artifacts)

        # Quick Test 1: Habituation
        decay_score, hab_details = test_habituation_decay(module)
        metrics["habituation_decay"] = decay_score

        # Reset or re-instantiate for clean Noisy-TV test
        if hasattr(candidate, "create_curiosity_module"):
            module_noise = candidate.create_curiosity_module(state_dim=4, action_dim=2)
        else:
            module_noise = candidate.CuriosityRewardModule(state_dim=4, action_dim=2)

        # Quick Test 2: Noisy TV
        noise_score, noise_details = test_noisy_tv_immunity(module_noise)
        metrics["noisy_tv_immunity"] = noise_score

        # Stage 1 combined score
        stage1_score = 0.5 * decay_score + 0.5 * noise_score
        metrics["combined_score"] = float(stage1_score)
        metrics["execution_time"] = float(time.perf_counter() - start_time)

        artifacts["stage1_habituation"] = hab_details
        artifacts["stage1_noisy_tv"] = noise_details

        return EvaluationResult(metrics=metrics, artifacts=artifacts)

    except Exception as e:
        metrics["execution_time"] = float(time.perf_counter() - start_time)
        artifacts["failure_stage"] = "stage1_exception"
        artifacts["stderr"] = str(e)
        artifacts["traceback"] = traceback.format_exc()
        artifacts["suggestion"] = "Check that compute_intrinsic_reward and update handle numpy array inputs properly."
        return EvaluationResult(metrics=metrics, artifacts=artifacts)


def evaluate(program_path: str) -> EvaluationResult:
    """
    Full 4-Pillar Evaluation for OpenEvolve.
    Runs comprehensive exploration benchmarks and computes multi-objective fitness.
    """
    set_global_seeds(42)
    start_time = time.perf_counter()
    metrics = {
        "combined_score": 0.0,
        "state_coverage": 0.0,
        "noisy_tv_immunity": 0.0,
        "habituation_decay": 0.0,
        "goal_reach_rate": 0.0,
        "execution_time": 0.0,
    }
    artifacts = {
        "failure_stage": None,
        "stderr": None,
        "traceback": None,
        "suggestion": None,
    }

    try:
        candidate = load_candidate_module(program_path)

        def make_module(s_dim=4, a_dim=2):
            if hasattr(candidate, "create_curiosity_module"):
                return candidate.create_curiosity_module(state_dim=s_dim, action_dim=a_dim)
            elif hasattr(candidate, "CuriosityRewardModule"):
                return candidate.CuriosityRewardModule(state_dim=s_dim, action_dim=a_dim)
            raise AttributeError("Candidate must provide 'create_curiosity_module' or 'CuriosityRewardModule'.")

        # 1. Habituation Decay
        mod1 = make_module(4, 2)
        hab_score, hab_details = test_habituation_decay(mod1)
        metrics["habituation_decay"] = hab_score
        artifacts["habituation_details"] = hab_details

        # 2. Noisy-TV Stochastic Immunity
        mod2 = make_module(4, 2)
        noise_score, noise_details = test_noisy_tv_immunity(mod2)
        metrics["noisy_tv_immunity"] = noise_score
        artifacts["noisy_tv_details"] = noise_details

        # 3. State-Space Coverage in Sparse Maze
        mod3 = make_module(4, 4)
        cov_score, cov_details = test_state_space_coverage(mod3)
        metrics["state_coverage"] = cov_score
        artifacts["state_coverage_details"] = cov_details

        # 4. Goal Reach Rate in Sparse Chain
        mod4 = make_module(4, 2)
        goal_score, goal_details = test_goal_reach_rate(mod4)
        metrics["goal_reach_rate"] = goal_score
        artifacts["goal_reach_details"] = goal_details

        # --- Comprehensive Fitness Formula ---
        # Weighting:
        # 30% State Coverage (breadth of exploration)
        # 25% Habituation Decay (boredom on familiar states)
        # 25% Noisy-TV Immunity (resistance to white noise traps)
        # 20% Goal Reach Rate (deep navigation speed)
        combined = (
            0.30 * cov_score +
            0.25 * hab_score +
            0.25 * noise_score +
            0.20 * goal_score
        )
        metrics["combined_score"] = float(max(0.0, min(1.0, combined)))
        metrics["execution_time"] = float(time.perf_counter() - start_time)

        # Construct helpful suggestions for next LLM mutation
        if noise_score < 0.5:
            artifacts["suggestion"] = (
                "Noisy-TV immunity is low. The module overvalues unpredictable noise. "
                "Consider projecting states through an inverse dynamics model or RND target to isolate learnable dynamics."
            )
        elif hab_score < 0.5:
            artifacts["suggestion"] = (
                "Habituation decay is sluggish. Repeated visits do not sufficiently reduce curiosity. "
                "Incorporate stronger inverse-frequency weighting or faster predictor gradient steps."
            )
        elif cov_score < 0.5:
            artifacts["suggestion"] = (
                "State-space coverage is limited. Try amplifying intrinsic bonuses for novel distant states "
                "or using entropy / variance-based exploration incentives."
            )
        else:
            artifacts["suggestion"] = "Solid baseline performance across all 4 pillars! Optimize computational efficiency and goal convergence speed."

        return EvaluationResult(metrics=metrics, artifacts=artifacts)

    except Exception as e:
        metrics["execution_time"] = float(time.perf_counter() - start_time)
        artifacts["failure_stage"] = "full_evaluation_exception"
        artifacts["stderr"] = str(e)
        artifacts["traceback"] = traceback.format_exc()
        artifacts["suggestion"] = "Ensure candidate functions return valid finite floats and handle edge cases gracefully."
        return EvaluationResult(metrics=metrics, artifacts=artifacts)


def evaluate_stage2(program_path: str) -> EvaluationResult:
    """Full evaluation alias for cascade stages."""
    return evaluate(program_path)


if __name__ == "__main__":
    target_path = sys.argv[1] if len(sys.argv) > 1 else "initial_program.py"
    print(f"Testing evaluator on: {target_path}")

    print("\n--- STAGE 1 QUICK CHECK ---")
    res1 = evaluate_stage1(target_path)
    print(f"Metrics: {res1.metrics}")
    print(f"Artifacts: {res1.artifacts}")

    print("\n--- FULL 4-PILLAR EVALUATION ---")
    res2 = evaluate(target_path)
    print(f"Metrics: {res2.metrics}")
    print(f"Artifacts: {res2.artifacts}")

    assert "combined_score" in res2.metrics, "Error: 'combined_score' missing from metrics!"
    print("\n✅ Evaluator validation successful!")
