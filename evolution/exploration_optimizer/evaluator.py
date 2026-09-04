"""
OpenEvolve Evaluator for Novel Exploration Strategy Optimization in Reinforcement Learning.

Evaluates candidate ExplorationPolicyModule implementations against 4 core pillars:
1. Deep Diffusion Speed (30%): Rate of escaping bottlenecks and reaching deep distant horizons without wandering.
2. State Visitation Entropy (30%): Uniformity and breadth of state-space coverage in a 6x6 grid world.
3. Action Coherence & Anti-Dithering (20%): Temporally extended action commitment without getting trapped.
4. Exploitation Annihilation (20%): Policy precision in exploiting clear, confident optimal actions.

MAP-Elites Dimensions:
- diffusion_speed: Rate of reaching distant frontiers [0.0, 1.0]
- state_entropy: Shannon entropy of state visitation distribution [0.0, 1.0]
"""

from __future__ import annotations

import importlib.util
import os
import random
import sys
import time
import traceback
from typing import Any, Dict, Tuple

import numpy as np
from openevolve.evaluation_result import EvaluationResult


def set_global_seeds(seed: int = 42):
    """Enforce strict deterministic reproducibility."""
    random.seed(seed)
    np.random.seed(seed)


def load_candidate_module(program_path: str):
    """Safely and dynamically load candidate exploration module from file path."""
    module_name = os.path.basename(program_path).replace(".py", "")
    spec = importlib.util.spec_from_file_location(module_name, program_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module specification from {program_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_diffusion_speed(policy_module) -> Tuple[float, Dict[str, Any]]:
    """
    Pillar 1: Deep Diffusion Speed in a 20-Step Corridor.
    Measures how quickly the policy reaches the opposite end of a 20-step chain.
    Actions: 0=Left, 1=Right, 2=No-Op.
    """
    set_global_seeds(42)
    corridor_len = 20
    max_steps_per_ep = 60
    num_episodes = 8
    target_pos = corridor_len - 1

    steps_to_reach = []
    max_distances = []

    for ep in range(num_episodes):
        policy_module.reset_episode()
        pos = 0
        reached = False

        for step in range(max_steps_per_ep):
            state = np.array([pos / float(target_pos), (target_pos - pos) / float(target_pos)])
            # Flat/uncertain Q-values simulating uninformative early exploration
            q_values = np.zeros(3)

            action = policy_module.select_action(q_values, state, step=step, epsilon=0.3)
            prev_pos = pos

            if action == 1:
                pos = min(target_pos, pos + 1)
            elif action == 0:
                pos = max(0, pos - 1)

            policy_module.update(state, action, 0.0, state, False)

            if pos == target_pos:
                steps_to_reach.append(step + 1)
                reached = True
                break

        max_distances.append(pos)
        if not reached:
            steps_to_reach.append(max_steps_per_ep * 2)

    avg_steps = float(np.mean(steps_to_reach))
    avg_max_dist = float(np.mean(max_distances))

    # Score: Theoretical minimum is 19 steps. Random walk takes ~200 steps.
    dist_ratio = avg_max_dist / float(target_pos)
    speed_factor = 19.0 / max(19.0, avg_steps)
    diffusion_score = float(np.clip(0.6 * dist_ratio + 0.4 * speed_factor, 0.0, 1.0))

    details = {
        "avg_steps_to_frontier": round(avg_steps, 2),
        "avg_max_distance_reached": round(avg_max_dist, 2),
        "episodes_reached_frontier": int(sum(1 for s in steps_to_reach if s <= max_steps_per_ep)),
    }
    return diffusion_score, details


def test_state_visitation_entropy(policy_module) -> Tuple[float, Dict[str, Any]]:
    """
    Pillar 2: State Visitation Entropy in a 6x6 Grid World (36 States).
    Measures Shannon entropy of visited cells across 180 environment steps.
    Actions: 0=Up, 1=Right, 2=Down, 3=Left.
    """
    set_global_seeds(123)
    policy_module.reset()

    grid_size = 6
    total_cells = grid_size * grid_size
    counts = np.zeros((grid_size, grid_size), dtype=np.int32)

    r, c = 0, 0
    total_steps = 180

    for step in range(total_steps):
        counts[r, c] += 1
        state = np.array([r / float(grid_size - 1), c / float(grid_size - 1), 0.0, 0.0])
        # Neutral Q-values
        q_values = np.zeros(4)

        action = policy_module.select_action(q_values, state, step=step, epsilon=0.25)

        nr, nc = r, c
        if action == 0:
            nr = max(0, r - 1)
        elif action == 1:
            nc = min(grid_size - 1, c + 1)
        elif action == 2:
            nr = min(grid_size - 1, r + 1)
        elif action == 3:
            nc = max(0, c - 1)

        policy_module.update(state, action, 0.0, state, False)
        r, c = nr, nc

    # Calculate normalized Shannon Entropy
    probs = counts.flatten() / float(total_steps)
    nonzero = probs[probs > 0]
    shannon_entropy = -float(np.sum(nonzero * np.log2(nonzero)))
    max_entropy = float(np.log2(total_cells))  # log2(36) ~ 5.1699

    entropy_score = float(np.clip(shannon_entropy / max_entropy, 0.0, 1.0))
    unique_cells_visited = int(np.sum(counts > 0))

    details = {
        "unique_cells_visited": unique_cells_visited,
        "shannon_entropy": round(shannon_entropy, 4),
        "max_possible_entropy": round(max_entropy, 4),
        "entropy_coverage_ratio": round(entropy_score, 4),
    }
    return entropy_score, details


def test_action_coherence(policy_module) -> Tuple[float, Dict[str, Any]]:
    """
    Pillar 3: Action Coherence & Anti-Dithering.
    Measures action persistence (intentional sequences) vs jittery random dithering.
    """
    set_global_seeds(777)
    policy_module.reset()

    actions_taken = []
    dummy_state = np.array([0.5, 0.5])
    q_flat = np.zeros(4)

    for step in range(120):
        a = policy_module.select_action(q_flat, dummy_state, step=step, epsilon=0.3)
        actions_taken.append(a)
        policy_module.update(dummy_state, a, 0.0, dummy_state, False)

    # Lag-1 action autocorrelation / persistence
    pairs_same = sum(1 for i in range(len(actions_taken) - 1) if actions_taken[i] == actions_taken[i + 1])
    persistence_ratio = pairs_same / float(len(actions_taken) - 1)

    # Opposite oscillation count (immediate dithering back and forth: e.g. 0 -> 2 -> 0 or 1 -> 3 -> 1)
    dither_count = 0
    for i in range(len(actions_taken) - 2):
        a1, a2, a3 = actions_taken[i], actions_taken[i + 1], actions_taken[i + 2]
        if (a1 == 0 and a2 == 2 and a3 == 0) or (a1 == 1 and a2 == 3 and a3 == 1):
            dither_count += 1
        elif (a1 == 2 and a2 == 0 and a3 == 2) or (a1 == 3 and a2 == 1 and a3 == 3):
            dither_count += 1

    dither_ratio = dither_count / float(max(1, len(actions_taken) - 2))

    # Ideal persistence is in range [0.30, 0.65]. Too high (>0.85) means frozen repetition.
    if persistence_ratio < 0.20:
        coherence_base = 0.5  # purely random dithering
    elif persistence_ratio > 0.85:
        coherence_base = 0.4  # frozen policy
    else:
        coherence_base = 0.8 + 0.2 * (1.0 - abs(persistence_ratio - 0.45) / 0.35)

    coherence_score = float(np.clip(coherence_base - 1.5 * dither_ratio, 0.0, 1.0))

    details = {
        "persistence_ratio": round(persistence_ratio, 4),
        "dither_ratio": round(dither_ratio, 4),
        "coherence_score": round(coherence_score, 4),
    }
    return coherence_score, details


def test_exploitation_annihilation(policy_module) -> Tuple[float, Dict[str, Any]]:
    """
    Pillar 4: Exploitation Precision & Uncertainty Annihilation.
    When Q-values clearly distinguish the optimal action with confidence,
    the policy must select the optimal action with high reliability (>= 90%).
    """
    set_global_seeds(999)
    policy_module.reset()

    correct_selections = 0
    total_trials = 60

    for i in range(total_trials):
        optimal_action = i % 4
        # Synthetic Q-vector where one action has a clear advantage (+3.5 spread)
        q = np.ones(4) * 0.5
        q[optimal_action] = 4.0

        state = np.random.uniform(-1.0, 1.0, size=4)
        chosen = policy_module.select_action(q, state, step=i, epsilon=0.1)

        if chosen == optimal_action:
            correct_selections += 1

    accuracy = correct_selections / float(total_trials)
    score = float(np.clip(accuracy, 0.0, 1.0))

    details = {
        "optimal_action_accuracy": round(accuracy, 4),
        "correct_trials": correct_selections,
        "total_trials": total_trials,
    }
    return score, details


def evaluate_stage1(program_path: str) -> EvaluationResult:
    """Stage 1 Quick Check."""
    set_global_seeds(42)
    start_time = time.perf_counter()
    metrics: Dict[str, float] = {}
    artifacts: Dict[str, Any] = {}

    try:
        candidate = load_candidate_module(program_path)
        if hasattr(candidate, "create_exploration_module"):
            mod = candidate.create_exploration_module(state_dim=4, action_dim=3, seed=42)
        elif hasattr(candidate, "ExplorationPolicyModule"):
            mod = candidate.ExplorationPolicyModule(state_dim=4, action_dim=3, seed=42)
        else:
            raise AttributeError("Candidate must define 'create_exploration_module' or 'ExplorationPolicyModule'")

        q = np.array([0.1, 1.5, 0.2])
        s = np.array([0.0, 0.0, 0.0, 0.0])
        a = mod.select_action(q, s, step=0, epsilon=0.1)

        if not (0 <= a < 3):
            return EvaluationResult(metrics={"combined_score": 0.0}, artifacts={"error": "Invalid action returned"})

        metrics["stage1_score"] = 1.0
        metrics["combined_score"] = 0.5
        metrics["execution_time"] = float(time.perf_counter() - start_time)
        return EvaluationResult(metrics=metrics, artifacts=artifacts)

    except Exception as e:
        metrics["combined_score"] = 0.0
        metrics["execution_time"] = float(time.perf_counter() - start_time)
        artifacts["stderr"] = str(e)
        artifacts["traceback"] = traceback.format_exc()
        return EvaluationResult(metrics=metrics, artifacts=artifacts)


def evaluate(program_path: str) -> EvaluationResult:
    """Full 4-Pillar Evaluation of Candidate Exploration Policy."""
    set_global_seeds(42)
    start_time = time.perf_counter()
    metrics: Dict[str, float] = {}
    artifacts: Dict[str, Any] = {}

    try:
        candidate = load_candidate_module(program_path)
        if hasattr(candidate, "create_exploration_module"):
            make_mod = candidate.create_exploration_module
        elif hasattr(candidate, "ExplorationPolicyModule"):
            make_mod = lambda s_dim, a_dim, seed=42: candidate.ExplorationPolicyModule(s_dim, a_dim, seed=seed)
        else:
            raise AttributeError("Candidate must define 'create_exploration_module' or 'ExplorationPolicyModule'")

        # 1. Deep Diffusion Speed (30%)
        mod1 = make_mod(2, 3, seed=42)
        diff_score, diff_details = test_diffusion_speed(mod1)
        metrics["diffusion_speed"] = diff_score
        artifacts["diffusion_details"] = diff_details

        # 2. State Visitation Entropy (30%)
        mod2 = make_mod(4, 4, seed=123)
        ent_score, ent_details = test_state_visitation_entropy(mod2)
        metrics["state_entropy"] = ent_score
        artifacts["entropy_details"] = ent_details

        # 3. Action Coherence & Anti-Dithering (20%)
        mod3 = make_mod(2, 4, seed=777)
        coh_score, coh_details = test_action_coherence(mod3)
        metrics["action_coherence"] = coh_score
        artifacts["coherence_details"] = coh_details

        # 4. Exploitation Annihilation (20%)
        mod4 = make_mod(4, 4, seed=999)
        exp_score, exp_details = test_exploitation_annihilation(mod4)
        metrics["exploit_accuracy"] = exp_score
        artifacts["exploit_details"] = exp_details

        # Combined Fitness Score [0.0 - 1.0]
        combined = (
            0.30 * diff_score +
            0.30 * ent_score +
            0.20 * coh_score +
            0.20 * exp_score
        )
        metrics["combined_score"] = float(np.clip(combined, 0.0, 1.0))
        metrics["execution_time"] = float(time.perf_counter() - start_time)

        return EvaluationResult(metrics=metrics, artifacts=artifacts)

    except Exception as e:
        metrics["combined_score"] = 0.0
        metrics["execution_time"] = float(time.perf_counter() - start_time)
        artifacts["stderr"] = str(e)
        artifacts["traceback"] = traceback.format_exc()
        return EvaluationResult(metrics=metrics, artifacts=artifacts)


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "initial_program.py"
    print(f"Testing exploration evaluator on: {target}")
    res = evaluate(target)
    print("\n--- EVALUATION RESULTS ---")
    print("Metrics:", res.metrics)
    print("Artifacts:", res.artifacts)
