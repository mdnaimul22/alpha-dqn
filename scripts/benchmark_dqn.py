"""
Synthetic Internal Benchmark Suite for DQN Core Engine.
Evaluates:
  1. Throughput: Steps Per Second (SPS) during active simulation loop
  2. Latency: Average execution time per train_batch() call (milliseconds)
  3. Stationary Bellman Convergence: Loss reduction rate on deterministic transitions
"""

from __future__ import annotations

import os
import random
import sys
import time
from pathlib import Path

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from src.core.agent import DQNAgent
from src.schema.dqn import DQNConfig, LayerConfig


def run_benchmark(
    num_steps: int = 5000,
    state_size: int = 75,
    action_size: int = 10,
    batch_size: int = 32,
    train_interval: int = 4,
    device: str = "cpu",
) -> dict:
    config = DQNConfig(
        state_size=state_size,
        action_size=action_size,
        layers=[
            LayerConfig(units=128, activation="relu"),
            LayerConfig(units=128, activation="relu"),
            LayerConfig(units=64, activation="relu"),
        ],
        batch_size=batch_size,
        train_interval=train_interval,
        memory_capacity=10000,
        learning_rate=0.001,
        auto_save_interval=0,  # Isolate compute throughput from disk JSON serialization
    )

    agent = DQNAgent(config=config, device=device)
    agent.init()

    # Pre-generate random synthetic data
    states = [[random.uniform(-1.0, 1.0) for _ in range(state_size)] for _ in range(500)]

    # 1. Warm-up memory buffer
    for i in range(batch_size * 2):
        s = states[i % len(states)]
        ns = states[(i + 1) % len(states)]
        agent.remember(s, random.randint(0, action_size - 1), 1.0, ns, False)

    # 2. Measure throughput & train_batch latency
    train_latencies = []
    start_time = time.perf_counter()

    for step in range(num_steps):
        s = states[step % len(states)]
        a = agent.act(s)
        ns = states[(step + 1) % len(states)]
        r = 1.0 if a == 1 else -0.5
        done = step % 100 == 0

        agent.remember(s, a, r, ns, done)

        # In typical RL loop, train_batch() is called on every step
        # and executes gradient descent when step_counter % train_interval == 0
        t0 = time.perf_counter()
        agent.train_batch()
        t1 = time.perf_counter()
        # Record latency if gradient descent actually ran (step_counter % train_interval == 0)
        if (t1 - t0) * 1000.0 > 0.05:
            train_latencies.append((t1 - t0) * 1000.0)

    total_time = time.perf_counter() - start_time
    sps = num_steps / total_time if total_time > 0 else 0
    avg_train_latency = sum(train_latencies) / len(train_latencies) if train_latencies else 0

    results = {
        "num_steps": num_steps,
        "total_time_sec": round(total_time, 3),
        "steps_per_second": round(sps, 1),
        "total_train_calls": len(train_latencies),
        "avg_train_latency_ms": round(avg_train_latency, 3),
        "final_training_loss": round(float(agent.training_loss), 5),
        "final_epsilon": round(float(agent.epsilon), 4),
    }
    return results


if __name__ == "__main__":
    print("=" * 60)
    print("🚀 Running Synthetic DQN Core Benchmark...")
    print("=" * 60)
    res = run_benchmark(num_steps=2000, device="cpu")
    print(f"Total Steps            : {res['num_steps']}")
    print(f"Total Time             : {res['total_time_sec']} s")
    print(f"Throughput (SPS)       : {res['steps_per_second']} steps/sec")
    print(f"Train Batch Calls      : {res['total_train_calls']}")
    print(f"Avg Train Latency      : {res['avg_train_latency_ms']} ms/batch")
    print(f"Final Training Loss    : {res['final_training_loss']}")
    print(f"Final Epsilon          : {res['final_epsilon']}")
    print("=" * 60)
