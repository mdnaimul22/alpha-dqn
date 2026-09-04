"""
Empirical Latency Benchmark for Alpha-DQN.
Measures both:
  1. Pure In-Memory Python Neural Inference (agent.act)
  2. Full HTTP REST Network Round-Trip (POST /api/dqn/act via Uvicorn/FastAPI)
"""

import time
import numpy as np
import httpx

from src.schema.dqn import DQNConfig
from src.core.agent import DQNAgent

def benchmark_in_memory(iterations: int = 1000):
    cfg = DQNConfig(state_size=6, action_size=5, layers=[{"units": 64, "activation": "relu"}, {"units": 64, "activation": "relu"}])
    agent = DQNAgent(config=cfg)
    agent.init()
    
    state = [0.1, -0.2, 0.05, 0.0, -0.4, 0.3]
    
    # Warmup
    for _ in range(50):
        agent.act(state)
        
    latencies = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        agent.act(state)
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000.0) # in ms
        
    arr = np.array(latencies)
    return {
        "mean_ms": float(np.mean(arr)),
        "median_p50_ms": float(np.median(arr)),
        "p95_ms": float(np.percentile(arr, 95)),
        "p99_ms": float(np.percentile(arr, 99)),
        "min_ms": float(np.min(arr)),
        "max_ms": float(np.max(arr)),
        "mean_us": float(np.mean(arr) * 1000.0), # in microseconds
    }

def benchmark_http_api(iterations: int = 500):
    client = httpx.Client(base_url="http://127.0.0.1:8001", timeout=5.0)
    payload = {
        "session_id": "spaceship_live_session",
        "state": [0.1, -0.2, 0.05, 0.0, -0.4, 0.3],
        "force_pure_neural": False,
    }
    
    # Warmup
    for _ in range(20):
        client.post("/api/dqn/act", json=payload)
        
    latencies = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        resp = client.post("/api/dqn/act", json=payload)
        t1 = time.perf_counter()
        assert resp.status_code == 200
        latencies.append((t1 - t0) * 1000.0) # in ms
        
    arr = np.array(latencies)
    return {
        "mean_ms": float(np.mean(arr)),
        "median_p50_ms": float(np.median(arr)),
        "p95_ms": float(np.percentile(arr, 95)),
        "p99_ms": float(np.percentile(arr, 99)),
        "min_ms": float(np.min(arr)),
        "max_ms": float(np.max(arr)),
    }

if __name__ == "__main__":
    print("⏳ Benchmarking Pure In-Memory Inference (1,000 iterations)...")
    mem_stats = benchmark_in_memory(1000)
    
    print("⏳ Benchmarking Full HTTP REST API Round-Trip (500 iterations)...")
    http_stats = benchmark_http_api(500)
    
    print("\n" + "="*60)
    print("📊 EMPIRICAL LATENCY BENCHMARK RESULTS")
    print("="*60)
    print("1️⃣ PURE IN-MEMORY (DIRECT PYTHON SDK):")
    print(f"   • Mean Latency:       {mem_stats['mean_ms']:.4f} ms ({mem_stats['mean_us']:.1f} µs)")
    print(f"   • Median (P50):       {mem_stats['median_p50_ms']:.4f} ms")
    print(f"   • P95 Latency:        {mem_stats['p95_ms']:.4f} ms")
    print(f"   • P99 Latency:        {mem_stats['p99_ms']:.4f} ms")
    print(f"   • Min Latency:        {mem_stats['min_ms']:.4f} ms")
    print(f"   • Throughput:         ~{int(1000.0 / mem_stats['mean_ms']):,} inferences/sec")
    
    print("\n2️⃣ HTTP REST API (LOCALHOST POST /api/dqn/act):")
    print(f"   • Mean Latency:       {http_stats['mean_ms']:.4f} ms")
    print(f"   • Median (P50):       {http_stats['median_p50_ms']:.4f} ms")
    print(f"   • P95 Latency:        {http_stats['p95_ms']:.4f} ms")
    print(f"   • P99 Latency:        {http_stats['p99_ms']:.4f} ms")
    print(f"   • Min Latency:        {http_stats['min_ms']:.4f} ms")
    print(f"   • Throughput:         ~{int(1000.0 / http_stats['mean_ms']):,} requests/sec")
    print("="*60)
