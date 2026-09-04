"""
Validation script for OpenEvolve Curiosity Evaluator.
Verifies that evaluator.py can correctly load and score initial_program.py.
"""

import json
import os
import sys
from evaluator import evaluate, evaluate_stage1


def run_test():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    program_path = os.path.join(current_dir, "initial_program.py")

    print("=" * 60)
    print("🛡️  OPENEVOLVE EVALUATOR VALIDATION (CURIOSITY MODULE)")
    print("=" * 60)

    if not os.path.exists(program_path):
        print(f"❌ Error: {program_path} not found.")
        sys.exit(1)

    print(f"[*] Testing candidate: {os.path.basename(program_path)}")

    try:
        # 1. Stage 1 Quick Check
        print("\n--- STAGE 1 QUICK CHECK ---")
        res1 = evaluate_stage1(program_path)
        print(f"Score: {res1.metrics.get('combined_score', 0):.4f} in {res1.metrics.get('execution_time', 0):.3f}s")
        print(f"Metrics: {json.dumps(res1.metrics, indent=2)}")

        # 2. Stage 2 Full 4-Pillar Evaluation
        print("\n--- FULL 4-PILLAR EVALUATION ---")
        res2 = evaluate(program_path)
        score = res2.metrics.get("combined_score", 0.0)
        print(f"Combined Score: {score:.4f} in {res2.metrics.get('execution_time', 0):.3f}s")
        print("\n📊 Detailed Pillar Metrics:")
        for k, v in res2.metrics.items():
            print(f"  - {k}: {v}")

        print("\n🎨 Diagnostic Artifacts & Suggestions:")
        print(json.dumps(res2.artifacts, indent=2))

        print("\n" + "=" * 60)
        if score > 0.0:
            print("✅ SUCCESS: Evaluator is operational and produces valid fitness signals.")
            print(f"Baseline combined fitness: {score:.4f}")
        else:
            print("⚠️ WARNING: Evaluator returned 0.0 score.")
        print("=" * 60)

    except Exception as e:
        print(f"\n❌ CRITICAL FAILURE: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    run_test()
