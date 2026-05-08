"""
run_all.py — Run all Day 22 lab steps sequentially (or a specific step)
========================================================================
Usage:
    python run_all.py          # run all 4 steps
    python run_all.py --step 3 # run only step 3
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

STEPS = {
    1: ("01_langsmith_rag_pipeline.py",  "LangSmith RAG Pipeline"),
    2: ("02_prompt_hub_ab_routing.py",   "Prompt Hub & A/B Routing"),
    3: ("03_ragas_evaluation.py",        "RAGAS Evaluation (~15-20 min)"),
    4: ("04_guardrails_validator.py",    "Guardrails AI Validators"),
}

PYTHON = sys.executable


def run_step(step_num: int) -> bool:
    script, label = STEPS[step_num]
    script_path = Path(__file__).parent / script
    if not script_path.exists():
        print(f"  ❌ Script not found: {script}")
        return False

    print(f"\n{'=' * 60}")
    print(f"  Step {step_num}: {label}")
    print(f"{'=' * 60}")
    t0 = time.time()

    result = subprocess.run([PYTHON, str(script_path)], check=False)

    elapsed = time.time() - t0
    if result.returncode == 0:
        print(f"\n  ✅ Step {step_num} completed in {elapsed:.1f}s")
        return True
    else:
        print(f"\n  ❌ Step {step_num} failed (exit code {result.returncode}) after {elapsed:.1f}s")
        return False


def main():
    parser = argparse.ArgumentParser(description="Run Day 22 lab steps")
    parser.add_argument("--step", type=int, choices=STEPS.keys(),
                        help="Run only this step number (1-4)")
    args = parser.parse_args()

    total_start = time.time()

    if args.step:
        steps_to_run = [args.step]
    else:
        steps_to_run = list(STEPS.keys())

    results = {}
    for s in steps_to_run:
        results[s] = run_step(s)

    print(f"\n{'=' * 60}")
    print("  Final Summary")
    print(f"{'=' * 60}")
    for s, ok in results.items():
        icon = "✅" if ok else "❌"
        print(f"  {icon} Step {s}: {STEPS[s][1]}")
    total_elapsed = time.time() - total_start
    print(f"\n  Total time: {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)")
    print(f"{'=' * 60}")

    if not all(results.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()
