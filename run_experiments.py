import sys
import subprocess
import time
import argparse

def run_command(cmd):
    print(f"\n{'='*70}\n[RUNNING]: {' '.join(cmd)}\n{'='*70}", flush=True)
    start = time.time()
    res = subprocess.run(cmd)
    elapsed = time.time() - start
    print(f"\n[DONE in {elapsed:.1f}s]: {' '.join(cmd)}\n", flush=True)
    return res.returncode

def main():
    parser = argparse.ArgumentParser(description="Run full ReAct benchmark on HotpotQA and FEVER.")
    parser.add_argument("--count", type=int, default=100, help="Number of samples per benchmark (default: 100)")
    args = parser.parse_args()

    count = args.count
    print("=" * 70, flush=True)
    print(f"STARTING FULL {count}-SAMPLE BENCHMARK FOR HOTPOTQA & FEVER", flush=True)
    print("=" * 70, flush=True)

    # 1. HotpotQA
    ret1 = run_command([
        sys.executable, "evaluate.py",
        "--task", "hotpotqa",
        "--count", str(count),
        "--output", f"results_hotpotqa_{count}.json"
    ])
    if ret1 != 0:
        print("[ERROR] HotpotQA evaluation failed!", flush=True)
        return ret1

    # 2. FEVER
    ret2 = run_command([
        sys.executable, "evaluate.py",
        "--task", "fever",
        "--count", str(count),
        "--output", f"results_fever_{count}.json"
    ])
    if ret2 != 0:
        print("[ERROR] FEVER evaluation failed!", flush=True)
        return ret2

    print("\n" + "=" * 70, flush=True)
    print(f"ALL {count}-SAMPLE EXPERIMENTS COMPLETED SUCCESSFULLY!", flush=True)
    print("=" * 70, flush=True)
    return 0

if __name__ == "__main__":
    sys.exit(main())
