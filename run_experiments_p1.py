"""Runner script for CAMO-Build Prototype 1 full attribution experiments (Runbook §8 & §9)."""
from __future__ import annotations
import subprocess
import sys
import time
from pathlib import Path
from datetime import datetime

EXPERIMENTS = [
    # mode, seed, benchmark_file
    ("none", 42, "benchmark_camo_none.json"),
    ("none", 123, "benchmark_camo_none.json"),
    ("none", 2026, "benchmark_camo_none.json"),
    ("density", 42, "benchmark_camo_density.json"),
    ("density", 123, "benchmark_camo_density.json"),
    ("density", 2026, "benchmark_camo_density.json"),
]

def main():
    root = Path(__file__).resolve().parent
    progress_file = root / "logs" / "experiments_progress.txt"
    progress_file.parent.mkdir(exist_ok=True)
    
    print(f"[{datetime.now().isoformat()}] Starting Prototype 1 Full Experiments ({len(EXPERIMENTS)} runs, 50 epochs each)")
    with progress_file.open("a", encoding="utf-8") as f:
        f.write(f"\n--- Batch Started at {datetime.now().isoformat()} ---\n")
        
    for idx, (mode, seed, bench) in enumerate(EXPERIMENTS, 1):
        cmd = [
            sys.executable,
            "train_camo_p1.py",
            "--benchmark-report", bench,
            "--mass-mode", mode,
            "--epochs", "50",
            "--seed", str(seed),
        ]
        start_t = time.time()
        msg_start = f"[{datetime.now().isoformat()}] [{idx}/{len(EXPERIMENTS)}] RUNNING: mode={mode}, seed={seed} (cmd: {' '.join(cmd)})"
        print(msg_start, flush=True)
        with progress_file.open("a", encoding="utf-8") as f:
            f.write(msg_start + "\n")
            f.flush()

        res = subprocess.run(cmd, cwd=str(root))
        elapsed = time.time() - start_t
        msg_end = f"[{datetime.now().isoformat()}] [{idx}/{len(EXPERIMENTS)}] FINISHED: mode={mode}, seed={seed} (exit={res.returncode}, elapsed={elapsed/60:.2f}m)"
        print(msg_end, flush=True)
        with progress_file.open("a", encoding="utf-8") as f:
            f.write(msg_end + "\n")
            f.flush()

        if res.returncode != 0:
            print(f"ERROR in run {idx}! Aborting remaining runs.", flush=True)
            sys.exit(res.returncode)

    print(f"[{datetime.now().isoformat()}] ALL EXPERIMENTS COMPLETED SUCCESSFULLY!")
    with progress_file.open("a", encoding="utf-8") as f:
        f.write(f"--- All Completed at {datetime.now().isoformat()} ---\n")

if __name__ == "__main__":
    main()
