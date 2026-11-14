"""Evaluate newly registered headline seeds without touching frozen artifacts."""
from __future__ import annotations

import hashlib, json, os, subprocess, sys
from datetime import datetime, timezone
from pathlib import Path

from camo_build.protocol import verify_protocol

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "runs" / "full_cloud_validation"
EVAL = ROOT / "evaluate_camo_full_cloud.py"
SEEDS = (4409, 5513)
VOTES = (1103, 2207, 3301)

def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()

def ckpt(arch: str, seed: int) -> Path:
    if arch == "s5": return ROOT / "runs" / f"s5_row_softmax_p4096_seed{seed}" / "best.pth"
    return ROOT / "checkpoints" / f"best_camo_p15_density_bce_lovasz0.5_p4096_seed{seed}.pth"

def main() -> None:
    v = verify_protocol(ROOT, ROOT / "protocol" / "s0")
    if not v["passed"]: raise RuntimeError(v)
    eh = sha256(EVAL)
    for arch in ("s5", "s2"):
        for seed in SEEDS:
            c = ckpt(arch, seed)
            if not c.is_file(): raise FileNotFoundError(c)
            out = OUT / f"{arch}_seed{seed}.json"
            if out.exists(): continue
            log = OUT / "logs" / f"{arch}_seed{seed}.log"
            log.parent.mkdir(parents=True, exist_ok=True)
            cmd = [sys.executable, str(EVAL), "--architecture", arch,
                   "--checkpoint", str(c), "--output", str(out),
                   "--training-seed", str(seed)]
            with log.open("w", encoding="utf-8", buffering=1) as f:
                r = subprocess.run(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT,
                                   env=os.environ | {"PYTHONUNBUFFERED":"1"})
            if r.returncode != 0: raise RuntimeError(f"{arch} seed {seed} failed; see {log}")
            report = json.loads(out.read_text(encoding="utf-8"))
            if report.get("evaluator_sha256") != eh or report.get("checkpoint_sha256") != sha256(c):
                raise RuntimeError(f"integrity mismatch: {out}")
            if report.get("vote_seeds") != list(VOTES) or report.get("dry_run") is not False:
                raise RuntimeError(f"protocol mismatch: {out}")
    print("extension full-cloud evaluation complete")

if __name__ == "__main__": main()
