"""
BatchRunHMM_Adaptive_Subfolders.py
Two-phase run:
  1) Primer pass: run one replicate for every condition (if needed).
  2) Adaptive pass: continue reps until convergence or MAX_REPS.
"""

import subprocess
import itertools
import math
import statistics
import time
from pathlib import Path
import pandas as pd
import os
import gc
import shutil  # for removing incomplete rep folders

# ---------------- Configuration ----------------
ROOT = Path(__file__).resolve().parent
HMM_SCRIPT = ROOT / "HMM.py"
OUTPUT_ROOT = ROOT / "outputs"
OUTPUT_ROOT.mkdir(exist_ok=True)

sample_size = "1k"
rounds = [f"{r}r" for r in range(11, 14)]
specificities = ["verylow"]
binders = ["5aas", "10aas", "15aas", "20aas"]
no_error = "false"

TARGET_PRECISION = 0.05   # stop when 95% CI half-width <= 5% of mean
MAX_REPS = 10
MIN_REPS = 9  # minimum reps before checking convergence

# --- Cleanup Configuration ---
DELETE_FRAGMENT_DETAILS = True
FRAGMENT_DETAILS_NAME = "fragment_details.csv"
SUCCESS_MARKER = "statistics.csv"  # use this to detect complete reps

# ---------------- Helpers ----------------
def mean_confidence_interval(data, confidence=0.95):
    n = len(data)
    if n < 2:
        return (data[0] if n == 1 else float("nan")), float("inf")
    mean = statistics.mean(data)
    sd = statistics.stdev(data)
    se = sd / math.sqrt(n)
    h = se * 1.96  # 95% CI half-width
    return mean, h

def get_condition_results(summary_csv, sample_size, round_tag, binder, spec, no_error):
    if not summary_csv.exists():
        return []
    usecols = ["sample_size", "round", "binder", "specificity", "no_error", "correct"]
    dtypes  = {
        "sample_size": "category",
        "round": "category",
        "binder": "category",
        "specificity": "category",
        "no_error": "category",
        "correct": "float64",
    }
    df = pd.read_csv(summary_csv, usecols=usecols, dtype=dtypes)
    mask = (
        (df["sample_size"] == sample_size)
        & (df["round"] == round_tag)
        & (df["binder"] == binder)
        & (df["specificity"] == spec)
        & (df["no_error"].astype(str) == str(no_error))
    )
    subset = df.loc[mask, "correct"].to_numpy()
    del df
    gc.collect()
    return subset.tolist()

def cleanup_fragment_details(folder: Path, filename: str = FRAGMENT_DETAILS_NAME):
    """
    Delete all fragment_details.csv files inside folder (recursively).
    """
    count = 0
    for p in folder.rglob(filename):
        try:
            p.unlink(missing_ok=True)
            count += 1
        except Exception:
            pass
    if count > 0:
        print(f"  🧹 Deleted {count} '{filename}' file(s) under {folder}")
    return count

def prune_incomplete_reps(folder: Path, marker: str = SUCCESS_MARKER):
    """
    Remove any rep* directory under `folder` that does NOT contain the success marker (statistics.csv).
    Returns (deleted_count, deleted_paths).
    """
    deleted = []
    if not folder.exists():
        return 0, deleted

    # Only look at immediate rep* subfolders under each condition dir. But also allow a global call at OUTPUT_ROOT.
    for condition_dir in ([folder] if folder.name.startswith(("5aas","10aas","15aas","20aas")) else folder.iterdir()):
        if not condition_dir.is_dir():
            continue
        # inside a condition dir:  .../binder_spec_round/repN/
        rep_dirs = [p for p in condition_dir.iterdir() if p.is_dir() and p.name.startswith("rep")]
        for rep in rep_dirs:
            marker_path = rep / marker
            if not marker_path.exists():
                try:
                    shutil.rmtree(rep)
                    deleted.append(rep)
                except Exception:
                    # non-fatal; leave it and continue
                    pass
    if deleted:
        print(f"  🗑️  Pruned {len(deleted)} incomplete rep folder(s) (missing {marker}).")
    return len(deleted), deleted

def existing_rep_indices(condition_dir: Path):
    return sorted(
        [int(p.name.replace("rep", "")) for p in condition_dir.iterdir() if p.is_dir() and p.name.startswith("rep")]
    )

def run_one_rep(round_tag, spec, binder, base_env):
    """
    Runs a single replicate for the given (round, spec, binder).
    Returns:
      - True if the rep completed and produced SUCCESS_MARKER
      - False if failed
      - 'skipped' if already converged or already has >= 1 rep (used during primer pass)
    """
    condition_name = f"{binder}_{spec}_{round_tag}"
    condition_dir = OUTPUT_ROOT / condition_name
    condition_dir.mkdir(exist_ok=True)

    # Per-condition prune of incomplete reps (e.g., from past failed runs)
    prune_incomplete_reps(condition_dir)

    # Remove any leftover fragment_details for this condition
    if DELETE_FRAGMENT_DETAILS:
        cleanup_fragment_details(condition_dir)

    summary_csv = OUTPUT_ROOT / "summary.csv"
    all_correct = get_condition_results(summary_csv, sample_size, round_tag, binder, spec, no_error)
    reps = existing_rep_indices(condition_dir)

    # If converged already, skip
    if len(all_correct) >= MIN_REPS:
        mean_acc, half_width = mean_confidence_interval(all_correct)
        if half_width <= TARGET_PRECISION * mean_acc:
            print(f"  {condition_name}: already converged ({len(all_correct)} runs, mean={mean_acc:.4f} ±{half_width:.4f})")
            return 'skipped'

    # Pick next rep index
    next_rep = (max(reps) + 1) if reps else 1
    rep_dir = condition_dir / f"rep{next_rep}"
    rep_dir.mkdir(exist_ok=True)

    print(f"  {condition_name}: Run rep {next_rep} ...")
    log_path = rep_dir / "run.log"
    with open(log_path, "wb") as logf:
        cmd = [
            "python", str(HMM_SCRIPT),
            "--sample_size", sample_size,
            "--round", round_tag,
            "--binder", binder,
            "--specificity", spec,
            "--output", str(rep_dir),
            "--no_error", no_error
        ]
        start = time.time()
        try:
            subprocess.run(
                cmd,
                check=True,
                stdout=logf,
                stderr=subprocess.STDOUT,
                env=base_env
            )
        except subprocess.CalledProcessError as e:
            print(f"  {condition_name}: run failed: {e}")
            try:
                shutil.rmtree(rep_dir)
                print(f"  ❌ Removed incomplete folder: {rep_dir}")
            except Exception:
                print(f"  ⚠️ Could not remove {rep_dir}; please delete manually if needed.")
            return False

        runtime = time.time() - start

    # Post-run cleanup
    if DELETE_FRAGMENT_DETAILS:
        cleanup_fragment_details(rep_dir)

    # Ensure SUCCESS_MARKER exists; if not, prune this rep
    if not (rep_dir / SUCCESS_MARKER).exists():
        try:
            shutil.rmtree(rep_dir)
            print(f"  ⚠️ Missing {SUCCESS_MARKER}. Removed incomplete folder: {rep_dir}")
            return False
        except Exception:
            print(f"  ⚠️ Missing {SUCCESS_MARKER} and could not remove {rep_dir}.")
            return False

    print(f"  {condition_name}: completed in {runtime/60:.2f} min")
    return True

# ---------------- Main ----------------
summary_csv = OUTPUT_ROOT / "summary.csv"
total_conditions = len(rounds) * len(specificities) * len(binders)
print(f"Adaptive Monte Carlo batch (two-phase, subfolder version): {total_conditions} total conditions.\n")

# --- Global cleanup before running anything ---
if DELETE_FRAGMENT_DETAILS:
    print(f"Performing global fragment-details cleanup in {OUTPUT_ROOT} ...")
    total_deleted = cleanup_fragment_details(OUTPUT_ROOT)
    print(f"  ✅ Removed {total_deleted} total '{FRAGMENT_DETAILS_NAME}' file(s) before starting.\n")

# Also prune any leftover incomplete rep folders globally
print(f"Pruning incomplete reps (missing {SUCCESS_MARKER}) in {OUTPUT_ROOT} ...")
pruned_count, _ = prune_incomplete_reps(OUTPUT_ROOT)
print(f"  ✅ Pruned {pruned_count} incomplete rep folder(s) before starting.\n")

# --- Thread/Env setup ---
base_env = os.environ.copy()
base_env.setdefault("OMP_NUM_THREADS", "1")
base_env.setdefault("MKL_NUM_THREADS", "1")
base_env.setdefault("OPENBLAS_NUM_THREADS", "1")
base_env.setdefault("NUMEXPR_NUM_THREADS", "1")
base_env.setdefault("HMM_SKIP_FRAGMENT_DETAILS", "1")  # tell HMM.py to skip per-fragment logs

# ---------------- Phase 1: Primer pass (one rep each) ----------------
print("=== Phase 1: primer pass (one replicate per condition) ===")
for round_tag, spec, binder in itertools.product(rounds, specificities, binders):
    condition_name = f"{binder}_{spec}_{round_tag}"
    condition_dir = OUTPUT_ROOT / condition_name
    condition_dir.mkdir(exist_ok=True)

    # Decide if we need a primer rep
    prior = get_condition_results(summary_csv, sample_size, round_tag, binder, spec, no_error)
    if len(prior) >= 1:
        print(f"  {condition_name}: already has ≥1 run, skipping primer.")
        continue

    # Run exactly one rep
    result = run_one_rep(round_tag, spec, binder, base_env)
    if result is False:
        print(f"  {condition_name}: primer rep failed (will be retried in adaptive phase if needed).")

print("\n=== Phase 1 complete ===\n")

# ---------------- Phase 2: Adaptive pass ----------------
print("=== Phase 2: adaptive pass (continue until convergence or MAX_REPS) ===")
for round_tag, spec, binder in itertools.product(rounds, specificities, binders):
    condition_name = f"{binder}_{spec}_{round_tag}"
    condition_dir = OUTPUT_ROOT / condition_name
    condition_dir.mkdir(exist_ok=True)

    # Load current state
    all_correct = get_condition_results(summary_csv, sample_size, round_tag, binder, spec, no_error)
    reps_done = len(existing_rep_indices(condition_dir))

    # Check if already converged
    if len(all_correct) >= MIN_REPS:
        mean_acc, half_width = mean_confidence_interval(all_correct)
        if half_width <= TARGET_PRECISION * mean_acc:
            print(f"  {condition_name}: already converged ({len(all_correct)} runs, mean={mean_acc:.4f} ±{half_width:.4f})")
            continue
        else:
            print(f"  {condition_name}: continuing (have {len(all_correct)} runs, mean={mean_acc:.4f} ±{half_width:.4f})")
    else:
        print(f"  {condition_name}: have {len(all_correct)} run(s); need at least {MIN_REPS} before checking convergence.")

    # Keep adding reps up to MAX_REPS
    while reps_done < MAX_REPS:
        ok = run_one_rep(round_tag, spec, binder, base_env)
        if ok is False:
            # failed run—stop trying this condition this pass (prevents tight failure loops)
            break

        # refresh state
        reps_done = len(existing_rep_indices(condition_dir))
        all_correct = get_condition_results(summary_csv, sample_size, round_tag, binder, spec, no_error)

        if len(all_correct) >= MIN_REPS:
            mean_acc, half_width = mean_confidence_interval(all_correct)
            print(f"  {condition_name}: after {len(all_correct)} runs -> mean={mean_acc:.4f}, ±{half_width:.4f}")
            if half_width <= TARGET_PRECISION * mean_acc:
                print(f"  {condition_name}: ✅ converged (±{half_width:.4f} ≤ {TARGET_PRECISION*100:.1f}% of mean)\n")
                break

print("\n✅ Adaptive batch complete. Combined results are in test/outputs/summary.csv")
