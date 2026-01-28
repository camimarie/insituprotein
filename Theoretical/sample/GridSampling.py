"""
Description: This file contains the GridSampling module for performing grid sampling of protein sequences using parallel processing.
"""

from __future__ import annotations
import argparse, csv, json, os, random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
from concurrent.futures import ProcessPoolExecutor
import sys
sys.setrecursionlimit(10_000) # required for deep tries

from sample.SampleQueue import SequenceProcessor
from sample.GridSamplingHelpers import (
    BaseParams, load_json,
    parse_grid, default_fix_grid, default_anchordigest_grid,
    ic_from_trie, multiset_from_trie, make_seed
)

# ----------------- Globals for worker cache -----------------
_G_BASE: BaseParams | None = None
_G_PROTEOME: Dict[str, Dict] | None = None

def _worker_init(params_json_path: str, proteome_json_path: str):
    """Load heavy inputs once per worker (cuts pickling/IPC)."""
    global _G_BASE, _G_PROTEOME
    _G_BASE = BaseParams.from_json(Path(params_json_path))
    _G_PROTEOME = load_json(Path(proteome_json_path))

# ----------------- Job & worker -----------------
@dataclass(frozen=True)
class GridJob:
    pid: str
    L: int
    fix_p: float
    anch_p: float
    dig_p: float
    anchor_reagent: str
    digest_reagent: str
    iterations: int
    edman_mode: str
    edman_cap: int
    seed: int
    job_index: int  # deterministic index for minimal seeding
    store_fragments: bool = False # whether to store fragment JSONs in detailed output (costly files)

def run_job(job: GridJob) -> Tuple[List, List[List]]:
    """Execute a single (pid, fix, anch, dig) bundle across its replicates."""
    assert _G_BASE is not None and _G_PROTEOME is not None, "Worker globals not initialized"
    base = _G_BASE
    seq  = _G_PROTEOME[job.pid]["sequence"]

    det_rows: List[List] = []
    ic_vals: List[float] = []

    for r in range(1, job.iterations + 1):
        # Minimal deterministic seeding, independent per (job, replicate)
        rep_seed = make_seed(job.seed, job.job_index, r)
        random.seed(rep_seed); np.random.seed(rep_seed)

        # probe: longest fragment => edman rounds
        sp_probe = SequenceProcessor(n_trials=1)
        probe_params = base.build(job.fix_p, job.anch_p, job.anchor_reagent,
                                  job.dig_p, job.digest_reagent)
        seq_after = sp_probe.pre_cleave(seq, **probe_params)
        frags     = sp_probe.cleave(seq_after, **probe_params)
        max_len   = max((len(f) for f in frags), default=0)

        if job.edman_mode == "dynamic":
            rounds = int(max_len)
        elif job.edman_mode == "clamp":
            rounds = int(min(max_len, max(0, job.edman_cap)))
        else:
            raise ValueError("edman_mode must be 'dynamic' or 'clamp'")

        # full run at fixed rounds
        random.seed(rep_seed); np.random.seed(rep_seed)
        sp = SequenceProcessor(n_trials=1)
        run_params = base.build(job.fix_p, job.anch_p, job.anchor_reagent,
                                job.dig_p, job.digest_reagent,
                                edman_rounds=rounds)
        sp.run({job.pid: {"sequence": seq}}, **run_params)

        ic = ic_from_trie(sp.Trie, job.L)
        ic_vals.append(ic)
        frags_map = multiset_from_trie(sp.Trie)

        # Conditionally store fragment JSON to save space
        frag_json = json.dumps(frags_map) if job.store_fragments else ""

        det_rows.append([
            job.pid, job.L,
            f"{job.fix_p:.2f}", f"{job.anch_p:.2f}", f"{job.dig_p:.2f}",
            job.anchor_reagent, job.digest_reagent,
            rounds, r,
            f"{ic:.6f}", frag_json
        ])

    # summary row for this (pid, fix, anch, dig)
    if ic_vals:
        mu = float(np.mean(ic_vals))
        sd = float(np.std(ic_vals, ddof=0))
        mn, mx = min(ic_vals), max(ic_vals)
    else:
        mu = sd = mn = mx = 0.0

    summ_row = [
        job.pid, job.L,
        f"{job.fix_p:.2f}", f"{job.anch_p:.2f}", f"{job.dig_p:.2f}",
        job.anchor_reagent, job.digest_reagent,
        job.edman_mode, job.edman_cap if job.edman_mode == "clamp" else "",
        f"{mu:.6f}", f"{sd:.6f}", f"{mn:.6f}", f"{mx:.6f}", len(ic_vals)
    ]
    return summ_row, det_rows

# ----------------- Orchestration -----------------
def main():
    ap = argparse.ArgumentParser(description="Parallel grid sweep (core).")
    ap.add_argument("--proteome_json", type=Path, required=True)
    ap.add_argument("--params_json",  type=Path, required=True)
    ap.add_argument("--out_dir",      type=Path, required=True)
    ap.add_argument("--anchor_reagent", choices=["acx","epoxide"], required=True)
    ap.add_argument("--digest_reagent", choices=["trypsin","lysc","prok"], required=True)
    ap.add_argument("--iterations", type=int, default=10)
    ap.add_argument("--fix_grid",  type=str, default="")
    ap.add_argument("--anch_grid", type=str, default="")
    ap.add_argument("--dig_grid",  type=str, default="")
    ap.add_argument("--edman_mode", choices=["dynamic","clamp"], default="dynamic")
    ap.add_argument("--edman_cap", type=int, default=15)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--max_workers", type=int, default=None)
    ap.add_argument("--store_fragments", action="store_true",
                    help="If set, include fragment JSONs in detailed CSV output (large files).")
    args = ap.parse_args()

    # deterministic top-level RNG
    random.seed(args.seed); np.random.seed(args.seed & 0x7FFFFFFF)

    # IO
    out = args.out_dir; out.mkdir(parents=True, exist_ok=True)
    det_path = out / "grid_ic_detailed.csv"
    sum_path = out / "grid_ic_summary.csv"

    # Inputs (master only; workers load their own via initializer)
    proteome: Dict[str, Dict] = load_json(args.proteome_json)

    # Grids
    fix_grid  = parse_grid(args.fix_grid,  default_fix_grid())
    anch_grid = parse_grid(args.anch_grid, default_anchordigest_grid())
    dig_grid  = parse_grid(args.dig_grid,  default_anchordigest_grid())

    # Deterministic job ordering via sorted protein IDs
    proteome_keys = sorted(proteome.keys())

    # Job generator (stable index)
    def jobs():
        idx = 0
        for fix in fix_grid:
            for dig in dig_grid:
                for anch in anch_grid:
                    for pid in proteome_keys:
                        L = len(proteome[pid]["sequence"])
                        yield GridJob(
                            pid=pid, L=L,
                            fix_p=fix, anch_p=anch, dig_p=dig,
                            anchor_reagent=args.anchor_reagent,
                            digest_reagent=args.digest_reagent,
                            iterations=args.iterations,
                            edman_mode=args.edman_mode,
                            edman_cap=args.edman_cap,
                            seed=args.seed,
                            job_index=idx,
                            store_fragments=args.store_fragments,
                        )
                        idx += 1

    max_workers = args.max_workers or (os.cpu_count() or 4)
    chunksize = max(16, 4 * max_workers)  # reduce scheduler/IPC overhead

    with det_path.open("w", newline="") as fdet, sum_path.open("w", newline="") as fsum:
        det_w = csv.writer(fdet)
        sum_w = csv.writer(fsum)
        det_w.writerow([
            "protein_id","prot_len","fixation","anchoring","digestion",
            "anchor_reagent","digest_reagent","edman_rounds","replicate",
            "info_content","fragments_json"
        ])
        sum_w.writerow([
            "protein_id","prot_len","fixation","anchoring","digestion",
            "anchor_reagent","digest_reagent","edman_mode","edman_cap",
            "mean_ic","stdev_ic","min_ic","max_ic","n"
        ])

        # Note: initializer loads BaseParams + proteome once per worker
        with ProcessPoolExecutor(
            max_workers=max_workers,
            initializer=_worker_init,
            initargs=(str(args.params_json), str(args.proteome_json))
        ) as ex:
            completed = 0
            submitted = 0
            progress_interval = 500  # print every 500 completed jobs

            # Iterate results from ex.map (generator yields as jobs finish)
            for result in ex.map(run_job, jobs(), chunksize=chunksize):
                summ_row, det_rows = result
                sum_w.writerow(summ_row)
                det_w.writerows(det_rows)

                completed += 1
                submitted += 1  # both increase together since map yields in-order

                if completed % progress_interval == 0:
                    print(f"[progress] completed={completed} submitted={submitted}", flush=True)

            print(f"[done] completed={completed} submitted={submitted}", flush=True)

    print(f"Wrote:\n  - {det_path}\n  - {sum_path}")

if __name__ == "__main__":
    try:
        import multiprocessing as mp
        mp.set_start_method("spawn")
    except Exception:
        pass
    main()
