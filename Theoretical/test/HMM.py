"""
This file contains the Hidden Markov Model implementation and scoring functions for protein fragment matching.
(Fast + correct: big chunks, but restores old per-occurrence semantics + correct aggregation paths)
"""

import json, time, os, math, csv, gc, argparse, importlib.util, pickle, sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import pandas as pd
import numpy as np
os.environ["NUMBA_NUM_THREADS"] = "1"
import numba
from math import floor

ISPS_ROOT = Path(__file__).resolve().parents[1]
if str(ISPS_ROOT) not in sys.path:
    sys.path.insert(0, str(ISPS_ROOT))

from Worker import init_worker, process_fragment_batch

# ---------------- Global paths ----------------
HERE = Path(__file__).resolve().parent              # .../test
PARAM_DIR = HERE / "parameters"                    # test/parameters
RUNS_DIR  = HERE / "runs"                          # test/runs
PROTEOME_JSON = ISPS_ROOT / "preprocessing" / "proteomes" / "protein-data-243273.json"
SAMPLEREF_DIR = ISPS_ROOT / "samplereference" / "data" / "243273"

# ---------------- CLI ----------------
def parse_arguments():
    p = argparse.ArgumentParser("Run HMM.")
    p.add_argument("--sample_size", required=True, help="1k")
    p.add_argument("--round",       required=True, help="5-15r")
    p.add_argument("--binder",      required=True, help="5aas, 10aas, 15aas, 20aas")
    p.add_argument("--specificity", required=True, help="perfect, medium, low, verylow")
    p.add_argument("--output",      required=True, help="output directory for results")
    p.add_argument("--no_error",    default="false", help="true sets conjugation and cleavage fail probs to 0")
    return p.parse_args()

# ---------------- Utilities ----------------
def _require_file(p: Path, why: str):
    if not p.exists():
        raise FileNotFoundError(f"Missing {why}: {p}")

# ---------------- Scoring helpers ----------------
def min_informative_from_rounds(edman_rounds: int):
    """Dynamic threshold: floor(0.5 * edman_rounds), at least 1."""
    return max(1, floor(0.5 * int(edman_rounds)))

def informative_count(s: str):
    """# of non-'X' positions."""
    return sum(ch != 'X' and ch != 'x' for ch in s)

def is_informative_ref(s: str, min_cnt: int):
    """Keep sequences with >= min_cnt informative positions for reference fragments."""
    return informative_count(s) >= min_cnt

def is_informative_query(s: str, min_cnt: int, n_binders: int, edman_rounds: int):
    """
    Keep queries with >= dynamic threshold of informative rounds.
    A round is informative if any of its m channels is not 'X'.
    """
    informative = 0
    for i in range(edman_rounds):
        a = i * n_binders
        b = a + n_binders
        block = s[a:b]
        if any(ch != 'X' and ch != 'x' for ch in block):
            informative += 1
            if informative >= min_cnt:
                return True
    return informative >= min_cnt

def topk(entries, k=50):
    return sorted(entries, key=lambda x: x[1], reverse=True)[:k]

@numba.njit(cache=True, fastmath=False)
def _logaddexp(a, b):
    if a == -np.inf: return b
    if b == -np.inf: return a
    m = a if a > b else b
    return m + math.log(math.exp(a - m) + math.exp(b - m))

# =======================================================
#             Round-based DP (no e_lookup)
# =======================================================

@numba.njit(cache=True, fastmath=False)
def compute_forward_jit_rounds(
    T_arr,
    S_arr,
    correct_chan,
    m,
    R,
    pB_corr, pX_corr, pB_inc, pX_inc, c,
    log_d, log_1md,
    epsilon_forward, min_prob
):
    N = S_arr.shape[0]
    dp = np.full((R + 1, N + 1), -np.inf)
    dp[0, 0] = 0.0
    floor_log = math.log(min_prob)

    for r in range(R):
        row_max = -np.inf
        for i in range(N + 1):
            cur = dp[r, i]
            if cur == -np.inf:
                continue

            if i == N:
                log_emiss_round = 0.0
                base = r * m
                for k in range(m):
                    obs = T_arr[base + k]
                    if obs != 0:
                        log_emiss_round += floor_log
                val_stay = cur + log_d + log_emiss_round
                if val_stay > dp[r+1, i]:
                    dp[r+1, i] = val_stay
                val_adv = cur + log_1md + log_emiss_round
                if val_adv > dp[r+1, i]:
                    dp[r+1, i] = val_adv
                if dp[r+1, i] > row_max:
                    row_max = dp[r+1, i]
                continue

            tcode = S_arr[i]
            corr_k = correct_chan[tcode]
            base = r * m

            le = 0.0
            for k in range(m):
                obs = T_arr[base + k]
                chan = k + 1
                is_corr = (corr_k == chan)

                if obs == chan:
                    pB = pB_corr if is_corr else pB_inc
                    le += math.log((1.0 - c) * pB)
                else:
                    if obs != 0:
                        le += floor_log
                    else:
                        pX = pX_corr if is_corr else pX_inc
                        le += math.log(c + (1.0 - c) * pX)

            val_stay = cur + log_d + le
            val_adv = cur + log_1md + le

            cur_next_stay = dp[r+1, i]
            if val_stay > cur_next_stay:
                dp[r+1, i] = val_stay
            else:
                dp[r+1, i] = _logaddexp(cur_next_stay, val_stay)

            if i < N:
                cur_next_adv = dp[r+1, i+1]
                if val_adv > cur_next_adv:
                    dp[r+1, i+1] = val_adv
                else:
                    dp[r+1, i+1] = _logaddexp(cur_next_adv, val_adv)

            if dp[r+1, i] > row_max: row_max = dp[r+1, i]
            if i < N and dp[r+1, i+1] > row_max: row_max = dp[r+1, i+1]

        thr = row_max - epsilon_forward
        for j in range(N + 1):
            if dp[r+1, j] < thr:
                dp[r+1, j] = floor_log

    total = -np.inf
    for j in range(N + 1):
        total = _logaddexp(total, dp[R, j])
    return total

@numba.njit(cache=True, fastmath=False)
def compute_viterbi_jit_rounds(
    T_arr, S_arr, correct_chan, m, R,
    pB_corr, pX_corr, pB_inc, pX_inc, c,
    log_d, log_1md,
    epsilon_viterbi, min_prob
):
    N = S_arr.shape[0]
    dp = np.full((R + 1, N + 1), -np.inf)
    dp[0, 0] = 0.0
    floor_log = math.log(min_prob)

    for r in range(R):
        row_max = -np.inf
        for i in range(N + 1):
            cur = dp[r, i]
            if cur == -np.inf:
                continue

            if i == N:
                log_emiss_round = 0.0
                base = r * m
                for k in range(m):
                    obs = T_arr[base + k]
                    if obs != 0:
                        log_emiss_round += floor_log
                val_stay = cur + log_d + log_emiss_round
                if val_stay > dp[r+1, i]:
                    dp[r+1, i] = val_stay
                val_adv = cur + log_1md + log_emiss_round
                if val_adv > dp[r+1, i]:
                    dp[r+1, i] = val_adv
                if dp[r+1, i] > row_max: row_max = dp[r+1, i]
                continue

            tcode = S_arr[i]
            corr_k = correct_chan[tcode]
            base = r * m

            le = 0.0
            for k in range(m):
                obs = T_arr[base + k]
                chan = k + 1
                is_corr = (corr_k == chan)
                if obs == chan:
                    pB = pB_corr if is_corr else pB_inc
                    le += math.log((1.0 - c) * pB)
                else:
                    if obs != 0:
                        le += floor_log
                    else:
                        pX = pX_corr if is_corr else pX_inc
                        le += math.log(c + (1.0 - c) * pX)

            val = cur + log_d + le
            if val > dp[r+1, i]:
                dp[r+1, i] = val

            if i < N:
                val2 = cur + log_1md + le
                if val2 > dp[r+1, i+1]:
                    dp[r+1, i+1] = val2

            if dp[r+1, i] > row_max: row_max = dp[r+1, i]
            if i < N and dp[r+1, i+1] > row_max: row_max = dp[r+1, i+1]

        thr = row_max - epsilon_viterbi
        for j in range(N + 1):
            if dp[r+1, j] < thr:
                dp[r+1, j] = floor_log

    best = -np.inf
    for j in range(N + 1):
        if dp[R, j] > best:
            best = dp[R, j]
    return best

# ---------------- Fragment matcher ----------------
class FragmentMatcher:
    def __init__(self, parameters, effective_list):
        self.parameters = parameters

        self.obs_binder_list = [aa.strip() for aa in effective_list if aa.strip()]
        self.obs_binder_set  = set(self.obs_binder_list)
        self.m = len(self.obs_binder_list)

        self.true_aa_list = list("ARNDCEQGHILKMFPSTWYV")
        self.true_aa_set  = set(self.true_aa_list)

        self.binder_prob = self.parameters['binder_prob']
        assert isinstance(self.binder_prob, list) and len(self.binder_prob) == 2
        assert len(self.binder_prob[0]) == 2 and len(self.binder_prob[1]) == 2

        self.true_char_to_code = np.full(256, -1, np.int32)
        self.true_char_to_code[ord('X')] = 0
        for i, aa in enumerate(self.true_aa_list, start=1):
            self.true_char_to_code[ord(aa)] = i

        self.obs_char_to_code = np.full(256, -1, np.int32)
        self.obs_char_to_code[ord('X')] = 0
        for j, aa in enumerate(self.obs_binder_list, start=1):
            self.obs_char_to_code[ord(aa)] = j

        self.correct_chan = np.zeros(1 + len(self.true_aa_list), dtype=np.int32)
        for ti, aa in enumerate(self.true_aa_list, start=1):
            self.correct_chan[ti] = (self.obs_binder_list.index(aa) + 1) if aa in self.obs_binder_set else 0

        self.fragment_to_proteins = {}
        self.all_fragments = []
        self.all_fragment_codes = []
        self._cache = {}

    def _string_to_true_codes(self, s: str) -> np.ndarray:
        arr = np.frombuffer(s.encode('ascii'), np.uint8)
        codes = self.true_char_to_code[arr]
        if (codes < 0).any():
            codes = codes.copy()
            codes[codes < 0] = 0
        return codes

    def _string_to_obs_codes(self, s: str) -> np.ndarray:
        arr = np.frombuffer(s.encode('ascii'), np.uint8)
        codes = self.obs_char_to_code[arr]
        if (codes < 0).any():
            codes = codes.copy()
            codes[codes < 0] = 0
        return codes

    def index_fragments(self, fragments):
        for frag, pc in tqdm(fragments, desc="Indexing reference fragments"):
            self.fragment_to_proteins[frag] = pc
            self.all_fragments.append(frag)
            self.all_fragment_codes.append(self._string_to_true_codes(frag))

    def _scores_params(self):
        c = float(self.parameters['edman_conjug_fail_prob'])
        d = float(self.parameters['edman_cleave_fail_prob'])
        log_d   = math.log(d) if d > 0 else -np.inf
        log_1md = math.log(1 - d) if d < 1 else -np.inf

        pB_corr, pX_corr = [float(x) for x in self.binder_prob[0]]
        pB_inc,  pX_inc  = [float(x) for x in self.binder_prob[1]]
        return pB_corr, pX_corr, pB_inc, pX_inc, c, log_d, log_1md

    def find_best_match(self, query, beam_epsilon=5.0, forward_epsilon=10.0):
        if query in self._cache:
            return self._cache[query]

        T_codes = self._string_to_obs_codes(query).astype(np.int32)
        m = self.m
        if m == 0 or T_codes.shape[0] % m != 0:
            result = (None, "No matches found", None, {"query_fragment": query, "candidate_entries": []})
            self._cache[query] = result
            return result
        R = T_codes.shape[0] // m

        pB_corr, pX_corr, pB_inc, pX_inc, c, log_d, log_1md = self._scores_params()
        cc = self.correct_chan.astype(np.int32)

        n_refs = len(self.all_fragment_codes)
        vit_scores = np.empty(n_refs, dtype=np.float32)
        best_vit = -np.inf
        for i, S_codes in enumerate(self.all_fragment_codes):
            s_arr = S_codes.astype(np.int32)
            v = compute_viterbi_jit_rounds(
                T_codes, s_arr, cc, m, R,
                pB_corr, pX_corr, pB_inc, pX_inc, c,
                log_d, log_1md,
                5.0, 1e-50
            )
            vit_scores[i] = np.float32(v)
            if v > best_vit:
                best_vit = v

        threshold = best_vit - beam_epsilon
        narrowed_idx = np.nonzero(vit_scores >= threshold)[0]
        narrowed = [(self.all_fragments[i], self.all_fragment_codes[i]) for i in narrowed_idx]

        candidate_entries = []
        for frag, S_codes in narrowed:
            s_arr = S_codes.astype(np.int32)
            fw = compute_forward_jit_rounds(
                T_codes, s_arr, cc, m, R,
                pB_corr, pX_corr, pB_inc, pX_inc, c,
                log_d, log_1md,
                forward_epsilon, 1e-50
            )
            candidate_entries.append((frag, fw))

        dbg = {"query_fragment": query, "candidate_entries": candidate_entries[:10]}
        if not candidate_entries:
            result = (None, "No matches found", None, dbg)
            self._cache[query] = result
            return result

        top = topk(candidate_entries, 10)

        per_protein_log = {}
        for frag, logL in top:
            counts = self.fragment_to_proteins.get(frag, {})
            if not counts:
                continue
            for pid, cnt in counts.items():
                if cnt <= 0:
                    continue
                term = logL + math.log(cnt)
                cur = per_protein_log.get(pid, None)
                per_protein_log[pid] = term if cur is None else _logaddexp(cur, term)

        if not per_protein_log:
            result = (None, "No matches found", None, dbg)
            self._cache[query] = result
            return result

        vals = list(per_protein_log.items())
        maxlog = max(v for _, v in vals)
        prot_weights = [(pid, math.exp(v - maxlog)) for pid, v in vals]
        Z = sum(w for _, w in prot_weights) or 1.0
        prot_weights = [(pid, w / Z) for pid, w in prot_weights]
        prot_weights.sort(key=lambda x: x[1], reverse=True)

        top_pid, top_w = prot_weights[0]
        second_w = prot_weights[1][1] if len(prot_weights) > 1 else 0.0
        dbg["per_protein"] = prot_weights[:10]

        if len(prot_weights) > 1 and top_w < 1.5 * second_w:
            result = (None, "Uncertain", None, dbg)
        else:
            result = (top_pid, "Match", top_w, dbg)

        self._cache[query] = result
        return result

# ---------------- Streaming chunk builder ----------------
def occurrence_chunk_generator(aggregated_fragments, n_binders, edman_rounds, chunk_occurrences, total_chunks):
    """
    Yields (chunk, chunk_id, total_chunks) where chunk is a list of (frag, {pid:1}) occurrences.
    This restores the OLD per-occurrence semantics without building one huge list.
    """
    min_cnt_q = min_informative_from_rounds(edman_rounds)
    chunk = []
    chunk_id = 1

    for frag, pc in aggregated_fragments:
        if not is_informative_query(frag, min_cnt_q, n_binders, edman_rounds):
            continue

        for pid, count in pc.items():
            # expand to occurrences (old semantics)
            for _ in range(count):
                chunk.append((frag, {pid: 1}))
                if len(chunk) >= chunk_occurrences:
                    yield (chunk, chunk_id, total_chunks)
                    chunk_id += 1
                    chunk = []

    if chunk:
        yield (chunk, chunk_id, total_chunks)

# ---------------- Main pipeline ----------------
def main():
    start_time = time.time()
    args = parse_arguments()

    os.makedirs(args.output, exist_ok=True)
    SKIP_FRAGMENT_DETAILS = os.getenv("HMM_SKIP_FRAGMENT_DETAILS", "0") == "1"

    queue_path = RUNS_DIR / args.binder / f"Queue{args.binder}.py"
    param_file = PARAM_DIR / f"{args.specificity}.json"
    _require_file(param_file, "parameter JSON")
    _require_file(PROTEOME_JSON, "proteome JSON")
    _require_file(queue_path, "queue file")

    spec = importlib.util.spec_from_file_location("QueueModule", str(queue_path))
    queue_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(queue_module)

    with open(PROTEOME_JSON, "r") as f:
        protein_dict = json.load(f)
    with open(param_file, "r") as f:
        parameters = json.load(f)

    parameters['edman_rounds'] = int(args.round.strip().lower().replace("r", ""))
    if str(args.no_error).lower() == "true":
        parameters['edman_conjug_fail_prob'] = 0.0
        parameters['edman_cleave_fail_prob'] = 0.0

    # ---- Reference indexing ----
    sample_ref_pkl = SAMPLEREF_DIR / f"{args.round}{args.sample_size}.pkl"
    _require_file(sample_ref_pkl, "sample reference PKL")
    with open(sample_ref_pkl, "rb") as f:
        trie = pickle.load(f)

    training_fragments = trie.get_fragments_with_counts()
    min_cnt_ref = min_informative_from_rounds(parameters['edman_rounds'])
    training_fragments = [(frag, pc) for (frag, pc) in training_fragments if is_informative_ref(frag, min_cnt_ref)]

    effective_list = list(getattr(queue_module, "BINDERS"))
    matcher = FragmentMatcher(parameters, effective_list)
    matcher.index_fragments(training_fragments)

    # ---- Test set generation ----
    print("\nGenerating test fragments...")
    test_sp = queue_module.SequenceProcessor(1)
    test_sp.run(protein_dict, **parameters)
    aggregated_fragments = test_sp.Trie.get_fragments_with_counts()

    # ---- Prepare chunked occurrences (fast + correct) ----
    n_binders = len(effective_list)
    edman_rounds = parameters['edman_rounds']
    min_cnt_q = min_informative_from_rounds(edman_rounds)

    # Count total occurrences (cheap; no expansion yet)
    total_occ = 0
    for frag, pc in aggregated_fragments:
        if is_informative_query(frag, min_cnt_q, n_binders, edman_rounds):
            total_occ += sum(pc.values())

    max_workers = int(os.getenv("HMM_WORKERS", "31"))
    cmax_workers = int(os.getenv("HMM_WORKERS", "31"))

    # Aim for enough chunks to keep workers busy (load balancing)
    target_chunks_per_worker = int(os.getenv("HMM_CHUNKS_PER_WORKER", "6"))  # 4-10 is good
    target_chunks = max(1, max_workers * target_chunks_per_worker)

    # Choose chunk size so we get ~target_chunks
    chunk_occurrences = max(100, math.ceil(total_occ / target_chunks))
    chunk_occurrences = int(os.getenv("HMM_CHUNK_OCCURRENCES", str(chunk_occurrences)))  # allow override

    total_chunks = max(1, math.ceil(total_occ / chunk_occurrences))

    print(f"Total occurrences: {total_occ}")
    print(f"Workers: {max_workers}")
    print(f"Chunk occurrences: {chunk_occurrences}")
    print(f"Total chunks: {total_chunks} (target ~{target_chunks})")

    print(f"Scoring ~{total_occ} fragment occurrences in {total_chunks} chunks across {max_workers} workers...")

    # ---- Multiprocessing start method (match old behavior as much as possible) ----
    import multiprocessing as mp
    if mp.get_start_method(allow_none=True) is None:
        try:
            mp.set_start_method("fork")
        except RuntimeError:
            mp.set_start_method("spawn")

    # ---- Result aggregation (matches old) ----
    results = {'correct': 0, 'uncertain': 0, 'false_positive': 0, 'no_predictions': 0, 'details': {}}
    if not SKIP_FRAGMENT_DETAILS:
        results['fragment_details'] = []

    protein_acc = {}      # used if SKIP_FRAGMENT_DETAILS
    protein_results = {}  # used if NOT SKIP_FRAGMENT_DETAILS
    NO_PRED = {'No matches found', 'No valid matches', 'Uncertain'}

    chunks_iter = occurrence_chunk_generator(
        aggregated_fragments=aggregated_fragments,
        n_binders=n_binders,
        edman_rounds=edman_rounds,
        chunk_occurrences=chunk_occurrences,
        total_chunks=total_chunks
    )

    with ProcessPoolExecutor(max_workers=max_workers, initializer=init_worker, initargs=(matcher,)) as ex:
        for worker_results in tqdm(ex.map(process_fragment_batch, chunks_iter),
                                   total=total_chunks,
                                   desc="Processing"):
            for res in worker_results:
                true_pid = res['true_protein']

                if SKIP_FRAGMENT_DETAILS:
                    acc = protein_acc.setdefault(true_pid, {'total': 0, 'valid': 0, 'counts': {}})
                    acc['total'] += 1
                    if res.get('status') not in NO_PRED and res.get('predicted') is not None:
                        acc['valid'] += 1
                        p_id = res['predicted']
                        acc['counts'][p_id] = acc['counts'].get(p_id, 0) + 1
                else:
                    protein_results.setdefault(true_pid, []).append(res)
                    results['fragment_details'].append(res)

    # ---- Evaluation (same as old) ----
    print("\nEvaluating protein-level results...")
    iterator = protein_acc.items() if SKIP_FRAGMENT_DETAILS else protein_results.items()

    for true_pid, payload in tqdm(iterator):
        if SKIP_FRAGMENT_DETAILS:
            total = payload['total']
            valid = payload['valid']
            counts = payload['counts']
            if total == 0:
                results['no_predictions'] += 1
                results['details'][true_pid] = {
                    'result': 'uncertain',
                    'total_fragments': 0,
                    'valid_predictions': 0,
                    'prediction_counts': {}
                }
                continue
        else:
            preds = payload
            if not preds:
                results['no_predictions'] += 1
                results['details'][true_pid] = {
                    'result': 'uncertain',
                    'total_fragments': 0,
                    'valid_predictions': 0,
                    'prediction_counts': {}
                }
                continue

            counts = {}
            valid = 0
            for p in preds:
                if p.get('status') in NO_PRED:
                    continue
                pred_pid = p.get('predicted')
                if pred_pid is None:
                    continue
                counts[pred_pid] = counts.get(pred_pid, 0) + 1
                valid += 1
            total = len(preds)

        if valid == 0:
            results['uncertain'] += 1
            label = 'uncertain'
        else:
            ranked = sorted(counts.items(), key=lambda x: x[1], reverse=True)
            top_pid, top_n = ranked[0]
            second_n = ranked[1][1] if len(ranked) > 1 else 0
            if top_n >= 1.5 * second_n:
                if top_pid == true_pid:
                    results['correct'] += 1
                    label = 'correct'
                else:
                    results['false_positive'] += 1
                    label = 'false_positive'
            else:
                results['uncertain'] += 1
                label = 'uncertain'

        results['details'][true_pid] = {
            'result': label,
            'total_fragments': total,
            'valid_predictions': valid,
            'prediction_counts': counts
        }

    # ---- Saving outputs ----
    exec_time = time.time() - start_time
    outdir = Path(args.output)
    outdir.mkdir(parents=True, exist_ok=True)

    rows = []
    for pid, d in results['details'].items():
        row = {
            'protein_id': pid,
            'result': d['result'],
            'total_fragments': d['total_fragments'],
            'valid_predictions': d['valid_predictions']
        }
        for i, (pp, c) in enumerate(sorted(d['prediction_counts'].items(), key=lambda x: x[1], reverse=True), 1):
            row[f'prediction_{i}'] = pp
            row[f'count_{i}'] = c
        rows.append(row)
    pd.DataFrame(rows).to_csv(outdir / 'protein_summary.csv', index=False)

    if not SKIP_FRAGMENT_DETAILS and 'fragment_details' in results:
        pd.DataFrame(results['fragment_details']).to_csv(outdir / 'fragment_details.csv', index=False)

    tot_proteins = results['correct'] + results['false_positive'] + results['uncertain'] + results['no_predictions']
    acc_ex_unc = (results['correct'] / (results['correct'] + results['false_positive'])) if (results['correct'] + results['false_positive']) else None
    proteins_with_any_correct = {pid for pid, d in results['details'].items() if d['prediction_counts'].get(pid, 0) > 0}

    stats_df = pd.DataFrame({
        'metric': [
            'correct','false_positive','uncertain','no_predictions',
            'accuracy (excluding uncertain)',
            'proteins_with_at_least_one_correct_fragment',
            'percentage_proteins_with_correct_fragments',
            'exec_time_sec'
        ],
        'value': [
            results['correct'], results['false_positive'], results['uncertain'], results['no_predictions'],
            acc_ex_unc,
            len(proteins_with_any_correct),
            (len(proteins_with_any_correct)/tot_proteins if tot_proteins else 0),
            exec_time
        ]
    })
    stats_df.to_csv(outdir / 'statistics.csv', index=False)

    lines = []
    lines.append("Protein-Level Results:")
    lines.append(f"Correct Proteins: {results['correct']}")
    lines.append(f"False Positive Proteins: {results['false_positive']}")
    lines.append(f"Uncertain Proteins: {results['uncertain']}")
    lines.append(f"Proteins with No Predictions: {results['no_predictions']}")
    lines.append(f"Total Proteins: {tot_proteins}")
    lines.append(f"Protein-Level Accuracy (excluding uncertain): {acc_ex_unc:.2f}" if acc_ex_unc is not None else "Protein-Level Accuracy: N/A")
    lines.append(f"Execution Time (s): {exec_time:.2f}")
    with open(outdir / "summary.txt", "w") as f:
        f.write("\n".join(lines))

    global_summary = ISPS_ROOT / "test/outputs" / "summary.csv"
    global_summary.parent.mkdir(exist_ok=True)
    newfile = not global_summary.exists()
    with open(global_summary, "a", newline="") as f:
        w = csv.writer(f)
        if newfile:
            w.writerow(["sample_size","round","binder","specificity","no_error","correct","false_positive","uncertain"])
        w.writerow([args.sample_size, args.round, args.binder, args.specificity, args.no_error,
                    results['correct'], results['false_positive'], results['uncertain']])

    print(f"\nDone. Results saved to {outdir}")

if __name__ == "__main__":
    main()
    gc.collect()
