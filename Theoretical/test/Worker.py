# Worker.py
import os

GLOBAL_MATCHER = None
SKIP_FRAGMENT_DETAILS = os.getenv("HMM_SKIP_FRAGMENT_DETAILS", "0") == "1"

def init_worker(matcher):
    global GLOBAL_MATCHER
    GLOBAL_MATCHER = matcher

def process_fragment_batch(args):
    """
    args: (fragment_batch, chunk_id, total_chunks)
    fragment_batch: list of (frag_str, {true_pid: 1})
    Returns: list of result dicts (one per occurrence)
    """
    fragment_batch, chunk_id, total_chunks = args
    matcher = GLOBAL_MATCHER
    if matcher is None:
        return []

    skip_details = SKIP_FRAGMENT_DETAILS
    find_match = matcher.find_best_match

    batch_results = []
    for fragment, protein_counter in fragment_batch:
        # Score once per occurrence (same as original semantics)
        predicted_pid, status, weight, debug_info = find_match(fragment)

        if skip_details:
            debug_info = None
            frag_out = None
        else:
            frag_out = fragment

        # protein_counter is {true_pid: 1} by construction
        # (keep this loop robust anyway)
        for true_pid in protein_counter:
            batch_results.append({
                "true_protein": true_pid,
                "fragment": frag_out,
                "predicted": predicted_pid,
                "status": status,
                "weighted_count": weight,   # keep field parity with old
                "debug_info": debug_info
            })

    return batch_results
