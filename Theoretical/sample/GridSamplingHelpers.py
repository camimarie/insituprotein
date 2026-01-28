"""
Description: This file contains helper functions and classes for grid sampling of protein sequences.
"""


from __future__ import annotations
import json
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

# ----------------- Constants -----------------
CANONICAL = set("ARNDCEQGHILKMFPSTWYV")
EPOXIDE_AAS = ["C","H","K","D","E","Y","N_term"]
ACX_AAS     = ["K","N_term"]
PROK_AAS    = ["L","V","A","I","F","Y","W","T","E"]
TRYPSIN_AAS = ["K","R"]
LYSC_AAS    = ["K"]

# ----------------- Defaults -----------------
def parse_grid(s: str, default: List[float]) -> List[float]:
    """Parse a comma-separated float list, formatted to 2dp; return default if empty."""
    return default if not s else [float(f"{float(tok.strip()):.2f}") for tok in s.split(",") if tok.strip()]

def default_fix_grid() -> List[float]:
    """0, 0.01, 0.05, then 0.1..1.0"""
    return [0.0, 0.01, 0.05] + [round(x/10,2) for x in range(1,11)]

def default_anchordigest_grid() -> List[float]:
    return [round(x/10,2) for x in range(0,11)]

def load_json(p: Path) -> Dict:
    return json.loads(p.read_text())

# ----------------- RNG (deterministic) -----------------
def make_seed(base:int, job_index:int, replicate:int) -> int:
    """
    Minimal, deterministic seed with no hashing.
    Uses a stable job index (from a deterministic enumeration) plus replicate.
    Mask to 31 bits for NumPy's seeding expectations.
    """
    return (base + job_index * 1_000_003 + replicate * 97) & 0x7FFFFFFF

# ----------------- IC math -----------------
def ic_from_trie(trie, true_len: int) -> float:
    """
    IC = sum_over_fragments( multiplicity(fragment) * #canonical_AAs_in(fragment) ) / protein_length
    Optimized: avoid building strings; accumulate canonical counts along the path.
    """
    if true_len <= 0:
        return 0.0
    total = 0
    # stack holds (node, canonical_count_so_far)
    stack = [(trie.root, 0)]
    while stack:
        node, canon = stack.pop()
        if getattr(node, "is_end", False):
            mult = sum(getattr(node, "protein_counter", {}).values())
            total += canon * mult
        for ch, child in getattr(node, "children", {}).items():
            stack.append((child, canon + (1 if ch in CANONICAL else 0)))
    return total / true_len

def multiset_from_trie(trie) -> Dict[str, int]:
    """Return {fragment: total_multiplicity} using trie.get_all_fragments()."""
    out: Dict[str,int] = {}
    for frag, pc in trie.get_all_fragments():
        out[frag] = out.get(frag, 0) + sum(pc.values())
    return out

# ----------------- Parameter builder -----------------
@dataclass
class BaseParams:
    """Thin wrapper + builder for SequenceProcessor params."""
    d: Dict

    @staticmethod
    def from_json(path: Path) -> "BaseParams":
        return BaseParams(load_json(path))

    def __post_init__(self):
        # Build a static template once, minus fields we overwrite per call.
        static = deepcopy(self.d)
        static.pop("fixation_prob", None)
        static.pop("anchor_prob", None)
        static.pop("cleave_before_prob", None)
        static.pop("cleave_after_prob", None)
        static.pop("edman_rounds", None)
        self._static = static

    def build(self, fix_p: float, anch_p: float,
              anchor_reagent: str,
              dig_p: float, digest_reagent: str,
              edman_rounds: int | None = None) -> Dict:
        """Return a fully populated params dict for SequenceProcessor without deep JSON copy."""
        # top-level shallow copy is enough since we only add small dicts
        params = dict(self._static)

        # fixation
        params["fixation_prob"] = {"K":fix_p, "C":fix_p, "Y":fix_p, "R":fix_p, "N_term":fix_p}

        # anchoring
        if anchor_reagent == "acx":
            params["anchor_prob"] = {aa: anch_p for aa in ACX_AAS}
        elif anchor_reagent == "epoxide":
            params["anchor_prob"] = {aa: anch_p for aa in EPOXIDE_AAS}
        else:
            raise ValueError("anchor_reagent must be 'acx' or 'epoxide'")

        # digestion
        if digest_reagent == "trypsin":
            params["cleave_after_prob"] = {k: dig_p for k in TRYPSIN_AAS}
        elif digest_reagent == "lysc":
            params["cleave_after_prob"] = {k: dig_p for k in LYSC_AAS}
        elif digest_reagent == "prok":
            params["cleave_after_prob"] = {k: dig_p for k in PROK_AAS}
        else:
            raise ValueError("digest_reagent must be 'trypsin','lysc','prok'")

        if edman_rounds is not None:
            params["edman_rounds"] = int(edman_rounds)

        return params
