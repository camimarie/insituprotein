"""
Description: This file contains the implementation of a queue data structure for processing protein sequences, specifically for testing in 10 amino acids.
"""

from collections import deque
import random
import sys
import os

# Add the test directory to the path so we can import Trie
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from Trie import Trie
import numpy as np

FIXED_AA = "J"
ANCHORED_AA = "Z"
UNCONJUGATED_AA = "B"
BINDERS = "ANDEQILKFV" # 10 aas

class SequenceProcessor:
    def __init__(self, n_trials):
        self.queue = deque()
        self.n_trials = n_trials
        self.Trie = Trie()
    
    def enqueue_initial_sequences(self, protein_dict):
        """
        Populates the queue with initial sequences from the protein dictionary.
        """
        # enqueue the sequences n_trials times
        for _ in range(self.n_trials):
            for protein_id, data in protein_dict.items():
                self.queue.append((protein_id, data['sequence']))

    def _fixation(self, sequence, fixation_prob):
        """
        Modifies the sequence based on fixation probabilities. Supports per-residue and N-terminus fixation.
        """
        # no fixation, no changes
        if not fixation_prob: 
            return sequence 
        
        sequence = list(sequence)

        # N-terminus fixation if specified
        p_nterm = fixation_prob.get("N_term", None)
        if p_nterm is not None and len(sequence) > 0 and random.random() < p_nterm:
            sequence[0] = FIXED_AA

        # per-residue fixation except N-terminus
        for i, aa in enumerate(sequence[1:]):
            p = fixation_prob.get(aa, None)
            if p is not None and random.random() < p:
                sequence[i + 1] = FIXED_AA

        return ''.join(sequence)
    
    def _anchoring(self, sequence, anchor_prob):
        """
        Modifies the sequence based on anchoring probabilities. Supports per-residue and N-terminus and C-terminus anchoring.
        """
        # no anchoring, no changes
        if not anchor_prob: 
            return sequence
        
        sequence = list(sequence)

        # N-terminus anchoring if specified
        p_nterm = anchor_prob.get("N_term", None)
        if p_nterm is not None and len(sequence) > 0 and random.random() < p_nterm:
            sequence[0] = ANCHORED_AA

        # C-terminus anchoring if specified
        p_cterm = anchor_prob.get('C_term', None)
        if p_cterm is not None and len(sequence) > 0 and random.random() < p_cterm:
            sequence[-1] = ANCHORED_AA

        # per-residue anchoring except N-terminus and C-terminus
        for i, aa in enumerate(sequence[1:-1]):
            p = anchor_prob.get(aa, None)
            if p is not None and random.random() < p:
                sequence[i + 1] = ANCHORED_AA

        return ''.join(sequence)
    
    def pre_cleave(self, sequence, **params):
        """
        Applies fixation and anchoring to the sequence.
        """
        # fixation
        fixation_prob = params.get("fixation_prob")
        sequence = self._fixation(sequence, fixation_prob)
        
        # anchoring
        anchor_prob = params.get("anchor_prob")
        sequence = self._anchoring(sequence, anchor_prob)

        return sequence
    
    def cleave(self, sequence, **params):
        """
        Digests the sequence into fragments based on cleavage probabilities.
        """
        fragments = []
        current_fragment = []
        first_fragment_can_edman_prob = params.get("first_fragment_can_edman_prob")

        for i, aa in enumerate(sequence):
            current_fragment.append(aa)

            if self.__should_cleave(sequence, i, **params):
                fragments.append(''.join(current_fragment))
                current_fragment = []

        if current_fragment:
            fragments.append(''.join(current_fragment))

        # if N-terminus degradation is not allowed, or first AA of the first fragment is anchored/fixed, remove first fragment
        if fragments:
            if first_fragment_can_edman_prob == 0:
                fragments = fragments[1:]
            else:
                if fragments[0][0] in {ANCHORED_AA, FIXED_AA}: # {'Z','J'}
                    fragments = fragments[1:]
                elif random.random() < 1 - first_fragment_can_edman_prob:
                    fragments = fragments[1:]

        # keep only fragments that contain an anchor
        return [fragment for fragment in fragments if ANCHORED_AA in fragment]

    def __should_cleave(self, sequence, i, **params):
        """
        Checks if digestion should occur after the residue at index.
        """
        cleave_after_prob = params.get("cleave_after_prob") or {}
        cleave_before_prob = params.get("cleave_before_prob") or {}

        # cleave after present residue
        p_after = cleave_after_prob.get(sequence[i], None)
        if p_after is not None and random.random() < p_after:
            return True

        # cleave before next residue
        if i + 1 < len(sequence):
            p_before = cleave_before_prob.get(sequence[i + 1], None)
            if p_before is not None and random.random() < p_before:
                return True

        return False
    
    def binder_edman(self, fragment, **params):
        """
        Performs Edman degradation on the fragment and mask fragment based on binders.
        """
        binder_prob = params["binder_prob"]
        edman_conjug_fail_prob = params["edman_conjug_fail_prob"]
        edman_cleave_fail_prob = params["edman_cleave_fail_prob"]
        edman_rounds = int(params.get("edman_rounds", 15))

        amino_acids = BINDERS # 10 aas

        if ANCHORED_AA not in fragment:
            return None
        
        pB_corr, pX_corr = binder_prob[0]
        pB_inc,  pX_inc  = binder_prob[1]

        processed_fragment = []
        round_count = 0
        conjugated = False  # fragment is not conjugated at start

        while round_count < edman_rounds:
            # if fragment became empty, stop
            if not fragment:
                break
            # if anchor disappeared, stop
            if ANCHORED_AA not in fragment:
                break

            # attempt conjugation once per round if not already conjugated
            if not conjugated:
                conjugated = (random.random() >= edman_conjug_fail_prob)

            if not conjugated:
                # conjugation failed: all channels read 'X'
                processed_fragment.extend(['X'] * len(amino_acids))
            else:
                current_aa = fragment[0]
                for binder in amino_acids:
                    if current_aa == binder:
                        pB, pX = pB_corr, pX_corr     # correct channel
                    else:
                        pB, pX = pB_inc,  pX_inc      # incorrect channel
                        
                    emitted = np.random.choice([binder, 'X'], p=[pB, pX])
                    processed_fragment.append(emitted)

            # attempt cleavage at end of round (only if conjugated)
            if conjugated:
                if random.random() >= edman_cleave_fail_prob:
                    # successful cleavage: advance N-terminus and reset conjugation
                    fragment = fragment[1:] if fragment else fragment
                    conjugated = False
                # else: cleavage failure -> keep same residue; conjugation persists

            round_count += 1

        # pad remaining rounds with 'X' for each channel
        missing = edman_rounds - round_count
        if missing > 0:
            processed_fragment.extend(['X'] * (missing * len(amino_acids)))

        # if completely uninformative, drop it
        if not processed_fragment or all(ch == 'X' for ch in processed_fragment):
            return None

        return ''.join(processed_fragment)

    def process(self, **params):
        """
        Processes sequences in the queue through pre-cleavage, cleavage, and binder Edman degradation
        """
        initial_data = list(self.queue)
        self.queue.clear()

        for protein_id, sequence  in initial_data:
            sequence = self.pre_cleave(sequence, **params)
            fragments = self.cleave(sequence, **params)
            for fragment in fragments:
                self.queue.append((protein_id, fragment))

        while self.queue:
            protein_id, fragment = self.queue.popleft()
            fragment = self.binder_edman(fragment, **params)
            if fragment is not None:
                self.Trie.insert(protein_id, fragment)

    def run(self, protein_dict, **operation_params):
        """
        Main method to run the sequence processing.
        """
        self.enqueue_initial_sequences(protein_dict)
        self.process(**operation_params)