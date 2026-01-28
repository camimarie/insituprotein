import pickle
import os

def convert_sample_size(s):
    """Convert sample size string (e.g., '1k') to integer"""
    s = s.lower()
    if s.endswith("k"):
        return int(float(s[:-1]) * 1000)
    else:
        return int(s)

def process_sequence(chunk_size, protein_dict, parameters, output_file):
    """Process a chunk of sequences and save the resulting trie"""
    try:
        from .SampleQueue import SequenceProcessor
        sp = SequenceProcessor(n_trials=chunk_size)
        sp.run(protein_dict, **parameters)
        with open(output_file, 'wb') as f:
            pickle.dump(sp.Trie, f)
        print(f"Processed chunk with chunk_size {chunk_size}, saved to {output_file}")
    except Exception as e:
        print(f"Error processing chunk with chunk_size {chunk_size}: {e}")
        raise

def merge_tries(target_trie, source_trie):
    """Merge source trie into target trie"""
    def _merge_nodes(node1, node2):
        for pid, count in node2.protein_counter.items():
            node1.protein_counter[pid] = node1.protein_counter.get(pid, 0) + count

        if node2.is_end:
            node1.is_end = True

        for char, child_node in node2.children.items():
            if char not in node1.children:
                from .SampleTrie import TrieNode
                node1.children[char] = TrieNode()
            _merge_nodes(node1.children[char], child_node)
    _merge_nodes(target_trie.root, source_trie.root)

def print_fragments_with_proteins(trie):
    """Print number of fragments with total protein count >= 10"""
    fragments = trie.get_all_fragments()
    count = 0
    for fragment, protein_counter in fragments:
        total_protein_count = sum(protein_counter.values())
        if total_protein_count >= 10:
            count += 1
    print(f"Number of fragments with at least 10 protein counts: {count}") 