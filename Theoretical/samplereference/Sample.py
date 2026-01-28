import os
import json
import pickle
import shutil
import sys

# Add the current directory to Python path so we can import sample modules
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from samplereference.SampleQueue import SequenceProcessor
from samplereference.SampleTrie import Trie, TrieNode
from multiprocessing import Pool, cpu_count
import csv
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description="Generate samples from sequences")
    parser.add_argument("--sample_size", required=True, help="Sample size (e.g., 100, 1k, 10k)")
    parser.add_argument("--round", required=True, help="Edman rounds (e.g., 5r, 6r, etc.)")
    parser.add_argument("--output", required=True, help="Output directory for samples")
    parser.add_argument("--param_path", required=True, help="Path to the parameter json file")
    parser.add_argument("--proteome_path", required=True, help="Path to the protein data json file")
    return parser.parse_args()

def convert_sample_size(s):
    s = s.lower()
    if s.endswith("k"):
        return int(float(s[:-1]) * 1000)
    else:
        return int(s)
    
# process a chunk of sequences
def process_sequence(chunk_size, protein_dict, parameters, output_file):
    try:
        sp = SequenceProcessor(n_trials=chunk_size)
        sp.run(protein_dict, **parameters)
        # save trie temporarily to disk for merge
        with open(output_file, 'wb') as f:
            pickle.dump(sp.Trie, f)
        print(f"processed chunk with chunk_size {chunk_size}, saved to {output_file}")
    except Exception as e:
        print(f"error processing chunk with chunk_size {chunk_size}: {e}")
        raise

# merges source trie to target trie
def merge_tries(target_trie, source_trie):
    def _merge_nodes(node1, node2):
        # merge protein occurrence counts
        for pid, count in node2.protein_counter.items():
            node1.protein_counter[pid] = node1.protein_counter.get(pid, 0) + count

        if node2.is_end:
            node1.is_end = True

        for char, child_node in node2.children.items():
            if char not in node1.children:
                node1.children[char] = TrieNode()
            _merge_nodes(node1.children[char], child_node)
    _merge_nodes(target_trie.root, source_trie.root)

# print the number of fragments in the trie that have at least ten protein counts summed across ids
def print_fragments_with_proteins(trie):
    """
    prints the number of fragments where the sum of all protein counts is at least 10.
    """
    fragments = trie.get_all_fragments()
    count = 0
    for fragment, protein_counter in fragments:
        total_protein_count = sum(protein_counter.values())
        if total_protein_count >= 10:
            count += 1
    print(count)

if __name__ == "__main__":
    args = parse_arguments()

    # load parameters from the provided param file
    param_path = args.param_path
    try:
        with open(param_path, 'r') as file:
            parameters = json.load(file)
    except Exception as e:
        print(f"error loading parameters from {args.param_path}: {e}")
        exit(1)

    # override sample_depth with command-line sample_size
    parameters["sample_depth"] = convert_sample_size(args.sample_size)
    # override edman rounds using the provided round argument (e.g., "5r" -> 5)
    parameters["edman_rounds"] = int(args.round[:-1])
    try:
        with open(args.proteome_path, 'r') as file:
            protein_dict = json.load(file)
    except Exception as e:
        print(f"error loading protein data: {e}")
        exit(1)

    # number of processes and compute chunk sizes, handling remainder
    num_processes = cpu_count()
    base_chunk = parameters["sample_depth"] // num_processes
    remainder = parameters["sample_depth"] % num_processes
    chunk_sizes = [base_chunk] * num_processes
    for i in range(remainder):
        chunk_sizes[i] += 1

    # create directory for temporary files
    temp_dir = "generatedtries"
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)

    output_files = [os.path.join(temp_dir, f'trie_{i}.pkl') for i in range(num_processes)]

    # using a pool to parallelize the generation of tries and saving them to disk
    tasks = [(chunk_sizes[i], protein_dict, parameters, output_files[i]) for i in range(num_processes)]
    print(f"starting multiprocessing with {num_processes} processes")
    print(f"chunk sizes: {chunk_sizes}")
    print(f"output files: {output_files}")
    print(f"parameters: {parameters}")
    try:
        with Pool(processes=num_processes) as pool:
            pool.starmap(process_sequence, tasks)
    except Exception as e:
        print(f"error during multiprocessing: {e}")
        exit(1)

    print("check done")
    # merging tries from disk into a single trie
    master_trie = Trie()
    for output_file in output_files:
        try:
            with open(output_file, 'rb') as f:
                trie = pickle.load(f)
                print(f"merging trie from file {output_file}")
            print(f"number of sequences in individual trie: {len(trie.get_all_fragments())}")
            merge_tries(master_trie, trie)
        except Exception as e:
            print(f"error merging trie from file {output_file}: {e}")

    print(f"number of sequences in master trie after merging: {len(master_trie.get_all_fragments())}")

    # print the number of fragments in the trie that have at least ten protein ids
    print("number of fragments with at least ten protein ids:")
    print_fragments_with_proteins(master_trie)

    # delete generated tries
    try:
        shutil.rmtree(temp_dir)
    except Exception as e:
        print(f"error deleting temporary directory '{temp_dir}': {e}")

    # construct a base filename from round and sample_size (e.g., "5r1k")
    filename_base = f"{args.round}{args.sample_size}"

    # save merged trie to a pickle file in the output directory with name like "5r1k.pkl"
    output_trie_path = os.path.join(args.output, f"{filename_base}.pkl")
    try:
        with open(output_trie_path, 'wb') as f:
            pickle.dump(master_trie, f)
        print(f"merged trie saved to {output_trie_path}")
    except Exception as e:
        print(f"error saving merged trie: {e}")

    # dump merged trie results to a csv file in the output directory with name like "5r1k.csv"
    output_csv_path = os.path.join(args.output, f"{filename_base}.csv")
    try:
        results_array = master_trie.get_all_fragments()
        with open(output_csv_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['fragment', 'protein_counts'])  # writing header
            for fragment, protein_counter in results_array:
                writer.writerow([fragment, json.dumps(protein_counter)])
        print(f"results csv saved to {output_csv_path}")
    except Exception as e:
        print(f"error writing results to csv: {e}")