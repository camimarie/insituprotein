"""
Description: This file contains the implementation of a trie data structure that stores fragments.
"""

class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end = False
        self.protein_counter = {}  # dictionary to track occurrences per protein ID

class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, protein_id, fragment):
        """
        Inserts a fragment into the trie and updates the protein counter.
        """
        node = self.root
        for char in fragment:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        
        node.is_end = True
        node.protein_counter[protein_id] = node.protein_counter.get(protein_id, 0) + 1 # update protein counter per fragment

    def search(self, fragment):
        """
        Returns True if the fragment exists in the trie, False otherwise.
        """
        node = self.root
        for char in fragment:
            if char not in node.children:
                return False
            node = node.children[char]
        return node.is_end

    def get_all_fragments(self):
        """
        Returns a list of tuples: (fragment, {protein_id: count})
        """
        fragments = []

        def dfs(node, current_fragment):
            if node.is_end:
                fragments.append((current_fragment, node.protein_counter))
            for char, child in node.children.items():
                dfs(child, current_fragment + char)

        dfs(self.root, "")
        return fragments

    def get_fragments_with_counts(self):
        """
        Returns fragments along with their protein occurrence counts.
        """
        fragments = []
        for fragment, protein_counter in self.get_all_fragments():
            fragments.append((fragment, protein_counter))
        return fragments