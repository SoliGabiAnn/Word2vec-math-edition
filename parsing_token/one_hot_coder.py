import numpy as np

class DictionaryOneHotEncoder:
    def __init__(self, mapping_dict):
        """
        mapping_dict: {101: "Apple", 102: "Banana", ...}
        """
        self.mapping = mapping_dict
        # Get sorted unique keys to ensure consistent vector indexing
        self.unique_keys = sorted(list(mapping_dict.keys()))
        self.depth = len(self.unique_keys)
        
        # Internal maps for O(1) lookups
        # Number -> Index in vector
        self.num_to_idx = {num: i for i, num in enumerate(self.unique_keys)}
        # Index in vector -> Number
        self.idx_to_num = {i: num for i, num in enumerate(self.unique_keys)}

    def encode(self, number):
        """Number -> One-Hot Vector"""
        if number not in self.num_to_idx:
            raise ValueError(f"ID {number} not found in mapping dictionary.")
            
        vector = np.zeros(self.depth)
        vector[self.num_to_idx[number]] = 1
        return vector

    def decode_to_num(self, vector):
        """One-Hot Vector -> Number (ID)"""
        idx = np.argmax(vector)
        return self.idx_to_num[idx]

    def decode_to_word(self, vector):
        """One-Hot Vector -> Word (Label)"""
        number = self.decode_to_num(vector)
        return self.mapping[number]