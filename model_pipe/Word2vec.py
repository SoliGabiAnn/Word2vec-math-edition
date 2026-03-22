import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
from parsing_token.one_hot_coder import DictionaryOneHotEncoder


def sigmoid(x):
    x = np.clip(x, -50, 50)
    return 1.0 / (1.0 + np.exp(-x))


class Word2Vec:
    def __init__(self, vocab_size, one_hot_dict, d=100, lr=0.025):
        """
        Initialize Word2Vec model with Skip-Gram architecture.
        
        Args:
            vocab_size (int): Size of vocabulary
            one_hot_dict (DictionaryOneHotEncoder): One-hot encoder with vocabulary mapping
            d (int): Embedding dimension (default: 100)
            lr (float): Learning rate for optimization (default: 0.025)
        """
        self.vocab_size = vocab_size
        self.d = d
        self.lr = lr
        self.oh_dict = one_hot_dict
        # Input embedding: V × d
        self.W_in  = np.random.uniform(-0.8, 0.8, (vocab_size, d))
        # Output embedding: d × V
        self.W_out = np.random.uniform(-0.8, 0.8, (d, vocab_size))

    def neg_sampling_loss_and_grad(self, center_one_hot, context_one_hot, neg_samples_oh):
        """
        Compute loss and gradients using negative sampling.
        
        Implements Skip-Gram with negative sampling loss function:
        L = -log(sig(u_o · v_c)) - Σ log(sig(-u_k · v_c)) for negative samples k
        
        Args:
            center_one_hot (array): One-hot vector for center word (shape: V,)
            context_one_hot (array): One-hot vector for positive context word (shape: V,)
            neg_samples_oh (list): List of one-hot vectors for negative samples
            
        Returns:
            tuple: (loss, grad_v_c, grad_W_out, center_one_hot, neg_samples_oh)
                - loss (float): Scalar loss value
                - grad_v_c (array): Gradient w.r.t. center embedding (shape: d,)
                - grad_W_out (array): Gradient w.r.t. output embeddings (shape: d, V)
                - center_one_hot (array): Center word one-hot (for gradient computation)
                - neg_samples_oh (list): Negative samples (for gradient computation)
        """

        # Embedding layer: one‑hot @ W_in → (1,) @ (V,d) → (d,)
        # Equivalent to selecting the center word vector
        v_c = center_one_hot @ self.W_in  # shape (d,)

        # Output layer: u_o is the context vector for the positive context
        # context_one_hot is (V,); u_o = W_out @ context_one_hot → (d,)
        u_o = self.W_out @ context_one_hot  # shape (d,)

        # Positive score and sigmoid
        score_pos = np.dot(u_o, v_c)
        sig_pos   = sigmoid(score_pos)
        loss      = -np.log(sig_pos)

        # Gradient w.r.t. center vector (only positive term)
        grad_v_c  = (sig_pos - 1.0) * u_o

        # We need the indices of the positive context to avoid sampling it again
        context_idx = self.oh_dict.decode_to_num(context_one_hot)  # pick one active index
        grad_W_out  = np.zeros_like(self.W_out)

        # Negative terms
        for neg_one_hot in neg_samples_oh:
            # Build one‑hot for negative word k
            u_k = self.W_out @ neg_one_hot  # shape (d,)
            score_neg = np.dot(-u_k, v_c)
            sig_neg   = sigmoid(score_neg)

            loss     += -np.log(sig_neg)
            grad_v_c += -sig_neg * u_k
            k=self.oh_dict.decode_to_num(neg_one_hot)
            grad_W_out[:, k] += -sig_neg * v_c

        # Gradient for positive output vector (k = context_idx)
        grad_W_out[:, context_idx] = (sig_pos - 1.0) * v_c

        return loss, grad_v_c, grad_W_out, center_one_hot, neg_samples_oh

    def train_step(self, center_one_hot, context_one_hot, neg_samples_oh):
        """
        Perform one training step with gradient descent.
        
        Computes loss and gradients, then updates both embedding matrices:
        W_in ← W_in - η * ∇L/∇W_in
        W_out ← W_out - η * ∇L/∇W_out
        
        Args:
            center_one_hot (array): One-hot vector for center word (shape: V,)
            context_one_hot (array): One-hot vector for positive context word (shape: V,)
            neg_samples_oh (list): List of one-hot vectors for negative samples
            
        Returns:
            float: Loss value for this training step
        """
        loss, grad_v_c, grad_W_out, center_one_hot, neg_samples_oh = \
            self.neg_sampling_loss_and_grad(center_one_hot, context_one_hot, neg_samples_oh)

        # Use one‑hot to update input embedding (W_in)
        # (V,)[:, None] @ (d,) → (V,d) gradient
        grad_W_in = center_one_hot[:, None] * grad_v_c[None, :]
        self.W_in  -= self.lr * grad_W_in

        # Update output embedding (W_out) directly from column gradients
        self.W_out -= self.lr * grad_W_out

        return loss

    def get_word_vector(self, word_one_hot):
        """
        Extract learned embedding for a word.
        
        Args:
            word_one_hot (array): One-hot vector for word (shape: V,)
            
        Returns:
            array: Word embedding vector (shape: d,)
        """
        return word_one_hot @ self.W_in  # (d,)

    def save_weights(self, filepath):
        """
        Save trained weights to disk.
        
        Args:
            filepath (str): Path to save weights. Creates .npz file with W_in and W_out
                           Creates parent directories if they don't exist.
        """
        # Create parent directories if needed
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        np.savez(filepath, W_in=self.W_in, W_out=self.W_out)
        print(f"Weights saved to {filepath}")

    def load_weights(self, filepath):
        """
        Load pre-trained weights from disk.
        
        Args:
            filepath (str): Path to .npz file containing W_in and W_out
        """
        data = np.load(filepath)
        self.W_in = data['W_in']
        self.W_out = data['W_out']
        print(f"Weights loaded from {filepath}")

    def get_most_similar_words(self, word_id, topk=5):
        """
        Find the most similar words to a given word using cosine similarity.
        
        Args:
            word_id (int): Vocabulary ID of the query word
            topk (int): Number of similar words to return
            
        Returns:
            list: List of (word_id, similarity_score) tuples sorted by similarity
        """
        # Get the embedding for the query word
        target_embedding = self.W_in[word_id]  # shape (d,)
        
        # Normalize target embedding
        target_norm = np.linalg.norm(target_embedding)
        if target_norm == 0:
            return []
        target_normalized = target_embedding / target_norm
        
        similarities = []
        for i in range(self.vocab_size):
            if i == word_id:
                continue
            
            word_embedding = self.W_in[i]  # shape (d,)
            word_norm = np.linalg.norm(word_embedding)
            
            if word_norm == 0:
                similarity = 0
            else:
                word_normalized = word_embedding / word_norm
                similarity = np.dot(target_normalized, word_normalized)
            
            similarities.append((i, similarity))
        
        # Sort by similarity (descending) and return top-k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:topk]
