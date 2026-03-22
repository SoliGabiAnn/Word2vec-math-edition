import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from parsing_token.parsing_dict import DataPipline
from parsing_token.one_hot_coder import DictionaryOneHotEncoder
import numpy as np
from model_pipe.Word2vec import Word2Vec


def test_model(model, one_hot_dict, topk=5):
    """
    Test the trained Word2Vec model by finding similar words.
    
    Args:
        model (Word2Vec): Trained Word2Vec model
        one_hot_dict (DictionaryOneHotEncoder): One-hot encoder with vocabulary
        topk (int): Number of similar words to display for each test word
    """
    print("\n" + "="*60)
    print("MODEL TESTING: Finding Similar Words")
    print("="*60)
    
    # Get a sample of words to test (choose first 10 or fewer if vocab is smaller)
    num_test_words = min(10, len(one_hot_dict.unique_keys))
    test_word_ids = one_hot_dict.unique_keys[:num_test_words]
    
    for word_id in test_word_ids:
        word_label = one_hot_dict.mapping[word_id]
        similar_words = model.get_most_similar_words(word_id, topk=topk)
        
        print(f"\nWord: '{word_label}' (ID: {word_id})")
        print(f"  Most similar words:")
        for sim_id, similarity in similar_words:
            sim_label = one_hot_dict.mapping[sim_id]
            print(f"    - '{sim_label}' (ID: {sim_id}, similarity: {similarity:.4f})")


if __name__ == "__main__":
    pipe=DataPipline('/home/solic/Word2vec-math-edition/data/short_corpus.txt')
    data, neg=pipe.run(4,1e-1,30,5)

    one_hot_dict=DictionaryOneHotEncoder(pipe.int_to_voc)

    # Model initialization
    vocab_size = pipe.voc_size 
    d = 100
    model = Word2Vec(vocab_size,one_hot_dict, d=d, lr=0.01)

    print(f"Vocabulary size: {vocab_size}")
    print(f"Embedding dimension: {d}")
    print(f"Number of training pairs: {len(data)}")

    # Training
    print("\nStarting training...")
    num_epochs = 3

    for epoch in range(num_epochs):
        total_loss = 0.0
        n_steps = 0

        for center, context in data:
            neg_samples=[t[1] for t in neg if neg[0]==center]
            neg_sampl_oh=[one_hot_dict.encode(sampl) for sampl in neg_samples]
            center_one_hot,context_one_hot = one_hot_dict.encode(center),one_hot_dict.encode(context)
            loss = model.train_step(center_one_hot, context_one_hot, neg_sampl_oh)
            total_loss += loss
            n_steps += 1

        avg_loss = total_loss / n_steps
        print(f"Epoch {epoch}, avg loss {avg_loss:.4f}")

    # Save weights
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    results_dir = os.path.join(project_root, 'results')
    
    # Create results directory if it doesn't exist
    os.makedirs(results_dir, exist_ok=True)
    
    weights_path = os.path.join(results_dir, 'trained_weights.npz')
    model.save_weights(weights_path)

    # Test the model
    test_model(model, one_hot_dict, topk=5)







