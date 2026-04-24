"""
Toy Transformer Block — Step-by-Step Demo

This script walks through one complete transformer block in plain English.
We take a simple 3-word sentence and show exactly how a transformer
converts words into numbers, finds relationships between them,
and produces a rich, context-aware output for each word.

No training. No backpropagation. Just the forward pass — step by step.
"""

import numpy as np

# Helper: softmax converts raw numbers into probabilities (all values 0–1, sum to 1)
def softmax(x):
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / np.sum(e_x, axis=-1, keepdims=True)

# Helper: layer norm rescales each word's vector to have zero mean and unit variance
# This keeps numbers stable and prevents any single value from dominating
def layer_norm(x):
    return (x - x.mean(axis=1, keepdims=True)) / (x.std(axis=1, keepdims=True) + 1e-6)

# Fix random seed so results are the same every run (good for teaching)
np.random.seed(42)

# ==============================================================================
# STEP 1: Tokenization
# The model cannot read words — it reads numbers.
# So the first step is to split the sentence into individual words (called tokens).
# Each token will eventually be converted to a number, and then to a vector.
# ==============================================================================
sentence = "I love NLP"
tokens = sentence.split()
print("Step 1 — Tokenization")
print(f"  Sentence : '{sentence}'")
print(f"  Tokens   : {tokens}")
print()

# ==============================================================================
# STEP 2: Vocabulary & Token IDs
# We build a dictionary that gives each unique word a number (its ID).
# Words are sorted alphabetically so the numbering is always consistent.
# Example for 'I love NLP':
#   sorted words -> ['I', 'NLP', 'love']
#   vocab        -> {'I': 0, 'NLP': 1, 'love': 2}
#   sentence IDs -> [0, 2, 1]  (I=0, love=2, NLP=1)
# This is the same idea as a dictionary index in a textbook.
# ==============================================================================
vocab = {w: i for i, w in enumerate(sorted(set(tokens)))}
token_ids = [vocab[w] for w in tokens]
print("Step 2 — Vocabulary & Token IDs")
print("  Each unique word gets a unique integer ID (sorted alphabetically):")
for word, idx in vocab.items():
    print(f"    '{word}' -> ID {idx}")
print(f"  Sentence token IDs: {token_ids}  (maps each word in order to its ID)")
print()

# ==============================================================================
# STEP 3: Word Embeddings
# A token ID is just a number — it has no meaning on its own.
# We need to convert it into a rich vector that captures the word's "meaning".
# The embedding matrix is a lookup table: each row is one word's meaning vector.
#
# Each vector has embed_dim = 4 numbers, called dimensions.
# Think of each dimension as a hidden trait of the word, for example:
#   Dim 1 — is this word more about an action or a thing?
#   Dim 2 — is this word positive or negative in tone?
#   Dim 3 — is this word abstract (idea) or concrete (object)?
#   Dim 4 — is this word technical or everyday?
# (In real models these are learned automatically — here they are random.)
# ==============================================================================
vocab_size = len(vocab)
embed_dim = 4  # Use 4 dimensions to keep output readable in class
embedding_matrix = np.random.randn(vocab_size, embed_dim)

# Lookup: retrieve the embedding vector for each word in the sentence
embeddings = np.array([embedding_matrix[i] for i in token_ids])

print("Step 3 — Word Embeddings")
print(f"  Each word is translated into a vector of {embed_dim} numbers (its 'meaning' in math):")
for word, idx in vocab.items():
    print(f"    '{word}' (ID {idx}) -> {embedding_matrix[idx]}")
print(f"  Embeddings for our sentence words (in order):")
for token, emb in zip(tokens, embeddings):
    print(f"    '{token}' -> {emb}")
print()

# ==============================================================================
# STEP 4: Positional Encoding
# Transformers process ALL words at the same time (in parallel).
# This means they have no built-in sense of word order.
# We fix this by adding a "position signal" to each word's vector.
# Words at different positions get different signals, so the model
# can tell that 'I' came first, 'love' second, 'NLP' third.
# We use a sinusoidal pattern (sin/cos waves) — a simple, fixed formula.
# ==============================================================================
seq_len = len(tokens)
positional_encoding = np.zeros((seq_len, embed_dim))
for pos in range(seq_len):
    for i in range(embed_dim):
        if i % 2 == 0:
            positional_encoding[pos, i] = np.sin(pos / (10000 ** (i / embed_dim)))
        else:
            positional_encoding[pos, i] = np.cos(pos / (10000 ** (i / embed_dim)))

# Add position signal to each word's embedding
input_with_pos = embeddings + positional_encoding

print("Step 4 — Positional Encoding")
print("  Position signals (added to each word so the model knows word order):")
for pos, (token, vec) in enumerate(zip(tokens, positional_encoding)):
    print(f"    Position {pos} ('{token}') -> {vec}")
print("  Word embeddings + position signal = actual transformer input:")
for token, vec in zip(tokens, input_with_pos):
    print(f"    '{token}' -> {vec}")
print()

# ==============================================================================
# STEP 5: Self-Attention — Q, K, V
# This is the heart of the transformer.
# The idea: each word looks at every other word in the sentence and decides
# how much attention to pay to each one — based on meaning, not just position.
#
# To do this, each word is projected into three different roles:
#   Q (Query)  — "What am I looking for? What context do I need?"
#   K (Key)    — "What do I have to offer? What topic do I represent?"
#   V (Value)  — "If someone attends to me, here is the information I share."
#
# Example in English:
#   The word 'NLP' (Query) asks: "What should I pay attention to?"
#   The word 'love' (Key) says: "I represent positive sentiment."
#   If 'NLP' attends to 'love', it receives 'love's Value — its actual content.
# ==============================================================================
W_Q = np.random.randn(embed_dim, embed_dim)
W_K = np.random.randn(embed_dim, embed_dim)
W_V = np.random.randn(embed_dim, embed_dim)

Q = input_with_pos @ W_Q  # Each word's "question"
K = input_with_pos @ W_K  # Each word's "label"
V = input_with_pos @ W_V  # Each word's "actual content"

print("Step 5 — Self-Attention: Q, K, V")
print("  Q (Query) — what each word is 'asking' / looking for:")
for token, vec in zip(tokens, Q):
    print(f"    '{token}' -> {vec}")
print("  K (Key) — what each word 'represents' / offers:")
for token, vec in zip(tokens, K):
    print(f"    '{token}' -> {vec}")
print("  V (Value) — the actual content each word will share if attended to:")
for token, vec in zip(tokens, V):
    print(f"    '{token}' -> {vec}")
print()

# ==============================================================================
# STEP 6: Attention Scores & Weights
# We compute how much each word should attend to every other word.
# Score(i, j) = dot product of Q[i] and K[j], divided by sqrt(embed_dim).
# A higher score means word i finds word j more relevant.
# Dividing by sqrt(embed_dim) prevents scores from becoming too large.
# Softmax then turns these raw scores into probabilities (0 to 1, sum = 1).
# ==============================================================================
d_k = Q.shape[-1]
scores = Q @ K.T / np.sqrt(d_k)
attn_weights = softmax(scores)

print("Step 6 — Attention Scores & Weights")
print("  How much does each word attend to every other word?")
print("  (Read row by row: e.g., row 'I' shows how much 'I' attends to each word)")
header = "           " + "  ".join(f"{t:>8}" for t in tokens)
print("  Raw scores (before softmax):")
print(header)
for i, row in enumerate(scores):
    print(f"    '{tokens[i]}' -> " + "  ".join(f"{v:8.4f}" for v in row))
print("  Attention weights (after softmax, each row sums to 1.0):")
print(header)
for i, row in enumerate(attn_weights):
    print(f"    '{tokens[i]}' -> " + "  ".join(f"{v:8.4f}" for v in row) + f"  [sum={row.sum():.2f}]")
print()

# ==============================================================================
# STEP 7: Attention Output
# Each word's new representation is a weighted mix of all Value vectors.
# If 'I' attends 60% to itself, 30% to 'love', 10% to 'NLP',
# its output = 0.6 * V['I'] + 0.3 * V['love'] + 0.1 * V['NLP']
# Result: every word now carries context from the whole sentence.
# ==============================================================================
attn_output = attn_weights @ V
print("Step 7 — Attention Output")
print("  Each word's vector is now a blend of all words, weighted by attention.")
print("  Each word now 'knows' about the other words in the sentence:")
for token, vec in zip(tokens, attn_output):
    print(f"    '{token}' -> {vec}")
print()

# ==============================================================================
# STEP 8: Add & Norm (after attention)
# We add the original word vectors back to the attention output (residual connection).
# Why? So the model doesn't forget the original word — it builds on it.
# Then we normalize: scale all numbers so each word's vector has mean=0, std=1.
# Why normalize? To keep numbers stable and prevent any one value from exploding.
# ==============================================================================
attn_residual = input_with_pos + attn_output
attn_norm = layer_norm(attn_residual)
print("Step 8 — Add & Norm (after attention)")
print("  Original input + attention output, then normalized per word:")
for token, vec in zip(tokens, attn_norm):
    print(f"    '{token}' -> {vec}")
print()

# ==============================================================================
# STEP 9: Feed-Forward Network (FFN)
# After attention, each word's vector is passed through a small neural network
# applied independently to each word.
# Layer 1: expands the vector to a larger space (more room to think) + ReLU
#   ReLU simply zeroes out any negative number — it introduces non-linearity.
# Layer 2: compresses it back to the original size.
# Think of it as: attention found relationships between words;
# FFN now deeply processes each word's updated meaning on its own.
# ==============================================================================
ffn_hidden_dim = 6  # Expand to 6 dimensions internally, then project back to 4
W1 = np.random.randn(embed_dim, ffn_hidden_dim)
b1 = np.random.randn(ffn_hidden_dim)
W2 = np.random.randn(ffn_hidden_dim, embed_dim)
b2 = np.random.randn(embed_dim)

ffn_hidden = np.maximum(0, attn_norm @ W1 + b1)  # Layer 1 + ReLU
ffn_output = ffn_hidden @ W2 + b2               # Layer 2 — back to embed_dim

print("Step 9 — Feed-Forward Network")
print("  Each word's vector is individually transformed by a 2-layer network:")
for token, vec in zip(tokens, ffn_output):
    print(f"    '{token}' -> {vec}")
print()

# ==============================================================================
# STEP 10: Add & Norm (after FFN) — Final Output
# Same pattern as after attention: add the previous output back (residual),
# then normalize. This is the final output of one complete transformer block.
#
# What does this output represent in plain English?
# Each word's vector no longer just means that word in isolation.
# It now encodes the word IN CONTEXT of the whole sentence.
# For example, 'love' now knows it is surrounded by 'I' and 'NLP'.
# This output can be fed into the next transformer block, or used to
# predict the next word, classify the sentence, answer a question, etc.
# ==============================================================================
ffn_residual = attn_norm + ffn_output
ffn_norm = layer_norm(ffn_residual)

print("Step 10 — Add & Norm (after FFN)")
print("  FFN output + residual, then normalized — this is the final output:")
for token, vec in zip(tokens, ffn_norm):
    print(f"    '{token}' -> {vec}")
print()

print("=" * 60)
print("FINAL OUTPUT — One Complete Transformer Block")
print(f"Input : '{sentence}'  ({seq_len} words, each as a {embed_dim}-dimensional vector)")
print(f"Output: {seq_len} context-aware vectors — each word now understands its surroundings")
print("=" * 60)
for token, vec in zip(tokens, ffn_norm):
    print(f"  '{token}' -> {vec}")
print()
print("  In plain English: 'I', 'love', and 'NLP' are no longer just isolated words.")
print("  Each word's vector now carries meaning shaped by the full sentence context.")
