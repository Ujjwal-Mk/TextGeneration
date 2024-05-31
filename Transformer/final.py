import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import tensorflow as tf
from helper_functions import *
importTensorflow(memory=4090)
import numpy as np
import time
from tensorflow.keras import layers

# Hyperparameters
batch_size = 64  # Number of sequences processed in parallel
block_size = 256  # Maximum context length for predictions
max_iters = 2000
eval_interval = 500
learning_rate = 3e-4
eval_iters = 200
n_embd = 384
n_head = 6
n_layer = 6
dropout = 0.2

# Set random seed for reproducibility
tf.random.set_seed(1337)

# Load and preprocess text data
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# Create character mappings
chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

# Convert text to tensor and split into training and validation sets
data = tf.constant(encode(text), dtype=tf.int64)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

# Function to generate batches of data
def get_batch(split):
    data = train_data if split == "train" else val_data
    ix = np.random.randint(len(data) - block_size, size=batch_size)
    x = tf.stack([data[i:i+block_size] for i in ix])
    y = tf.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y

# Function to estimate loss
def estimate_loss():
    results = {}
    for split in ['train', 'val']:
        losses = []
        for _ in range(eval_iters):
            xb, yb = get_batch(split)
            logits, loss = model(xb, yb, training=False)
            losses.append(loss.numpy())
        results[split] = tf.reduce_mean(losses)
    return results

# Define the Transformer model components
class Head(tf.keras.layers.Layer):
    def __init__(self, head_size):
        super().__init__()
        self.key = layers.Dense(head_size, use_bias=False)
        self.query = layers.Dense(head_size, use_bias=False)
        self.value = layers.Dense(head_size, use_bias=False)
        self.tril = tf.constant(np.tril(np.ones((block_size, block_size), dtype=np.float32)))
        self.dropout = layers.Dropout(dropout)

    def call(self, x):
        B, T, C = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2]
        k = self.key(x)
        q = self.query(x)
        wei = tf.matmul(q, k, transpose_b=True) * (tf.cast(C, dtype=tf.float32)**-0.5)
        wei = tf.where(self.tril[:T, :T] == 0, float('-inf'), wei)
        wei = tf.nn.softmax(wei, axis=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        out = tf.matmul(wei, v)
        return out

class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = [Head(head_size) for _ in range(num_heads)]
        self.proj = layers.Dense(n_embd)
        self.dropout = layers.Dropout(dropout)

    def call(self, x):
        out = tf.concat([h(x) for h in self.heads], axis=-1)
        out = self.proj(out)
        out = self.dropout(out)
        return out

class FeedForward(tf.keras.layers.Layer):
    def __init__(self, n_embd):
        super().__init__()
        self.net = tf.keras.Sequential([
            layers.Dense(4 * n_embd),
            layers.ReLU(),
            layers.Dense(n_embd),
            layers.Dropout(dropout),
        ])

    def call(self, x):
        return self.net(x)

class Block(tf.keras.layers.Layer):
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = layers.LayerNormalization()
        self.ln2 = layers.LayerNormalization()

    def call(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class GPTLanguageModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = layers.Embedding(vocab_size, n_embd)
        self.position_embedding_table = layers.Embedding(block_size, n_embd)
        self.blocks = tf.keras.Sequential([Block(n_embd, n_head) for _ in range(n_layer)])
        self.ln_f = layers.LayerNormalization()
        self.lm_head = layers.Dense(vocab_size)

    def call(self, idx, targets=None):
        B, T = tf.shape(idx)[0], tf.shape(idx)[1]
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(tf.range(T))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        if targets is None:
            return logits
        else:
            loss = tf.keras.losses.sparse_categorical_crossentropy(targets, logits, from_logits=True)
            return logits, tf.reduce_mean(loss)

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits = self(idx_cond)
            logits = logits[:, -1, :]
            probs = tf.nn.softmax(logits, axis=-1)
            idx_next = tf.random.categorical(probs, num_samples=1)
            idx = tf.concat([idx, idx_next], axis=1)
        return idx

# Initialize the model and optimizer
model = GPTLanguageModel()
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

# Function to train the model
@tf.function
def train_step(xb, yb):
    with tf.GradientTape() as tape:
        logits, loss = model(xb, yb)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss

# Training loop
for iter in range(max_iters):
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    xb, yb = get_batch('train')
    loss = train_step(xb, yb)

# Generate text from the model
context = tf.zeros((1, 1), dtype=tf.int64)
generated_sequence = model.generate(context, max_new_tokens=100)

with open('output.txt', 'wb') as f:
    f.write(decode(generated_sequence.numpy()[0]).encode('utf-8'))
