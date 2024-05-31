import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import tensorflow as tf
from helper_functions import *
importTensorflow(memory=4090)
import numpy as np
import time

with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)

stoi = { ch:i for i, ch in enumerate(chars) }
itos = { i:ch for i, ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: "".join([itos[i] for i in l])

data = tf.constant(encode(text), dtype=tf.int64)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

tf.random.set_seed(1337)
batch_size = 64
block_size = 256

learning_rate = 3e-4
max_iters = 5000
eval_interval = 500
eval_iters = 200
n_embd = 384
# head_size = 16
n_head = 4

def get_batch(split):
    data = train_data if split == "train" else val_data
    ix = np.random.randint(low = 0, high = len(data) - block_size, size=(batch_size, ))
    x = tf.stack([data[i:i+block_size] for i in ix])
    y = tf.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y

xb, yb = get_batch('train')
ix = np.random.randint(low = 0, high = len(data) - block_size, size=(4, ))
x = tf.stack([data[i:i+block_size] for i in ix])


class Head(tf.keras.Model):
    def __init__(self, head_size):
        super().__init__()
        self.key = tf.keras.layers.Dense(head_size, use_bias=False)
        self.query = tf.keras.layers.Dense(head_size, use_bias=False)
        self.value = tf.keras.layers.Dense(head_size, use_bias=False)
        self.tril = tf.constant(np.tril(tf.ones((block_size, block_size), dtype=tf.float32)))

    def call(self, x):
        B, T, C = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2]
        k = self.key(x)
        q = self.query(x)
        wei = q @ tf.transpose(k, perm=[0, 2, 1]) * tf.cast(C, dtype=tf.float32)**(-0.5)
        wei = tf.where(self.tril[:T,:T]==0, x=float('-inf'), y=wei)
        wei = tf.nn.softmax(wei, axis=-1)
        v = self.value(x)
        out = wei @ v
        return out

class MultiHeadAttention(tf.keras.Model):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = [Head(head_size) for _ in range(num_heads)]
    
    def call(self, x):
        return tf.concat([h(x) for h in self.heads], axis=-1)

class FeedForward(tf.keras.Model):
    def __init__(self, n_embd):
        super().__init__()
        self.model = tf.keras.models.Sequential([
            tf.keras.layers.Dense(n_embd),
            tf.keras.layers.ReLU()
        ])
    
    def call(self, x):
        return self.model(x)

class Block(tf.keras.Model):
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd//n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
    
    def call(self, x):
        x = self.sa(x)
        x = self.ffwd(x)
        return x

class BigramLanguageModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = tf.keras.layers.Embedding(vocab_size, n_embd)
        self.position_embedding_table = tf.keras.layers.Embedding(block_size, n_embd)
        # self.sa_head = MultiHeadAttention(4, n_embd//4)
        # self.ffwd = FeedForward(n_embd)
        self.blocks = tf.keras.models.Sequential([
            Block(n_embd, n_head),
            Block(n_embd, n_head),
            Block(n_embd, n_head)
        ])
        self.lm_head = tf.keras.layers.Dense(vocab_size)
    
    def call(self, idx, targets=None):
        B, T = tf.shape(idx)[0], tf.shape(idx)[1]
        token_emb = self.token_embedding_table(idx) # (B,T,C)
        pos_emb = self.position_embedding_table(tf.range(T))
        
        x = token_emb + pos_emb
        # x = self.sa_head(x)
        # x = self.ffwd(x)
        x = self.blocks(x)

        logits = self.lm_head(x) # (B,T,vocab_size)
        loss = None
        if targets is not None:
            lossF = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
            # print(idx.shape, targets.shape, logits.shape)
            loss = lossF(targets, logits)
        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_new = idx[:, -block_size:]
            logits, loss = self(idx_new)
            # print(logits.shape)
            # print(logits)
            logits = logits[:, -1, :]
            probs = tf.nn.softmax(logits, axis=-1)
            idx_next = tf.random.categorical(probs, num_samples = 1)
            idx = tf.concat([idx, idx_next], axis=-1)
            # print(_, idx)
        return idx
    
m = BigramLanguageModel()
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

@tf.function
def train_step(xb, yb):
    with tf.GradientTape() as tape:
        logits, loss = m(xb, yb)
    gradients = tape.gradient(loss, m.trainable_variables)
    optimizer.apply_gradients(zip(gradients, m.trainable_variables))
    return loss

def estimate_loss(model, eval_iters, get_batch):
    results = {}
    for split in ['train', 'val']:
        losses = []
        for _ in range(eval_iters):
            xb, yb = get_batch(split)
            _, loss = model(xb, yb)
            losses.append(loss.numpy())
        results[split] = tf.reduce_mean(losses)
    return results

for iter in range(max_iters):
    if iter % eval_interval == 0:
        losses = estimate_loss(m, eval_iters, get_batch)
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    xb, yb = get_batch('train')
    loss = train_step(xb, yb)

print()
context = tf.zeros((1, 1), dtype=tf.int64)
generated_sequence = m.generate(context, max_new_tokens=500)
print(decode(generated_sequence[0].numpy()))