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
batch_size = 4
block_size = 8

def get_batch(split):
    data = train_data if split == "train" else val_data
    ix = np.random.randint(low = 0, high = len(data) - block_size, size=(batch_size, ))
    x = tf.stack([data[i:i+block_size] for i in ix])
    y = tf.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y

xb, yb = get_batch('train')
ix = np.random.randint(low = 0, high = len(data) - block_size, size=(4, ))
x = tf.stack([data[i:i+block_size] for i in ix])


class BigramLanguageModel(tf.keras.Model):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = tf.keras.layers.Embedding(vocab_size, vocab_size)
    
    def call(self, idx, targets=None):
        logits = self.token_embedding_table(idx)
        loss = None
        if targets is not None:
            lossF = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
            # print(idx.shape, targets.shape, logits.shape)
            loss = lossF(targets, logits)
        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            logits, loss = self(idx)
            # print(logits.shape)
            # print(logits)
            logits = logits[:, -1, :]
            probs = tf.nn.softmax(logits, axis=-1)
            idx_next = tf.random.categorical(probs, num_samples = 1)
            idx = tf.concat([idx, idx_next], axis=-1)
            # print(_, idx)
        return idx
    
m = BigramLanguageModel(vocab_size)
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-2)

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

max_iters = 3000
eval_interval = 300
eval_iters = 300

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