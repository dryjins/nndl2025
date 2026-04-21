# MiniGPT: Character-Level Transformer Language Model

This document describes the design of a small GPT-style, decoder-only Transformer trained as a character-level language model. The goal is to implement a clean, fully working model that matches the standard GPT architecture while keeping the code and hyperparameters small enough for educational experiments.

---

## 1. Architecture Overview

### 1.1 High-Level Diagram (Text Form)

Input characters → Token IDs → **Token Embedding**  
+ **Positional Embedding** → Embedded sequence `X ∈ ℝ^{B×T×d_model}`

`X` → **Stack of L Transformer Decoder Blocks** (pre-norm):
- LayerNorm → Masked Multi-Head Self-Attention → Residual
- LayerNorm → Position-wise Feed-Forward Network (FFN) → Residual

Final hidden states → Final LayerNorm → **Linear Output Head** → Logits over vocabulary (one distribution per time step)

At training time, the model receives a sequence \(x_1, …, x_T\) and is trained to predict the next token at each position. At generation time, the model autoregressively appends one token at a time using causal masking.

---

## 2. Component Breakdown

### 2.1 Tokenizer

- **Type**: Character-level tokenizer.
- **Vocabulary**:
  - All distinct characters in the corpus (letters, digits, punctuation, whitespace).
  - Two dictionaries: `stoi: char → int`, `itos: int → char`.
- **Function**:
  - Training: map raw text to a sequence of integer IDs.
  - Inference: map generated IDs back to text.

This keeps the implementation simple and avoids dealing with BPE/WordPiece for this assignment.

### 2.2 Input Embedding + Positional Encoding

- **Token Embedding**:
  - `nn.Embedding(vocab_size, d_model)`
  - Maps each token ID to a dense vector of size `d_model`.

- **Positional Embedding** (learned):
  - `nn.Embedding(block_size, d_model)`
  - Positions are `0, 1, …, T-1` (with `T ≤ block_size`).
  - Position embedding is added element-wise to the token embedding:
    \[
      X = \text{TokenEmb}(idx) + \text{PosEmb}(positions)
    \]

This provides the model with information about token order without using sinusoidal encodings.

### 2.3 Multi-Head Self-Attention Block

- **Input**: `X ∈ ℝ^{B×T×d_model}`.
- **Linear projections**:
  - `Q = X W_Q`, `K = X W_K`, `V = X W_V`
  - Each of shape `(B, T, d_model)`, then reshaped to `(B, n_heads, T, d_k)` with `d_k = d_model / n_heads`.

- **Causal Mask**:
  - Lower-triangular mask `M ∈ {0,1}^{T×T}` with ones for allowed positions and zeros above the diagonal.
  - Attention scores:
    \[
      S = \frac{Q K^\top}{\sqrt{d_k}} \in ℝ^{B×n\_heads×T×T}
    \]
  - Masking:
    \[
      S_{ij} = -\infty \quad \text{if} \quad M_{ij} = 0
    \]
  - Softmax over the last dimension to obtain attention weights.

- **Multi-Head Combination**:
  - Weighted sum: `A = softmax(S) V`.
  - Concatenate heads: `(B, T, n_heads·d_k)` and project back to `(B, T, d_model)` via a final linear layer.

- **Causal Property**:
  - Each position can only attend to itself and earlier positions.
  - This ensures no information leakage from future tokens.

### 2.4 Position-wise Feed-Forward Network (FFN)

- **Form**:
  - `FFN(x) = Linear(d_model → ff_dim) → GELU → Linear(ff_dim → d_model)`
- **Operation**:
  - Applied independently to each position (same weights across all time steps).
  - Increases model capacity and introduces non-linearity after attention.

### 2.5 Residual Connections and Layer Normalization

- **Pre-norm configuration** in each Transformer block:
  1. `x_att = x + Dropout(SelfAttention(LayerNorm(x)))`
  2. `x_out = x_att + Dropout(FFN(LayerNorm(x_att)))`

- **LayerNorm**:
  - `nn.LayerNorm(d_model)` before attention and FFN sub-layers.
  - Stabilizes training and improves gradient flow, especially with multiple layers.

### 2.6 Output Head

- **Final LayerNorm**:
  - `x = LayerNorm_final(x)` to normalize final hidden states.

- **Linear Projection to Vocabulary**:
  - `head = nn.Linear(d_model, vocab_size)`
  - For each position `t`, the model outputs logits `ℝ^{vocab_size}`.
  - This yields a categorical distribution over the next character.

### 2.7 Generation Logic

- **Inputs**:
  - Initial context sequence `idx ∈ ℕ^{1×T_start}` (e.g., prompt as token IDs).
  - Hyperparameters: `max_new_tokens`, `temperature`, `top_k`.

- **Loop** (autoregressive):
  1. Truncate context to the last `block_size` tokens.
  2. Run `forward` to obtain logits.
  3. Take logits at the last position.
  4. If `temperature = 0.0`: greedy decoding (`argmax`).
  5. Else:
     - Scale logits by `1 / temperature`.
     - Optionally apply **top-k filtering**: keep the top `k` logits, set others to `-inf`.
     - Sample from the resulting softmax distribution.
  6. Append the sampled token to the context.
  7. Repeat until `max_new_tokens` are generated.

This implements both deterministic (greedy) and stochastic sampling from the learned language model.

---

## 3. Hyperparameters

The table below lists the main hyperparameters used in our baseline MiniGPT configuration.

| Hyperparameter | Symbol        | Value (example) | Description                                       |
|----------------|---------------|-----------------|---------------------------------------------------|
| Model dimension | `d_model`    | 128             | Hidden size of embeddings and all Transformer layers. |
| Feed-forward dim | `ff_dim`   | 512             | Inner dimension of the FFN (typically 2–4× `d_model`). |
| Number of layers | `n_layers` | 4               | Number of stacked Transformer decoder blocks.     |
| Number of heads  | `n_heads`  | 4               | Attention heads per block (`d_model / n_heads` must be integer). |
| Block size       | `block_size` | 128           | Maximum context length (number of tokens seen at once). |
| Dropout          | `dropout`  | 0.1             | Dropout rate in attention and FFN sub-layers.     |
| Vocabulary size  | `vocab_size` | ~100–200      | Number of unique characters in the corpus.        |
| Batch size       | `batch_size` | e.g. 64       | Number of sequences per mini-batch.               |
| Learning rate    | `lr`       | e.g. 3e-4       | Base learning rate for AdamW optimizer.           |
| Weight decay     | `wd`       | e.g. 1e-2       | L2 regularization via AdamW.                      |
| Training epochs  | `epochs`   | e.g. 10–20      | Full passes over the training data.               |
| Temperature      | `temperature` | 0.0–1.0      | Sampling temperature at inference.                |
| Top-k            | `top_k`    | e.g. 50         | Optional top-k truncation during sampling.        |

(Exact values can be adjusted depending on the dataset size and computational budget; the above are the settings used in our experiments.)

---

## 4. Training Plan

### 4.1 Data Split

- **Corpus**:
  - Single large text file (e.g., literary work or collection of texts).
- **Encoding**:
  - Convert entire corpus to a long sequence of character IDs.
- **Splits**:
  - Train: 80% of the sequence.
  - Validation: 10% of the sequence.
  - Test: 10% of the sequence.
- **Sampling training examples**:
  - Use sliding windows of length `block_size + 1`.
  - For each window:
    - Input `x`: first `block_size` characters.
    - Target `y`: next `block_size` characters (shifted by 1).

### 4.2 Batching Strategy

- Randomly sample starting indices within the training split.
- For each mini-batch:
  - Stack `B = batch_size` windows into tensors `x ∈ ℕ^{B×T}`, `y ∈ ℕ^{B×T}`.
- This produces i.i.d. training batches while preserving local temporal structure inside each window.

### 4.3 Optimization and Schedule

- **Optimizer**: AdamW.
- **Learning rate**: e.g., `3e-4`.
- **Gradient clipping**: optional, e.g., clip global norm at 1.0 for stability.
- **Number of epochs**: train until validation loss plateaus or a fixed number of epochs (e.g., 10–20).

Training objective:

- At each position \(t\), predict the next token \(x_{t+1}\).
- Use cross-entropy loss over the vocabulary, averaged across all positions and batch elements:
  \[
    \mathcal{L} = \frac{1}{B T} \sum_{b=1}^B \sum_{t=1}^T \text{CE}(p_\theta(x_{t+1}^{(b)} \mid x_{\le t}^{(b)}), x_{t+1}^{(b)})
  \]

### 4.4 Sampling and Evaluation

- **During training** (periodically, e.g., every N steps):
  - Generate samples from a fixed prompt using:
    - Greedy decoding (`temperature = 0.0`).
    - Stochastic decoding (`temperature > 0` and optional `top_k`).
  - Qualitatively inspect the generated text for:
    - Syntax-like structure (spacing, punctuation).
    - Stylistic resemblance to the corpus.

- **Final evaluation**:
  - Report training and validation loss curves.
  - Optionally compute perplexity on the validation/test set.
  - Provide representative generated samples for different prompts and decoding settings.

---

*This design document summarizes the architecture and training plan for the MiniGPT assignment. The implementation strictly follows this specification: decoder-only Transformer with causal masking, character-level tokenization, and next-token prediction with cross-entropy loss.*