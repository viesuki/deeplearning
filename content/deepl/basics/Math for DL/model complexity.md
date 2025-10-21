---
title: Model Complexity
tags: [math]
created: 2024-10-11
modified: 2024-10-12
---

> [!tip]
> When reading papers, it's very common to come across the computational complexity and the total parameters of a model. So I want to figure it out how to compute these two and how some packeges compute them. Let's dive in.
> Packges: [torchinfo](https://github.com/TylerYep/torchinfo); [ptflops](https://github.com/sovrasov/flops-counter.pytorch); [thop](https://github.com/Lyken17/pytorch-OpCounter);


# Computational Complexity

* **Terminology:**
- *MACs* = multiply-accumulate ops;
- *FLOPs* = floating point operations;
Many sources treat **1 MAC ≈ 2 FLOPs** (one mul + one add). Accounting conventions differ—state your convention when comparing.

> [!info] In most cases of modern packges, MACs is more often used.

* **Output size for 2D conv:**
  `H_out = floor((H + 2p - k_h)/s) + 1`, `W_out = floor((W + 2p - k_w)/s) + 1`.

* **Typical forward-pass MACs (inference):**

  * Linear: `MACs ≈ d_in * d_out`
  * Conv2d: `MACs ≈ H_out * W_out * C_out * (k_h * k_w * C_in)`
  * Grouped conv: divide by `groups`
  * Depthwise separable:
    `MACs_depthwise ≈ H_out*W_out*(k_h*k_w*C_in)`
    `MACs_pointwise ≈ H_out*W_out*(C_in*C_out)`
  * Self-Attention (seq n, dim d, heads h):
    `≈ 4*n*d^2 (projections) + 2*n^2*d (attn scores & apply V)`
  * Transformer FFN (d→4d→d): `≈ 8*n*d^2`
  * **Training** usually costs ≈ 2–3× the **forward** MACs.



# Total Parameters

Computing total parameters is very simple. 

## Linear (Fully-Connected)

**Formula:** $ {\color{red}params = d_{in} * d_{out} + d_{out}}$ (add bias only if enabled)

**Quick check (summary):**

```python
layer = nn.Linear(512, 1000, bias=True)
summary(layer, input_size=(1, 512))  
# (batch, d_in) → params should be 512*1000+1000 = 51300
```


## Standard 2D Convolution (no groups)

**Formula:** $\color{red}{params = k_h * k_w * C_{in} * C_{out} + C_{out}}$

**Quick check (summary):**

```python
conv = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1)
summary(conv, input_size=(1, 3, 224, 224))  # params should be 3*3*3*64 + 64 = 1792
```


## Grouped Convolution (groups = g)

**Formula:** $\color{red}{params = (k_h * k_w * C_{in} * C_{out}) / g + C_{out}}$

**Quick check (summary):**

```python
gconv = nn.Conv2d(32, 64, kernel_size=3, padding=1, groups=2)
summary(gconv, input_size=(1, 32, 56, 56))  # 3*3*32/2*64 + 64 = 9280
```

## Depthwise-Separable Convolution

**Formula:**

* Depthwise: $\color{red}{k_h * k_w * C_{in} + C_{in}}$
* Pointwise (1×1): $\color{red}{C_{in} * C_{out} + C_{out}}$
* **Total = depthwise + pointwise**

**Quick check (summary):**

```python
depthwise_separable = nn.Sequential(
    nn.Conv2d(32, 32, kernel_size=3, padding=1, groups=32),  # depthwise
    nn.Conv2d(32, 64, kernel_size=1)                         # pointwise
)
summary(depthwise_separable, input_size=(1, 32, 56, 56))
# (3*3*32 + 32) + (32*64 + 64) = 2432
```

---

## BatchNorm / LayerNorm (trainable parts only)

**Rule of thumb:** Count only learnable γ, β (running stats are buffers).

* BatchNorm2d(C): $\color{red}{params = 2 * C}$
* LayerNorm(D): $\color{red}{params = 2 * D}$

**Quick check (summary):**

```python
bn = nn.BatchNorm2d(64)
summary(bn, input_size=(1, 64, 32, 32))   # expect 2*64 params (γ, β)

ln = nn.LayerNorm(128)
summary(ln, input_size=(1, 10, 128))      # last dim = 128 → expect 2*128 params
```


## LSTM (per layer; shared across time)

**Formula:** 
Theory: $\color{red}{params = 4 * (d_{in} * h + h * h + h)}$ (4 gates; add bias per gate)
Practice: $\color{red}{params = 4 * (d_{in} * h + h * h + 2*h)}$ (pytorch uses combined matrices)

**Quick check (summary):**

```python
class LSTMWrap(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=128, hidden_size=256, num_layers=1, batch_first=True)
    def forward(self, x):
        y, _ = self.lstm(x)
        return y

model = LSTMWrap()
summary(model, input_size=(32, 10, 128))  # (batch, seq, d_in)
# 4*(128*256 + 256*256 + 2*256) = 395264
```


## GRU (per layer; shared across time)

**Formula:** $\color{red}{params = 3 * (d_{in} * h + h * h + [2]*h)}$

**Quick check (summary):**

```python
class GRUWrap(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru = nn.GRU(input_size=128, hidden_size=256, num_layers=1, batch_first=True)
    def forward(self, x):
        y, _ = self.gru(x)
        return y

model = GRUWrap()
summary(model, input_size=(32, 10, 128))
```


## Multi-Head Self-Attention (params only; not counting MLP)

**Heuristic:** $params = 4 * d^2 + 4*d$.

**Quick check (summary):**

```python
class MHAWrap(nn.Module):
    def __init__(self):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim=512, num_heads=8, batch_first=True)
    def forward(self, x):
        y, _ = self.mha(x, x, x)
        return y

model = MHAWrap()
summary(model, input_size=(32, 16, 512))  # (batch, seq, d)
# 4 * 512 * 512 + 4 * 512 = 1050624
```


## Transformer Full encoder layer:

```python
enc = nn.TransformerEncoderLayer(d_model=512, nhead=8, dim_feedforward=2048, batch_first=True)
summary(enc, input_size=(32, 16, 512))
# 3,152,384
```



---

# 3) Big-O (algorithmic) complexity

