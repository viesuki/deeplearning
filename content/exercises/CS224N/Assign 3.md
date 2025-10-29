---
title: Assign 3 Translation
tags: [NLP/NMT, RNN, attention]
created: 2024-10-29
modified: 2024-10-29
---




# Content

In Assignment 2, we will build a neural machine translation model using RNN and attention. Then we will do some analysis of NMT systems.



## Model constructure

![seq2seq](seq2seq.png)

1. Word embedding layer
2. Source CNN layer: 1D in the dimension of sentence length, kernel size = 2
3. **Encoder**: bidirectional LSTM

$$
 h_i^{enc} = [\overleftarrow{h_i^{enc}}, \overrightarrow{h_i^{enc}}]\ \text{where}\ h_i^{enc} \in \mathbb{R}^{2h\times 1}, \overleftarrow{h_i^{enc}}, \overrightarrow{h_i^{enc}} \in \mathbb{R}^{h\times 1} \qquad i \in [1, m] \tag{1}
$$

$$
 c_i^{enc} = [\overleftarrow{c_i^{enc}}, \overrightarrow{c_i^{enc}}]\ \text{where}\ c_i^{enc} \in \mathbb{R}^{2h\times 1}, \overleftarrow{c_i^{enc}}, \overrightarrow{c_i^{enc}} \in \mathbb{R}^{h\times 1} \qquad i \in [1, m] \tag{2}
$$ 

4. **Decoder**: LSTM
   
$$
h_0^{dec} = W_h[\overleftarrow{h_1^{enc}}, \overrightarrow{h_m^{enc}}]\ \text{where}\ h_0^{dec} \in \mathbb{R}^{h\times 1},\ W_h \in \mathbb{R}^{h\times 2h} \tag{3}
$$

$$
c_0^{dec} = W_c[\overleftarrow{c_1^{enc}}, \overrightarrow{c_m^{enc}}]\ \text{where}\ c_0^{dec} \in \mathbb{R}^{h\times 1},\ W_c \in \mathbb{R}^{h\times 2h} \tag{4}
$$

- On the $t^{th}$ step, we look up the embedding of the target word $y_t$ and concatenate it with the *combined output vector* $o_{t-1}\in \mathbb{R}^{h\times 1}$ to produce the input to the decoder at step $t$, which will be $\overline{y_t}\in \mathbb{R}^{(h+e)\times 1}$.

$$
h_t^{dec}, c_t^{dec} = \text{Decoder}(\overline{y_t}, h_{t-1}^{dec}, c_{t-1}^{dec})\ \text{where}\ h_t^{dec}, c_t^{dec} \in \mathbb{R}^{h\times 1}\tag{5}
$$

5. **Attention**: multiplicative attention

$$
e_{t} = (h_t^{dec})^{\top} W_{attProj} h_i^{enc}\ \text{where}\ e_{t} \in \mathbb{R}^{m\times 1}, W_{attProj} \in \mathbb{R}^{h\times 2h}\tag{6}
$$

$$
\alpha_{t} = \text{softmax}(e_{t})\ \text{where}\ \alpha_{t} \in \mathbb{R}^{m\times 1}\tag{7}
$$

$$
a_t = \sum_{i=1}^{m}\alpha_{t,i}h_i^{enc}\ \text{where}\ a_t \in \mathbb{R}^{2h\times 1}\tag{8}
$$

6. Combined output

$$
u_t = [a_t, h_t^{dec}]\ \text{where}\ u_t \in \mathbb{R}^{(3h)\times 1}\tag{9}
$$

$$
v_t = W_{u} u_t\ \text{where}\ v_t \in \mathbb{R}^{h\times 1}, W_{u} \in \mathbb{R}^{h\times (3h)}\tag{10}
$$

$$
o_t = dropout(tanh(v_t))\ \text{where}\ o_t \in \mathbb{R}^{h\times 1}\tag{11}
$$

7. Output layer

$$
P_t = softmax(W_{vocab} o_t)\ \text{where}\ P_t \in \mathbb{R}^{V\times 1}, W_{vocab} \in \mathbb{R}^{V\times h}\tag{12}
$$

> [!success] Results
> <span style="color: red">BLEU Score: 20.11</span><br/>
> epoch: 5, patience: 1, others are default.



## Attention score

$$
\begin{align}
  &\text{Dot product}\quad \alpha = q^{\top}k\\
  &\text{Mutiplicative}\quad \alpha = q^{\top}W_k k\\
  &\text{Additive}\quad \alpha = W_v \tanh (W_k k + W_q q)
\end{align}\tag{13}
$$

How to choose?
1. Dot product: simple and fast, but not good for capturing complexity.
2. Multiplicative: good for long sentences, but kind of slower but efficient.
3. Additive: good for complexity, but very slow and expensive.



## Analysis of NMT


### 1D CNN

Adding a 1D Convolutional layer can help capture local patterns and relationships between adjacent characters in the input sequence. This is particularly useful in Mandarin Chinese, where characters can combine to form words or morphemes. The convolutional layer can learn to recognize these combinations and extract relevant features, which can then be passed to the bidirectional encoder for further processing.


### Improvement

1. More data exposure: let the model to learn more from the data
2. More complex model: more layers, more heads for the model to capture more complex patterns
3. Advanced mechanisms: advanced mechanisms such as transformer, self-attention, etc.
4. Other training techniques: more epochs, adaptive learning rate, more dropout, etc.


### BLEU

Suppose we have a reference sentence $r$ and a candidate sentence $c$, and we want to calculate the BLEU score for $c$ given $r$. Here we set n-gram size to be 4. And $\lambda_n$ is the weight of the $n$-gram precision.

$$
p_n = \frac{\sum \limits_{ngram \in c} \min \left ( \max \limits_{i=1,\dotsb, k} Count_{r_i}(ngram), Count_c(ngram) \right )}{\sum \limits_{ngram \in c} Count_c(ngram)}\tag{14}
$$

$$
BP = \begin{cases}  1, \text{if } c \geq r \\ \exp \left ( 1 - \frac{r}{c} \right ), \text{otherwise} \end{cases}\tag{15}
$$

$$
BLEU = BP \times \exp \left ( \sum \limits_{n=1}^{4} \lambda_n \log p_n \right )\tag{16}
$$

> the BLEU score is not a perfect metric for evaluating NMT translations. It is a good metric for comparing translations with different reference translations, but it does not take into account the fact that there may be multiple reference translations.

**Advantages**
- It is fast and cheap to compute, as it does not require human annotators.
- It is objective and reproducible, as it is based on a fixed set of reference translations.

**Disadvantages**
- Surface n-gram overlap only: poor at judging meaning/adequacy, paraphrases, and longrange reordering; weak at sentence-level.
- Precision-heavy with a simple brevity penalty: can be gamed and struggles with morphology/rarer words and semantic correctness.


> [!Note]
> 1. In this assignment, we build the NMT model from scratch. I think it's a good way to understand the whole image of the model structure with both the math and the code.
> 2. I also see some limitations of NMT from this assignment. Like the BLEU score is not a perfect metric while it's a good metric for comparing translations with different reference translations.


# Code

[Assign 3 Completion](https://github.com/viesuki/CS224N-Spring-2024-Assignments/tree/main/assign3)