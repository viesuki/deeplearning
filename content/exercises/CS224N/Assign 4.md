---
title: Assign 4 Transformer
tags: [NLP/NMT, RNN, attention]
created: 2024-10-30
modified: 2024-11-10
---



# Content

In Assignment 3, we will explore the reason why using multihead self-attention and the positional encoding and the little but powerful pre-trained model [minGPT](https://github.com/karpathy/minGPT)



## Attention exploration

For a simple single-head self-attention, we can use the following formula to calculate the attention score:

$$
\begin{align}
  c &= \sum_{i=1}^{n}v_i\alpha_i\\
  \alpha_i &= \frac{exp(k_i^{\top})q}{\sum_{j=1}^{n}exp(k_j^{\top})q} 
\end{align}
\tag{1}
$$


### Single head

Suppose that all the keys are orthogonal, i.e., $k_i^{\top}k_j=0$ for all $i,j$.

- If $q$ is very closely aligned to $k_i$, then the attention score will be very high. The output vector $c$ will be very similar to $v_i$.

Think that we have a query vector $q$ incorporating information from two vectors $v_a$ and $v_b$ with corresponding keys $k_a$ and $k_b$:

$$
q = s\cdot\frac{k_a+k_b}{\left \| k_a+k_b  \right \|}\tag{2}
$$

Where $s$ is a large positive scalar. By doing this combination, we can achive a value $c\approx v_a+v_b$.
- However, if the key vector, let's say $k_a$ has large veriance. And when $k_a$ is a vector with great magnitude, the query vector $q$ will be very close to $k_a$, and the attention score will be very high. The output vector $c$ will be very similar to $v_a$.


### Muilti-head

Suppose we have two heads, so we can use the following formula to calculate the attention score combined the two values as before:

$$
\begin{align}
  q_1 &= s \cdot \frac{\mu_a}{\left \| \mu_a \right \|}\\
  q_2 &= s \cdot \frac{\mu_b}{\left \| \mu_b \right \|}
\end{align} \tag{3}
$$

- In this case, even $\mu_a$ sometimes can reach a very high value, the only impact is that $c_a$ will be focusing on $v_a$. The output vector $c$ will still be very similar to $\mu_a+\mu_b$.



## Positional embedding

> [!note]
> - Permuting the input will not change the attention score without positional embedding. Basically, when computing the attention score, if the input is permuted, the attention score will be permuted as same way.
> - $H_{perm} = PH$

This will be problematic because the meaning of a sentence often depends on the order of the words. For example, "The cat sat on the mat” has a different meaning than ”On the mat sat the cat". If the model cannot distinguish between these two sequences due to its permutation invariance, it may fail to capture important syntactic and semantic information necessary for understanding and generating coherent text.

### Fixed method

$$
\begin{align}
  \Phi_{(t,2i)} &= \sin\left(t/10000^{2i/d} \right)\\
  \Phi_{(t,2i+1)} &= \cos\left(t/10000^{2i/d} \right)
\end{align}\tag{4}
$$

Where $t \in {0,1,\dots,T-1}$ and $ i \in {0,1,\dots,d/2-1}$


### Learnable method

In some cases, we can just set the positional encoding as a learnable parameter.


### Rotary Positional Encoding

$$
RoPE(x_t^{(1)}, x_t^{(2)}, t) = \begin{pmatrix}
  cos t\theta & -sin t\theta\\
  sin t\theta & cos t\theta
\end{pmatrix} \begin{pmatrix}
x_t^{(1)} \\
x_t^{(2)}
\end{pmatrix}
\tag{5}
$$

Where $t$ is the position index and $\theta$ is the angle: $$\theta = 10000^{-2(i-1)/d}$$

$$
\begin{align}
    \begin{pmatrix}
    \cos t\theta_1 & -\sin t\theta_1 & 0 & 0 & \cdots & 0 & 0\\
    \sin t\theta_1 & \cos t\theta_1 & 0 & 0 & \cdots & 0 & 0\\
    0 & 0 & \cos t\theta_2 & -\sin t\theta_2 & \cdots & 0 & 0\\
    0 & 0 & \sin t\theta_2 & \cos t\theta_2 & \cdots & 0 & 0\\
    \vdots & \vdots & \vdots & \vdots & \ddots & \vdots & \vdots\\
    0 & 0 & 0 & 0 & \cdots & \cos t\theta_{d/2} & -\sin t\theta_{d/2}\\
    0 & 0 & 0 & 0 & \cdots & \sin t\theta_{d/2} & \cos t\theta_{d/2}
    \end{pmatrix}
    \begin{pmatrix}
        x^{(1)}_t \\ x^{(2)}_t \\ x^{(3)}_t \\ x^{(4)}_t \\ \vdots \\ x^{(d-1)}_t \\ x^{(d)}_t 
    \end{pmatrix}
\end{align}\tag{6}
$$

$$
\begin{align}
    \begin{pmatrix}
        \cos t\theta_1 + i\sin t\theta_1 \\
        \cos t\theta_2 + i\sin t\theta_2 \\
        \vdots \\
        \cos t\theta_{d/2} + i\sin t\theta_{d/2}
    \end{pmatrix}
    \odot
    \begin{pmatrix}
        x^{(1)}_t + i x^{(2)}_t \\
        x^{(3)}_t + i x^{(4)}_t \\
        \vdots \\
        x^{(d-1)}_t + i x^{(d)}_t
    \end{pmatrix}
\end{align}\tag{7}
$$

The above Equa.(6) and Equa.(7) is equivalent. And this positional embedding is a relative methods.


## Language pre-trained model

In this part, we have several tasks to do. All the details are in the [[#Code]] part.
1. Implemnent finetuning and make predictions;
2. Define a [span corruption function](https://arxiv.org/pdf/1910.10683);
3. Pretrain, finetune and make predictions;
4. Implement RoPE function;
5. Pretrain, finetune and make predictions;


### Results

| Type | Accuracy |
| --- | --- |
| Baseline | 5.00 % |
| Vallina(no pretraining) | 1.40 % |
| Vallina(pretraining) | 23.20 % |
| RoPE | 38.60 % |


> [!Note]
> 1. Tasks today shall be done today.




# Code

[Assign 4 Completion](https://github.com/viesuki/CS224N-Spring-2024-Assignments/tree/main/assign4)