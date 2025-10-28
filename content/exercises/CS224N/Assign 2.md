---
title: Assign 2 Dependency
tags: [NLP, linguistic]
created: 2024-10-28
modified: 2024-10-28
---



# Content

In Assignment 2, we will review the mathematics behind Word2Vec and build a neural dependency parser using PyTorch.


## Math for Word2Vec

The most important equation in Word2Vec is the following probability:

$$
P(w_i | w_j) = \frac{exp(w_i^{\top} w_j)}{ \sum_{k=1}^V exp(w_k^{\top} w_j)}\quad \text{OR}\quad P(O=o | C=c) = \frac{exp(u_o^{\top}v_c)}{\sum_{w\in Vocab} \exp(u_w^{\top}v_c)}\tag{1}
$$

And the objective function is (the colomns of $U$ are all the outsider vectors):

$$
\mathbf{J}_{naive-softmax}(v_c, o, U) = -\log P(O=o | C=c)\tag{2}
$$

which is the same as the cross-entropy loss. So to update the parameters like $v_c$ and $u_o$, we can use the following gradient equations:

$$
\begin{align}
    v_c &: v_c - \alpha \frac{\partial \mathbf{J}_{naive-softmax}(v_c, o, U)}{\partial v_c} \\
    &= v_c - (- u_o + \sum \limits_{x\in Vocab} \hat{y_x}u_x)\\
    &= v_c - U(\hat{y} - y)
\end{align}\tag{3}
$$

And for the $U$, we have:

$$
\bigtriangledown_U \mathbf{J}_{naive-softmax}(v_c, o, U) = v_c(\hat{y} - y)^{\top}\tag{4}
$$

For inference details, go to check [[1.word2vec|N1 Representing Words]] or [[#Code]]


## Adam optimizer

- Advantages of **Momentum**
  1. helps smooth out the updates by averaging the gradients over time.
  2. allows the model to maintain a consistent direction in parameter space, leading to faster convergence.

- Advantages of **adaptive learning rates**
  1. helps prevent the model from getting stuck in local minima or saddle points.
  2. prevents the parameters with large gradients from dominating the optimization process, making the optimization more stable.


## Dropout layer

Dropout should be applied during training because it helps prevent overfitting by randomly dropping units from the hidden layer. This helps to reduce the complexity of the model and prevent it from overfitting to the training data. However, dropout should not be applied during evaluation because it can lead to overfitting if the model is evaluated on data that it has not seen during training.


## Neural dependency parsing

One hidden layer classification, designed as follows:

$$
\begin{align}
    x &= [E_{w_1}, E_{w_2}, ..., E_{w_m}] \in \mathbb{R}^{d \times m}\\
    h &= ReLU(xW + b_1)\\
    l &= hU + b_2\\
    \hat{y} &= softmax(l)
\end{align}\tag{5}
$$

Results: Accracy of UAS
- Dev: 88.52%
- Test: 89.42%


## Parsing error

> [!example]
> - Moscow sent troops into Afghanistan.
> - Leaving the store unattended, I went outside to watch the parade.
> - I am extremely short.
> - Would you like brown rice or garlic naan?

| Error name                                | Explanation                                                                                                          | Example |
| ----------------------------------------- | -------------------------------------------------------------------------------------------------------------------- | --- |
| **Prepositional Phrase Attachment Error** | A prepositional phrase (PP) is attached to the wrong head word.                                                      | In “… **into Afghanistan** …”, the PP is wrongly attached to **troops**; the correct head it should attach to is the verb **sent** (“troops **sent [into Afghanistan]**”). |
| **Verb Phrase Attachment Error**          | A verb phrase (VP) is attached to the wrong head word in the clause.                                                 | In “**Leaving the store unattended**, I went outside to watch the parade,” the VP should attach to **went** (main clause), not to **watch**.                               |
| **Modifier Attachment Error**             | A modifier (e.g., adverb/adjective) is attached to the wrong head word.                                              | In “I am **extremely** **short**,” the adverb **extremely** should modify **short**, not **am**.                                                                           |
| **Coordination Attachment Error**         | In a coordination, the second conjunct is attached to the wrong head; it should attach to the first conjunct’s head. | In “Would you like **brown rice** or **garlic naan**?”, **garlic naan** should attach to the first conjunct’s head **rice** (i.e., *[brown **rice**] or [garlic naan]*).   |



> [!Note]
> 1. Understanding the mathmatics behind Word2Vec and neural dependency parsing is crucial for understanding the underlying mechanisms of these models.
> 2. The nueral dependency parsing model is a simple neural network but with excellent performance.
> 3. When buiding nueral network, it's a good way to understanding the whole image of the model structure.


# Code

[Assign 2 Completion](https://github.com/viesuki/CS224N-Spring-2024-Assignments/tree/main/assign2)