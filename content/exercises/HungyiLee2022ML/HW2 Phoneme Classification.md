---
title: HW2 Phoneme Classification
tags: [classification, Hung-yi Lee]
---

# Grading

> [!info] 
> 1. Simple: 0.45797
> 2. Medium: 0.69747
> 3. strong: 0.75028
> 4. Boss: 0.82324


# Solution

| Model  | HyperParameters <br> | Train <br> acc | Valid <br> acc | Public <br> Score | Private <br> Score |
| ---   | ---  | --- | --- | --- | --- |
| MLP | nframes = 21, hidden_layers=3, <br> batchnorm, dropout=0.25|...|... | 0.73379 | 0.73567 |
| LSTM | nframes = 21, hidden_layers=2,<br> fc_dropout=0.30 | 0.84755| 0.73834 | 0.74466 | 0.74330 |
| GRU | nframes = 21, hidden_layers=2, <br> fc_dropout=0.30 | 0.82196 | 0.74969 | 0.75266 | **0.75265** |
| GRU bi| nframes = 21, hidden_layers=2, <br> fc_dropout=0.30 | 0.82323 | 0.75220 | 0.75542 | 0.75627 |

# Reflection
> [!note] 
> 1. Bigger model makes better performance. I want to try more complecated model but my comuter doesn't alow. 😟
> 2. I used some tricks to seperate the traning process. (1) set larger n-frames and make deeper GRUbi model, (2)then seperate training data into 5 pecies and train 5 times using the last trained model params. But it didn't work well. The score was only about 0.757.
> 2. I did better than first homework because I made things organized.
> 3. Keep going.

# Code
[HW2 strong](https://github.com/viesuki/ML22-Lihongyi/blob/main/HW2_Phoneme/)
