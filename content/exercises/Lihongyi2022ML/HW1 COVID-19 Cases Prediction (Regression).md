---
title: HW1 COVID-19 Cases Prediction
tags: regression
---

# Grading

> [!info] 
> 1. [[#Simple]]: 2.284
> 2. [[#Medium]]: 1.494
> 3. [[#strong]]: 1.057
> 4. [[#Boss]]: 0.862


# Medium

> Hint: feature selection

Delete the first 37 one-hot features (state information)
|       | Score|
| ---   | ---  |
| epoch | 787 |
| train loss | 1.175 |
| val loss | 1.001 |
| public score | 1.269 |
| private score | **1.415** |


# Strong

> Hint: Different model architechtures and optimizers

Build deeper model
16-8-1 --> 32-16-8-1
|       | Score|
| ---   | ---  |
| epoch | 1265 |
| train loss | 1.123 |
| val loss | 0.955 |
| public score | 0.972 |
| private score | **1.027** |


# Boss

> Hint: L2 regularization and try more parameters

This is too hard. I give up.

My best score:
private score: 0.990; public score: 0.942

Actually I tried too many times and I just forgot how to get this score... Maybe based on strong model, I used LeakyReLU, added L2 regularization, and implemented k-fold cross validation, then selected the best model.

> [!note] For future DL training
> 1. Make structural records, every time just change one thing.
> 2. Change should be simple, don't change too much.
> 3. Organize code and try to get better, but don't fall into perfectionism.

# Code
[HW1 strong](https://github.com/viesuki/ML22-Lihongyi/blob/main/HW1_Covid19/)