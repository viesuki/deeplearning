---
title: Assign 1 Word2Vec
tags: [NLP]
created: 2024-10-18
modified: 2024-10-18
---

# Content
In Assignment 1, we will explore the word vectors. Here, we will explore two types of word vectors: those derived from *co-occurrence matrices*, and those derived via *GloVe*. 


**Co-occurrence Matrix**

**Example: Co-Occurrence with Fixed Window of n=1**:

Document 1: "all that glitters is not gold"

Document 2: "all is well that ends well"


|     *    | `<START>` | all | that | glitters | is   | not  | gold  | well | ends | `<END>` |
|----------|-------|-----|------|----------|------|------|-------|------|------|-----|
| `<START>`    | 0     | 2   | 0    | 0        | 0    | 0    | 0     | 0    | 0    | 0   |
| all      | 2     | 0   | 1    | 0        | 1    | 0    | 0     | 0    | 0    | 0   |
| that     | 0     | 1   | 0    | 1        | 0    | 0    | 0     | 1    | 1    | 0   |
| glitters | 0     | 0   | 1    | 0        | 1    | 0    | 0     | 0    | 0    | 0   |
| is       | 0     | 1   | 0    | 1        | 0    | 1    | 0     | 1    | 0    | 0   |
| not      | 0     | 0   | 0    | 0        | 1    | 0    | 1     | 0    | 0    | 0   |
| gold     | 0     | 0   | 0    | 0        | 0    | 1    | 0     | 0    | 0    | 1   |
| well     | 0     | 0   | 1    | 0        | 1    | 0    | 0     | 0    | 1    | 1   |
| ends     | 0     | 0   | 1    | 0        | 0    | 0    | 0     | 1    | 0    | 0   |
| `<END>`      | 0     | 0   | 0    | 0        | 0    | 0    | 1     | 1    | 0    | 0   |

Here I won't show the code which will be attached in the [[#code]] part.

**Results**
1. After I complete the code of Co-occurrence matrix part and pass all the simple check. My result figure is kind of flipping upside down compared to the reference answer.
2. While in the GloVe part, the result figure is flipping leftside right.

> [!Note]
> 1. For the word which has multiple meanings, it's hard to detect all the meaning space to fit the analogy test well.
> 2. And also for the analysis of bias should be payed attention to, like gender, race, religion etc. When training the model, we should use the corpus carefully.


# Code

[Assign 1 Completion](https://github.com/viesuki/CS224N-Spring-2024-Assignments/tree/main/assign1)