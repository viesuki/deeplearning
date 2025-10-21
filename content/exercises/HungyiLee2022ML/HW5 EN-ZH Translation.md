---
title: HW5 EN-ZH Translation
tags: [NLP/NMT, transformer, Hung-yi Lee]
created: 2024-10-19
modified: 2024-10-20
---

# Grading

> [!info] 
> 1. Simple: 14.58
> 2. Medium: 18.04
> 3. strong: 25.20
> 4. Boss: 29.13


# Solution

| Model  | HyperParameters <br> | Train <br> loss | Valid <br> loss | Valid <br> BLEU | Test <br> BLEU | Timer |
| ---   | ---  | --- | --- | --- | --- | --- |
| RNN | 1 | 4.0826 | 3.8945 | 16.58 | 17.19 | ~41 m |
| Transformer | 1-3 | 3.0306 | 3.2159 | 26.36 | 26.70 | ~6 h |



# Techniques

1. learning rate
   ```python
   def get_rate(d_model, step_num, warmup_step):
       lr = (d_model ** -0.5) * min(step_num ** -0.5, step_num * warmup_step ** -1.5)
       return lr
   ``` 
2. epoch = 30
3. Trasnformer
   ```python
      arch_args = Namespace(
          encoder_embed_dim=512,
          encoder_ffn_embed_dim=2048,
          encoder_layers=4,
          decoder_embed_dim=512,
          decoder_ffn_embed_dim=2048,
          decoder_layers=4,
          share_decoder_input_output_embed=True,
          dropout=0.3,
      )
   ```



# Reflection
> [!note] 
> 1. This homework is very similar to CS224N Assignment 3 which is a translation task of Chinese to English.
> 2. Using transformer will result in better performance but it takes a long time to train for a heavier model.
> 3. The module `fairseq` is useful but difficult to implement. I tried several times to implement it but failed. But finally I just built a new environment `python=3.8` and installed the old version of `fairseq`, which yielded the scores above.
> 4. In the process of training, it took a long time because it stopped over 10 times due to the hardware limitation of my laptop. So I stopped to the strong line.
> 5. I tried to use Kaggle too, which didn't work either due to the implementation of `fairseq`.



# Code
[HW4 strong](https://github.com/viesuki/ML22-Lihongyi/tree/main/HW5-Translation)

# Reference
[Aaricis HW5](https://aaricis.github.io/posts/Homework-5-Transformer/)

