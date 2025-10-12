---
title: HW4 Speaker Recognization
tags: [classification, transformer, Hung-yi Lee]
created: 2024-10-10
modified: 2024-10-12
---

# Grading

> [!info] 
> 1. Simple: 0.60824
> 2. Medium: 0.70375
> 3. strong: 0.77750
> 4. Boss: 0.86500


# Solution

| Model  | HyperParameters <br> | Train <br> acc | Valid <br> acc | Public <br> Score | Private <br> Score | Timer |
| ---   | ---  | --- | --- | --- | --- | --- |
| all | 1-5 | 1.00 | 0.9306 | **0.84325** | **0.84200** | ~35 m |
| all | 1-6 head 8 | 1.00  | 0.9460  | 0.83675 | 0.83550 | ... |
| all | 1-5 + 7, 1w steps | 1.00  | 0.9140  | 0.86250 | 0.86050 | 47 m |
| all | 1-7, 1w steps | 1.00  | 0.9325  | 0.83850 | 0.83325 | 138 m |
| all | 1-5 + 7 | 1.00  | 0.9085  | **0.86400** | **0.86100** | **36 m** |
| all | 1-5 + 7, 1w steps | 1.00  | 0.9140  | 0.86250 | 0.86050 | 47 m |
| all | 1-5 + 8 | 1.00  | 0.9384  | **0.87150** | **0.87325** | **35 m**|
| **all** | **1-5 + 8, 3 layers** | 1.00  | **0.9433**  | **0.88125** | **0.87900** | 47 m |



# Reflection
> [!note] 
> 1. As presented in the intro part of this homework by AT, I just adopted the idea of using **conformer**, **self attention pooling** and **ASMsoftmax** to solve the problem. Surprisingly, the result is very good. Then I tried to use more layers, more heads and other teches, but the result is not ideal, even worse.
> 2. Then I thought maybe because self attention pooling dropped too much information, so I decided to calculate the self attention pooling and max pooling after conformer layers, and concatenate them together and feed into the ASMsoftmax. Finally I got 0.86 which is so close to boss line.
> 3. When I increased the batch size from 32 to 128, the test accuracy increased, but the score decreased with much more time of training.
> 4. Finally, I just realized that why I combined max pooling and self attention pooling together. So I deleted the max pooling and got the score 0.87 over boss line. Then I tried to plus one more conformer layer and got the score 0.88!!!
> - I think the model is kind of good already if just using <span style="color: red">strategy 1 to 5, and only plus batch norm layer after the self attention pooling</span>, **we will get over the boss line easily within 35 mins**. And with one more conformer layer, we can get 0.88.
> - __Next time: focus on boss line strategy; use one more change one time; then record.__

> [!warning] Not Help
> 1. batch size 
> 2. train steps
> 3. more complexity (as 4 conformer layers will get around 0.865)



# Techniques

1. d_model = 80 --> 256
2. drop_out = 0.1 --> 0.3
3. Conformer Layer
   ```python
   from torchaudio.models import Conformer
   ......
      self.conformer = Conformer(
         input_dim=d_model,
         num_heads=4,
         ffn_dim=512,
         num_layers=2, 
         depthwise_conv_kernel_size=31, 
         dropout=dropout
      )
   ```
4. self attention pooling
   ```python
   class SelfAttentionPooling(nn.Module):
       def __init__(self, input_dim):
           super().__init__()
           self.attention = nn.Linear(input_dim, 1, bias=False)

       def forward(self, batch_rep, mask=None):
           # scores: (N, T)
           scores = self.attention(batch_rep).squeeze(-1)
           if mask is not None:
               scores = scores.masked_fill(~mask.bool(), float("-inf"))
           att_w = F.softmax(scores, dim=1)          # (N, T)
           utter_rep = torch.bmm(att_w.unsqueeze(1), batch_rep).squeeze(1)  # (N, H)
           return utter_rep
   ```
5. ASMsoftmax
   ```python
   class AMSoftmaxCE(nn.Module):
       def __init__(self, in_features, n_classes, s=30.0, m=0.35):
           super().__init__()
           self.W = nn.Linear(in_features, n_classes, bias=False)
           self.s, self.m = float(s), float(m)
       def forward(self, x, y=None):
           x = F.normalize(x, dim=-1)
           W = F.normalize(self.W.weight, dim=-1)
           logits = F.linear(x, W)              # cos
           logits = logits * self.s
           if y is None:
               return logits
           logits[torch.arange(x.size(0)), y] -= self.s * self.m
           return F.cross_entropy(logits, y)
   ```
6. batch_size = 32 --> 128
7. **concat out.max + batchnorm(out) + dropout**
   ```python
   class Classifier(nn.Module):
      ......
      def forward(self, x):       
         out = self.prenet(mels)        
         # (batch_size, length, d_model)
         out, _ = self.conformer(out, torch.full((out.shape[0],),out.shape[1]).cuda())
         # self attention pooling (batch_size, d_model)
         stats = self.pooling(out)
         out = out.max(dim=1)[0]
         # batchnorm
         out = self.bn(torch.cat([stats, out], dim=1))
         out = self.drop(out)
         out = self.classier(out)
   ```
8. batchnorm(out) only

# Code
[HW4 boss](https://github.com/viesuki/ML22-Lihongyi/blob/main/HW4_Speaker/)

# Reference
[Niku's Blog HW4](https://www.nikunokoya.com/posts/lhy2021_hw4/#conformer)

