# Training Compute-Optimal Large Language Models

We use a maximum learning rate of 2 × 10 −4 for the smallest models and 1 . 25 × 10 −4 for the largest
models. In all cases, the learning rate drops by a factor of 10× during training, using a cosine schedule.
We make the assumption that the cosine cycle length should be approximately matched to the number
of training steps. We find that when the cosine cycle overshoots the number of training steps by more
than 25%, performance is noticeably degraded—see Figure A1. 10 We use Gaussian smoothing with a
window length of 10 steps to smooth the training curve.