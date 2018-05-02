# Implementation of 'A Two Step Disentanglement Method'
Model implementation for [Two-Step Disentanglement for Financial Data](https://arxiv.org/abs/1709.00199) by Naama Hadad, Lior Wolf and Moni Shahar from Tel Aviv University. This work address the problem of disentanglement of factors that generate a given data into those that are correlated with the labeling and those that are not. The model employs adversarial training in a straightforward manner.


# Requirements
Jupyter notebook, Python 3.5, numpy, pytorch 0.4, Matplotlib is also used to plot results

# STILL UNDER DEVELOPMENT
## Progress
1. Implement S encoder + S classifier and achieve 99%+ accuracy in MNIST data set [***Working***]
2. Implement Decoder and Adversarial net and whole net on MNIST
3. Test it on Sprites data set 
