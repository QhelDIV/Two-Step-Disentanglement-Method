# Implementation of 'A Two Step Disentanglement Method'
Model implementation for [Two-Step Disentanglement for Financial Data](https://arxiv.org/abs/1709.00199) by Naama Hadad, Lior Wolf and Moni Shahar from Tel Aviv University. This work address the problem of disentanglement of factors that generate a given data into those that are correlated with the labeling and those that are not. The model employs adversarial training in a straightforward manner.


# Requirements
Jupyter notebook, Python 3.5, numpy, pytorch 0.4, Matplotlib is also used to plot results

# STILL UNDER DEVELOPMENT
Current results:
![alt text][mnistRecon]
![alt text][mnistInter]
![alt text][mnistSwitch]
![alt text][spriteRecon]
![alt text][spriteInter]
![alt text][spriteSwitch]
## Progress
1. Implement S encoder + S classifier and achieve 99%+ accuracy in MNIST data set [***Done.But only get 97% accuracy***]
2. Implement Decoder and Adversarial net and whole net on MNIST[***Done. But unstable***]
3. Test it on Sprites data set [***Done*** (see the results explanation in betterResults.pptx)] 
4. Extend the method to 3D point cloud [***TO DO***]


[mnistRecon]: imgs/mnistRecon.png
[mnistInter]: imgs/mnistInter.png
[mnistSwitch]: imgs/mnistSwitch.png
[spriteRecon]: imgs/spriteRecon.png
[spriteInter]: imgs/spriteInter.png
[spriteSwitch]: imgs/spriteSwitch.png
