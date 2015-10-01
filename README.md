# Training and Deep Learning classifier in Torch7 on GPU over CIFAR-10 dataset


## Requirements

## Data

## Classifier

## Code Description
	1. run_cifar10.lua  - main 
	2. data.lua - load data, and pre-process
	3. model.lua - setting up Neural Network architectures, couple of different models are available
	4. train.lua - train procedure
	5. test.lua - test procedure, reporting and plotting

## Experments and metrics
I want to examine:
- GPU .vs. CPU performance
- ReLU .vs. tanh performance, reproduce the graph ![alt text][img/relu_vs_tanh.jpeg] from [Krizhevsky et al.](http://www.cs.toronto.edu/~fritz/absps/imagenet.pdf "ImageNet Classification with Deep Convolutional
Neural Networks")
- measure times
- present confusion matrix
- plot error .vs. epoch
- on trained NN, plot error(weight)

## Results
