## Deep Learning classifier in Torch7 on GPU over CIFAR-10 dataset


### Requirements

### Data

### Classifier

### Code Description
	1. run_cifar10.lua  - main 
	2. data.lua - load data, and pre-process
	3. model.lua - setting up Neural Network architectures, couple of different models
	4. train.lua - train procedure
	5. test.lua - test procedure, reporting and plotting

### Experments and metrics
I want to examine:
- GPU .vs. CPU performance
- ReLU .vs. tanh performance, reproduce <a href="img/relu_vs_tanh.jpeg" target="_blank">the graph</a> from <a href="http://www.cs.toronto.edu/~fritz/absps/imagenet.pdf" target="_blank">Krizhevsky et al.</a>
- measure times
- present confusion matrix
- plot error .vs. epoch
- alter weights of trained NN, and without additional training, plot error=f(weight<sub>i,j</sub>)

### Results
