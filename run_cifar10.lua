--[[ Deep Learning classifier in Torch7 on GPU over CIFAR-10 dataset
run in Torch:
> th run_cifar10.lua

Good tutorials for CIFAR-10:
 https://github.com/szagoruyko/cifar.torch
 
--]]

require 'cutorch'

print('Start CIFAR-10 on GPU')
print('CIFAR-10 dataset from http://www.cs.toronto.edu/~kriz/cifar.html')
print('Running on device: ' .. cutorch.getDeviceProperties(cutorch.getDevice()).name)

print("Info on GPUs running on the server:")
os.execute("nvidia-smi")
print('Running from ')
os.execute('pwd')

require 'data.lua'
--dofile 'model.lua'
--dofile 'train.lua'
--dofile 'test.lua'

cifar10.download()