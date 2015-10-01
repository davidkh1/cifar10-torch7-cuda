--[[ Deep Learning classifier in Torch7 on GPU over CIFAR-10 dataset
run in Torch:
> th run_cifar10.lua

--]]

require 'cutorch'

print('Start CIFAR-10 on GPU')

print('Running on device: ' .. cutorch.getDeviceProperties(cutorch.getDevice()).name)

print "GPUs running on the server:"
os.execute("nvidia-smi")

