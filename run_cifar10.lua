--[[ Deep Learning classifier in Torch7 on GPU over CIFAR-10 dataset
run in Torch:
> th run_cifar10.lua

Good tutorials for CIFAR-10:
 https://github.com/szagoruyko/cifar.torch
 
--]]

require 'cutorch'
lapp = require 'pl.lapp'
require 'cunn'
require 'image'

----------------------------------------------------------------------
-- parse command-line options

-- Lines begining with '-' are flags; there may be a short and a long name;
-- lines begining wih '<var>' are arguments.  Anything in parens after
-- the flag/argument is either a default, a type name or a range constraint.
--
-- See the guide for 'lapp' on: https://github.com/stevedonovan/Penlight/blob/master/doc/manual/08-additional.md
-- Combine configure file with args: http://curiouser.cheshireeng.com/2014/09/03/trick-create-a-post-from-lua-part-3/

local args = require ('pl.lapp')[[
--   -s,--save          (default "logs")      subdirectory to save logs
--   -n,--network       (default "")          reload pretrained network
--   -m,--model         (default "convnet")   type of model tor train: convnet | mlp | linear
   -f,--full                                use the full dataset
--   -p,--plot        StochasticGradient                        plot while training
--   -o,--optimization  (default "SGD")       optimization: SGD | LBFGS 
--   -r,--learningRate  (default 0.05)        learning rate, for SGD only
   -b,--batchSize     (default 10)          batch size
--   -m,--momentum      (default 0)           momentum, for SGD only
--   -i,--maxIter       (default 3)           maximum nb of iterations per batch, for LBFGS
--   --coefL1           (default 0)           L1 penalty on the weights
--   --coefL2           (default 0)           L2 penalty on the weights
--   -t,--threads       (default 4)           number of threads
]]

print 'parameters for running:'
for k,v in pairs(args) do
      print('\t', k,v)
end
  
-- fix seed
torch.manualSeed(1)

function main()
   print('Start CIFAR-10 on GPU')
   print('CIFAR-10 dataset from http://www.cs.toronto.edu/~kriz/cifar.html')
   print('Running on device: ' .. cutorch.getDeviceProperties(cutorch.getDevice()).name)

   print("Info on GPUs running on the server:")
   os.execute("nvidia-smi")
   io.write('Running from '); io.flush(); os.execute('pwd');

--   require 'data.lua'
   require 'model.lua'
   require 'train.lua'
   require 'test.lua'

   -- data --
   cifar10.download()
   trainset = torch.load(cifar10.path_trainset_file)
   testset = torch.load(cifar10.path_testset_file)
   
   trainset.data = trainset.data:cuda() -- transfer the data across
   print('Training set:')
   print(trainset)
   
   testset.data = testset.data:cuda() -- transfer the data across
   print('Test set:')
   print(testset)
   
   -- Normalize each channel, and store mean/std
   -- per channel. These values are important, as they are part of
   -- the trainable parameters. At test time, test data will be normalized
   -- using these values.
   print '==> preprocessing data: normalize each feature (channel) globally'
   mean = {}
   std = {}
   for i=1,3 do
      -- normalize each channel globally:
      mean[i] = trainset.data[{ {},i,{},{} }]:mean()
      std[i] = trainset.data[{ {},i,{},{} }]:std()
      trainset.data[{ {},i,{},{} }]:add(-mean[i])
      trainset.data[{ {},i,{},{} }]:div(std[i])
   end
   
   
   classes = {'airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'}
   print('classes: ')
   print(classes)
   
   -- model --         
   net = model.constructModel(); 
   net:cuda()
   print('ConvNet configured: ')
   print(net)
   
   criterion = nn.ClassNLLCriterion()
   criterion = criterion:cuda() -- transfer the criterion
   

--   image.display(image.lena())
--   image.display(trainset.data[1])
   
--   Train the neural network First define an nn.StochasticGradient object then we'll give our dataset to this object's :train function.
   trainer = nn.StochasticGradient(net, criterion)
   trainer.learningRate = 0.001
   trainer.maxIteration = 5 
   
   print '==> training!'
--   print(trainset.data:size())
--   torch.randperm(trainset.data:size(), 'torch.LongTensor')
   trainer:train(trainset.data)
end

main()