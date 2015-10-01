--[[ Before running the data, go to 'data' directory and download CIFAR-10 dataset 
   from http://www.cs.toronto.edu/~kriz/cifar.html and converts it to Torch tables.
   Refer to https://github.com/soumith/cifar.torch
   
   It will create two files: cifar10-train.t7, cifar10-test.t7 Each of them is a table of the form:

      th> c10 = torch.load('cifar10-train.t7')
      th> print(c10)
      {
              data : ByteTensor - size: 50000x3x32x32
              label : ByteTensor - size: 50000
      }

--]]

require 'torch'
require 'paths'

cifar10={}

cifar10.path_remote = 'http://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz'
cifar10.path_dataset = 'datasets/cifar10.t7'
cifar10.path_trainset = paths.concat(cifar10.path_dataset, 'cifar10-train_32x32.t7')
cifar10.path_testset = paths.concat(cifar10.path_dataset, 'cifar10-test_32x32.t7')

function cifar10.download()
   if not paths.filep(cifar10.path_trainset) or not paths.filep(cifar10.path_testset) then
      local remote = cifar10.path_remote
      local tar = paths.basename(remote)
      os.execute('wget ' .. remote .. '; ' .. 'tar xvf ' .. tar .. '; rm ' .. tar)
   end
end

function cifar10.loadTrainSet(maxLoad, geometry)
   return cifar10.loadDataset(cifar10.path_trainset, maxLoad, geometry)
end

function mnist.loadTestSet(maxLoad, geometry)
   return cifar10.loadDataset(cifar10.path_testset, maxLoad, geometry)
end

function cifar10.loadDataset(fileName, maxLoad)
   cifar10.download()

   local f = torch.load(fileName, 'ascii')
   local data = f.data:type(torch.getdefaulttensortype())
   local labels = f.labels

   local nExample = f.data:size(1)
   if maxLoad and maxLoad > 0 and maxLoad < nExample then
      nExample = maxLoad
      print('<mnist> loading only ' .. nExample .. ' examples')
   end
   data = data[{{1,nExample},{},{},{}}]
   labels = labels[{{1,nExample}}]
   print('<mnist> done')

   local dataset = {}
   dataset.data = data
   dataset.labels = labels

   function dataset:normalize(mean_, std_)
      local mean = mean or data:view(data:size(1), -1):mean(1)
      local std = std_ or data:view(data:size(1), -1):std(1, true)
      for i=1,data:size(1) do
         data[i]:add(-mean[1][i])
         if std[1][i] > 0 then
            tensor:select(2, i):mul(1/std[1][i])
         end
      end
      return mean, std
   end

   function dataset:normalizeGlobal(mean_, std_)
      local std = std_ or data:std()
      local mean = mean_ or data:mean()
      data:add(-mean)
      data:mul(1/std)
      return mean, std
   end

   function dataset:size()
      return nExample
   end

   local labelvector = torch.zeros(10)

   setmetatable(dataset, {__index = function(self, index)
              local input = self.data[index]
              local class = self.labels[index]
              local label = labelvector:zero()
              label[class] = 1
              local example = {input, label}
                                       return example
   end})

   return dataset
end
--------------------------------




//another version:


os.execute('wget -c http://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz')
os.execute('tar -xvf cifar-10-binary.tar.gz')
local function convertCifar10BinToTorchTensor(inputFnames, outputFname)
   local nSamples = 0
   for i=1,#inputFnames do
      local inputFname = inputFnames[i]
      local m=torch.DiskFile(inputFname, 'r'):binary()
      m:seekEnd()
      local length = m:position() - 1
      local nSamplesF = length / 3073 -- 1 label byte, 3072 pixel bytes
      assert(nSamplesF == math.floor(nSamplesF), 'expecting numSamples to be an exact integer')
      nSamples = nSamples + nSamplesF
      m:close()
   end

   local label = torch.ByteTensor(nSamples)
   local data = torch.ByteTensor(nSamples, 3, 32, 32)

   local index = 1
   for i=1,#inputFnames do
      local inputFname = inputFnames[i]
      local m=torch.DiskFile(inputFname, 'r'):binary()
      m:seekEnd()
      local length = m:position() - 1
      local nSamplesF = length / 3073 -- 1 label byte, 3072 pixel bytes
      m:seek(1)
      for j=1,nSamplesF do
         label[index] = m:readByte()
         local store = m:readByte(3072)
         data[index]:copy(torch.ByteTensor(store))
         index = index + 1
      end
      m:close()
   end

   local out = {}
   out.data = data
   out.label = label
   print(out)
   torch.save(outputFname, out)
end

convertCifar10BinToTorchTensor({'cifar-10-batches-bin/data_batch_1.bin',
                                'cifar-10-batches-bin/data_batch_2.bin',
                                'cifar-10-batches-bin/data_batch_3.bin',
                                'cifar-10-batches-bin/data_batch_4.bin',
                                'cifar-10-batches-bin/data_batch_5.bin'},
   'cifar10-train.t7')

convertCifar10BinToTorchTensor({'cifar-10-batches-bin/test_batch.bin'},
   'cifar10-test.t7')
