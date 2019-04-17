import torch.nn as nn
import SpykeTorch.snn as snn
import SpykeTorch.functional as sf
import numpy as np
import SpykeTorch.utils as utils
import torchvision.transforms as transforms
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torchvision.datasets import MNIST
from torchvision.datasets import CIFAR10 as CIFAR
from sklearn import svm
from sklearn.metrics import accuracy_score
from tqdm import tqdm, trange

use_cuda = True
MAX_EPOCH = 1

class InputTransform:
   def __init__(self, filter):
      self.to_tensor = transforms.ToTensor()
      self.filter = filter
      self.temporal_transform = utils.Intensity2Latency(30, to_spike=True)
   def __call__(self, image):
      image = self.to_tensor(image) * 255
      image.unsqueeze_(0)
      C = image.size(1)
      x = []
      for c in range(C):
        img = image[:,c] # 1x32x32
        img.unsqueeze_(1) # 1x1x32x32
        img = self.filter(img) #1x2x32x32
        x.append(img)
      image = torch.cat(x, 1)
      #image = self.filter(image)
      image = sf.local_normalization(image, 8)
      return self.temporal_transform(image)


class CTNN(nn.Module):
    def __init__(self):
        super(CTNN, self).__init__()
        self.conv1 = snn.Convolution(2, 30, 5, 0.8, 0.02)  #(in_channels, out_channels, kernel_size, weight_mean=0.8, weight_std=0.02)
        self.conv2 = snn.Convolution(30, 100, 5, 0.8, 0.02)
        #self.conv3 = snn.Convolution(250, 200, 5, 0.8, 0.05)

        self.stdp1 = snn.STDP(self.conv1, (0.004, -0.003))
        self.stdp2 = snn.STDP(self.conv2, (0.004, -0.003))
        self.ctx = {"input_spikes": None, "potentials": None, "output_spikes":None, "winners":None}

    def save_data(self, input_spk, pot, spk, winners):
        self.ctx["input_spikes"] = input_spk
        self.ctx["potentials"] = pot
        self.ctx["output_spikes"] = spk
        self.ctx["winners"] = winners
        
    def forward(self, input, max_layer):
        input = sf.pad(input, (2,2,2,2))
        if self.training: #forward pass for train
            pot = self.conv1(input)
            spk, pot = sf.fire(pot, 15, True)
            pot = sf.pointwise_inhibition(pot) # inter-channel inhibition
            if max_layer == 1:
                winners = sf.get_k_winners(pot, 8, 3)
                self.save_data(input, pot, spk, winners)
                return spk, pot
            spk_in = sf.pad(sf.pooling(spk, 2, 2), (1,1,1,1))
            pot = self.conv2(spk_in)
            spk, pot = sf.fire(pot, 10, True)
            pot = sf.pointwise_inhibition(pot) # inter-channel inhibition
            if max_layer == 2:
                winners = sf.get_k_winners(pot, 5, 8)
                self.save_data(spk_in, pot, spk, winners)
                return spk, pot
        else:
            pot = self.conv1(input)
            spk, pot = sf.fire(pot, 15, True)
            pot = self.conv2(sf.pad(sf.pooling(spk, 2, 2), (1,1,1,1)))
           # spk = sf.fire(pot, 10)
           # pot = self.conv3(sf.pad(sf.pooling(spk, 3, 3), (2,2,2,2)))
           # spk = sf.fire_(pot)
            
            return spk, pot
    
    def stdp(self, layer_idx):
        if layer_idx == 1:
                self.stdp1(self.ctx["input_spikes"], self.ctx["potentials"], self.ctx["output_spikes"], self.ctx["winners"])
        if layer_idx == 2:
            self.stdp2(self.ctx["input_spikes"], self.ctx["potentials"], self.ctx["output_spikes"], self.ctx["winners"])


def train_unsupervised(network, data, layer_idx):
    network.train()
    for i in range(len(data)):
        data_in = data[i].cuda() if use_cuda else data[i]
        network(data_in, layer_idx)
        network.stdp(layer_idx)


def train(net, data_loader):
    net = net.cuda() if use_cuda else net
    for epoch in trange(MAX_EPOCH):
        for data, _ in tqdm(data_loader):
            train_unsupervised(net, data, 1)
    for epoch in trange(MAX_EPOCH):
        for data, _ in tqdm(data_loader):
            train_unsupervised(net, data, 2)
    return net


def inference(net, data_loader):
    net = net.cuda() if use_cuda else net
    net.eval()
    outputs = []
    labels = []
    for data, target in tqdm(data_loader):
        for i in range(len(data)):
            torch.cuda.empty_cache()
            data_in = data[i].cuda() if use_cuda else data[i]
            _, pot = net(data_in, 3)
            #spk = spk.cpu()
            pot=pot.cpu().numpy()
            #pot=pot[None, :]
            pot=GMP(pot) 
            outputs.append(pot)
        labels.append(target)
    outputs = np.vstack(outputs)
    #outputs = GMP(outputs)
    return outputs, np.hstack(labels)
    

def GMP(data):
    _, C, _, _ = data.shape
    final_pot = data[-1]
    max_pool = np.max(final_pot, axis=(1,2))
    return max_pool.reshape((1,C)) 

def sample_zero_mean(x):
    """
    Make each sample have a mean of zero by subtracting mean along the feature axis.
    :param x: float32(shape=(samples, features))
    :return: array same shape as x
    """
    return x - x.mean(axis=1, keepdims=True)
    pass


def gcn(x, scale=1., bias=0.01):
    """
    GCN each sample (assume sample mean=0)
    :param x: float32(shape=(samples, features))
    :param scale: factor to scale output
    :param bias: bias for sqrt
    :return: scale * x / sqrt(bias + sample variance)
    """
    var = np.var(x, axis=1, keepdims=True)
    return scale * x / np.sqrt(bias + var)
    pass



def preprocess(x, xtest):
    x = sample_zero_mean(x)
    x = gcn(x)
    xtest = sample_zero_mean(xtest)
    xtest = gcn(xtest)
    return x, xtest

if __name__ == "__main__":

    kernels = [ utils.DoGKernel(3,1,2), utils.DoGKernel(3,2,1)]

    filter = utils.Filter(kernels, padding = 6, thresholds = 50)

    transform = InputTransform(filter)

    data_root = 'data/'
    cifar_data_root = 'cifar/'

    # MNIST_train = utils.CacheDataset(MNIST(root=data_root, train=True, download=True, transform=transform)) # 60000 x 28 x 28
    # MNIST_test = utils.CacheDataset(MNIST(root=data_root, train=False, download=True, transform=transform)) # 10000 x 28 

    # MNIST_loader = DataLoader(MNIST_train, batch_size=1000, shuffle=True)
    # MNIST_test_loader = DataLoader(MNIST_test, batch_size=1000, shuffle=False)

    CIFAR_train = utils.CacheDataset(CIFAR(root=cifar_data_root, train=True, download=True, transform=transform)) # 60000 x 30 x 30
    CIFAR_test = utils.CacheDataset(CIFAR(root=cifar_data_root, train=False, download=True, transform=transform)) # 10000 x 30

    CIFAR_loader = DataLoader(CIFAR_train, batch_size=1000, shuffle=True)
    CIFAR_test_loader = DataLoader(CIFAR_test, batch_size=1000, shuffle=False)
    

    net = CTNN()
    clf = svm.SVC(verbose=True) # (60000 x 100)

    net = train(net, CIFAR_train)
    torch.save(net.state_dict(), "./checkpoint.pt")
    net.state_dict(torch.load("./cifar_checkpoint.pt"))
    train_outputs, train_y = inference(net, CIFAR_loader)
    test_outputs, test_y  = inference(net, CIFAR_test_loader)
    train_outputs, test_outputs = preprocess(train_outputs, test_outputs)
    
    clf.fit(train_outputs, train_y)
    acc=clf.score(train_outputs, train_y)
    print ("Training Accuracy is %.3f" % acc)
   
    acc = clf.score(test_outputs, test_y)
    print ("Test Accuracy is %.3f" % acc)
    






            
            
