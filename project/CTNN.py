import torch.nn as nn
import SpykeTorch.snn as snn
import SpykeTorch.functional as sf
import numpy as np
import SpykeTorch.utils as utils
import torchvision.transforms as transforms
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torchvision.datasets import MNIST
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm, trange

use_cuda = True
MAX_EPOCH = 5

class InputTransform:
   def __init__(self, filter):
      self.to_tensor = transforms.ToTensor()
      self.filter = filter
      self.temporal_transform = utils.Intensity2Latency(15, to_spike=True)
   def __call__(self, image):
      image = self.to_tensor(image) * 255
      image.unsqueeze_(0)
      image = self.filter(image)
      image = sf.local_normalization(image, 8)
      return self.temporal_transform(image)


class CTNN(nn.Module):
    def __init__(self):
        super(CTNN, self).__init__()
        self.conv1 = snn.Convolution(6, 30, 5, 0.8, 0.02)  #(in_channels, out_channels, kernel_size, weight_mean=0.8, weight_std=0.02)
        self.conv2 = snn.Convolution(30, 100, 3, 0.8, 0.05)
        self.conv3 = snn.Convolution(250, 200, 5, 0.8, 0.05)

        self.stdp1 = snn.STDP(self.conv1, (0.004, -0.003))
        self.stdp2 = snn.STDP(self.conv2, (0.004, -0.003))
        self.training = True
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
            if max_layer == 1:
                winners = sf.get_k_winners(pot, 5, 3)
                self.save_data(input, pot, spk, winners)
                return spk, pot
            spk_in = sf.pad(sf.pooling(spk, 2, 2), (1,1,1,1))
            pot = self.conv2(spk_in)
            spk, pot = sf.fire(pot, 10, True)
            if max_layer == 2:
                winners = sf.get_k_winners(pot, 8, 2)
                self.save_data(spk_in, pot, spk, winners)
                return spk, pot
        else:
            pot = self.conv1(input)
            spk = sf.fire(pot, 15)
            pot = self.conv2(sf.pad(sf.pooling(spk, 2, 2), (1,1,1,1)))
            spk = sf.fire(pot, 10)
            pot = self.conv3(sf.pad(sf.pooling(spk, 3, 3), (2,2,2,2)))
            spk = sf.fire_(pot)
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
    for data, target in tqdm(data_loader):
        for i in range(len(data)):
            data_in = data[i].cuda() if use_cuda else data[i]
            spk, pot = net(data_in, 3)
            outputs.append(np.array([[spk, pot]]))
    return np.vstack(outputs)
    
if __name__ == "__main__":

    kernels = [ utils.DoGKernel(3,3/9,6/9), utils.DoGKernel(3,6/9,3/9),
            utils.DoGKernel(7,7/9,14/9), utils.DoGKernel(7,14/9,7/9),
            utils.DoGKernel(13,13/9,26/9), utils.DoGKernel(13,26/9,13/9)]

    filter = utils.Filter(kernels, padding = 6, thresholds = 50)

    transform = InputTransform(filter)

    data_root = 'data/'

    MNIST_train = utils.CacheDataset(MNIST(root=data_root, train=True, download=True, transform=transform))
    MNIST_test = utils.CacheDataset(MNIST(root=data_root, train=False, download=True, transform=transform))

    MNIST_loader = DataLoader(MNIST_train, batch_size=1000, shuffle=True)
    MNIST_test_loader = DataLoader(MNIST_test, batch_size=len(MNIST_test), shuffle=False)

    net = CTNN()
    net = train(net, MNIST_loader)
    torch.save(net.state_dict(), "./checkpoint.pt")
    outputs = inference(net, MNIST_test_loader)
    np.save('outputs.npy', outputs)






            
            