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
from tqdm import tqdm, trange

use_cuda = True
MAX_EPOCH = 1
SMALL = 20000

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
        self.conv1 = snn.Convolution(2, 30, 5, 0.8, 0.02)  #(in_channels, out_channels, kernel_size, weight_mean=0.8, weight_std=0.02)
        self.conv2 = snn.Convolution(30, 100, 5, 0.8, 0.05)
        #self.conv3 = snn.Convolution(250, 200, 5, 0.8, 0.05)

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
            pot = sf.pointwise_inhibition(pot) # inter-channel inhibition
            if max_layer == 1:
                winners = sf.get_k_winners(pot, 5, 3)
                self.save_data(input, pot, spk, winners)
                return spk, pot
            spk_in = sf.pad(sf.pooling(spk, 2, 2), (1,1,1,1))
            pot = self.conv2(spk_in)
            spk, pot = sf.fire(pot, 10, True)
            pot = sf.pointwise_inhibition(pot) # inter-channel inhibition
            if max_layer == 2:
                winners = sf.get_k_winners(pot, 8, 2)
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


if __name__ == "__main__":

    kernels = [ utils.DoGKernel(3,1,2), utils.DoGKernel(3,2,1)]

    filter = utils.Filter(kernels, padding = 6, thresholds = 50)

    transform = InputTransform(filter)

    data_root = 'data/'

    MNIST_train = utils.CacheDataset(MNIST(root=data_root, train=True, download=True, transform=transform))
    MNIST_test = utils.CacheDataset(MNIST(root=data_root, train=False, download=True, transform=transform))
    
    MNIST_small = torch.utils.data.random_split(MNIST_train, lengths=[SMALL, len(MNIST_train)-SMALL])[0]

    MNIST_loader = DataLoader(MNIST_train, batch_size=1000, shuffle=True)
    MNIST_test_loader = DataLoader(MNIST_test, batch_size=1000, shuffle=False)
    MNIST_small_loader = DataLoader(MNIST_small, batch_size=500, shuffle=True)

    net = CTNN()
    clf = svm.SVC()

    #net = train(net, MNIST_loader)
    #torch.save(net.state_dict(), "./checkpoint.pt")
    net.state_dict(torch.load("./checkpoint.pt"))
    train_outputs, train_y = inference(net, MNIST_loader)
    
    train_mean = train_outputs.mean()
    train_outputs -= train_mean
    clf.fit(train_outputs, train_y)
    acc=clf.score(train_outputs, train_y)
    print ("Training Accuracy is %.3f" % acc)
    test_outputs, test_y  = inference(net, MNIST_test_loader)
    test_outputs -= train_mean
    
    acc = clf.score(test_outputs, test_y)
    print ("Test Accuracy is %.3f" % acc)
    






            
            
