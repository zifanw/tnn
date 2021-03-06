import torch.nn as nn
import SpykeTorch.snn as snn
import SpykeTorch.functional as sf
import numpy as np
import SpykeTorch.utils as utils
import torchvision.transforms as transforms
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torchvision.datasets import MNIST
from torch.autograd import Variable
from sklearn import svm
from sklearn.metrics import accuracy_score
from tqdm import tqdm, trange

use_cuda = True
MAX_EPOCH = 1
MAX_EPOCH2 = 10
Threshold_1 = 15
Threshold_2 = 10

# class InputTransform:
#    def __init__(self, filter):
#       self.to_tensor = transforms.ToTensor()
#       self.filter = filter
#       self.temporal_transform = utils.Intensity2Latency(30, to_spike=True)
#    def __call__(self, image):
#       image = self.to_tensor(image) * 255
#       image.unsqueeze_(0)
#       C = image.size(1)
#       x = []
#       for c in range(C):
#         img = image[:,c] # 1x32x32
#         img.unsqueeze_(1) # 1x1x32x32
#         img = self.filter(img) #1x2x32x32
#         x.append(img)
#       image = torch.cat(x, 1)
#       image = sf.local_normalization(image, 8)
#       return self.temporal_transform(image)

class InputTransform:
   def __init__(self, filter):
      self.to_tensor = transforms.ToTensor()
      self.filter = filter
      self.temporal_transform = utils.Intensity2Latency(30, to_spike=True)
   def __call__(self, image):
      image = self.to_tensor(image) * 255
      image.unsqueeze_(0)
      image = self.filter(image)
      image = sf.local_normalization(image, 8)
      return self.temporal_transform(image)


class CTNN(nn.Module):
    def __init__(self):
        super(CTNN, self).__init__()
        self.conv1 = snn.Convolution(2, 30, 7, 0.8, 0.02)  #(in_channels, out_channels, kernel_size, weight_mean=0.8, weight_std=0.02)
        self.conv2 = snn.Convolution(30, 100, 5, 0.8, 0.05)
        # self.conv3 = snn.Convolution(12, 300, 7, 0.8, 0.02)
        #self.conv4 = snn.Convolution(100, 200, 3, 0.8, 0.05)

        self.stdp1 = snn.STDP(self.conv1, (0.004, -0.003))
        self.stdp2 = snn.STDP(self.conv2, (0.004, -0.003))
        # self.stdp3 = snn.STDP(self.conv3, (0.004, -0.003))
        self.ctx = {"input_spikes": None, "potentials": None, "output_spikes":None, "winners":None}

        # self.pool = torch.nn.AdaptiveMaxPool3d((300,18,18))

    def save_data(self, input_spk, pot, spk, winners):
        self.ctx["input_spikes"] = input_spk
        self.ctx["potentials"] = pot
        self.ctx["output_spikes"] = spk
        self.ctx["winners"] = winners

    # def forward(self, input, max_layer):
    #     input = sf.pad(input, (2,2,2,2))
    #     if self.training: #forward pass for train
    #         if max_layer < 3:
    #             pot = self.conv1(input)
    #             spk, pot = sf.fire(pot, 10, True)
    #             pot = sf.pointwise_inhibition(pot) # inter-channel inhibition
    #             if max_layer == 1:
    #                 winners = sf.get_k_winners(pot, 5, 3)
    #                 self.save_data(input, pot, spk, winners)
    #                 return spk, pot
    #             spk_in = sf.pad(sf.pooling(spk, 2, 2), (1,1,1,1))
    #             pot = self.conv2(spk_in)
    #             spk, pot = sf.fire(pot, 10, True)
    #             pot = sf.pointwise_inhibition(pot) # inter-channel inhibition
    #             if max_layer == 2:
    #                 winners = sf.get_k_winners(pot, 8, 5)
    #                 self.save_data(spk_in, pot, spk, winners)
    #                 return spk, pot
    #         else:
    #             pot = self.conv3(input)
    #             spk, pot = sf.fire(pot, 10, True)
    #             pot = sf.pointwise_inhibition(pot) # inter-channel inhibition
    #             if max_layer == 3:
    #                 winners = sf.get_k_winners(pot, 5, 3)
    #                 self.save_data(input, pot, spk, winners)
    #                 return spk, pot
    #     else:
    #         pot = self.conv1(input)
    #         spk, pot = sf.fire(pot, 10, True)
    #         pot_1 = self.conv2(sf.pad(sf.pooling(spk, 2, 2), (1,1,1,1)))
    #         pot_2 = self.conv3(input)
    #         pot_2 = self.pool(pot_2)
    #         return spk, torch.cat([pot_1, pot_2], dim=1)

    def forward(self, input, max_layer):
        input = sf.pad(input, (2,2,2,2))
        if self.training: #forward pass for train
            pot = self.conv1(input)
            spk, pot = sf.fire(pot, Threshold_1, True)
            pot = sf.pointwise_inhibition(pot) # inter-channel inhibition
            if max_layer == 1:
                winners = sf.get_k_winners(pot, 5, 3)
                self.save_data(input, pot, spk, winners)
                return spk, pot
            spk_in = sf.pad(sf.pooling(spk, 2, 2), (1,1,1,1))
            pot = self.conv2(spk_in)
            spk, pot = sf.fire(pot, Threshold_2, True)
            pot = sf.pointwise_inhibition(pot) # inter-channel inhibition
            if max_layer == 2:
                winners = sf.get_k_winners(pot, 8, 5)
                self.save_data(spk_in, pot, spk, winners)
                return spk, pot

        else:
            pot = self.conv1(input)
            spk, pot = sf.fire(pot, Threshold_1, True)
            pot = self.conv2(sf.pad(sf.pooling(spk, 2, 2), (1,1,1,1)))
            # spk, pot = sf.fire(pot, 10, True)
            # pot = self.conv3(sf.pad(sf.pooling(spk, 2, 2), (1,1,1,1)))
           # spk = sf.fire(pot, 10)
           # pot = self.conv3(sf.pad(sf.pooling(spk, 3, 3), (2,2,2,2)))
           # spk = sf.fire_(pot)
    
            return spk, pot

    def stdp(self, layer_idx):
        if layer_idx == 1:
                self.stdp1(self.ctx["input_spikes"], self.ctx["potentials"], self.ctx["output_spikes"], self.ctx["winners"])
        if layer_idx == 2:
            self.stdp2(self.ctx["input_spikes"], self.ctx["potentials"], self.ctx["output_spikes"], self.ctx["winners"])
        if layer_idx == 3:
            self.stdp3(self.ctx["input_spikes"], self.ctx["potentials"], self.ctx["output_spikes"], self.ctx["winners"])

class Neural_Network(nn.Module):
    def __init__(self, output_size):
        super(NN, self).__init__()
        self.layer1 = nn.Linear(100, 1024) 
        self.act = nn.ReLU()
        self.layer2 = nn.Linear(1024, output_size)
    
    def forward(self, x):
        x = self.layer1(x)
        x = self.act(x)
        return self.layer2(x)

    
class Model:
    def __init__(self):
        self.net = Neural_Network(10)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=0.001)
        self.train_data_loader = None
        self.test_data_loader = None
        self.test_size = 0
    
    def get_dataset(self, dataset):
        train_dataset, test_dataset = dataset[0], dataset[1]
        self.test_size = test_dataset[0].shape[0]
        train_dataset = TensorDataset(train_dataset[0], train_dataset[1])
        test_dataset = TensorDataset(test_dataset[0], test_dataset[1])
        
        self.train_data_loader = DataLoader(self.train_dataset,
                                        batch_size=64,
                                        shuffle=True,
                                        num_workers=8)
        self.test_data_loader = DataLoader(test_dataset,
                                        batch_size=64,
                                        shuffle=False,
                                        num_workers=8)
        print ("-----------Training Data is Loaded----------------")

    

    def run(self):
        for e in range(MAX_EPOCH2):
            epoch_loss = 0
            correct = 0
            for batch_idx, (data, label) in enumerate(self.train_data_loader):
                self.optimizer.zero_grad()
                X = Variable(data)
                Y = Variable(label)
                self.model = self.model.cuda()
                X = X.cuda()
                Y = Y.cuda()
                output = self.model(X)
                loss = self.criterion(output, Y)
                loss.backward()
                self.optimizer.step()
                pred = output.data.max(1)[1]
                predicted = pred.eq(Y.data.view_as(pred))
                correct += predicted.sum()
                epoch_loss += loss.data[0]
            epoch_loss = epoch_loss.cpu()
            correct = correct.cpu()
            total_loss = epoch_loss.numpy()/self.train_size
            train_acc = correct.numpy()/self.train_size
            print("epoch: {0}, loss: {1:.8f}, train_acc: {2:.8f}".format(e, total_loss, train_acc))
        self.save_model('./clfmodel.pt')

    def inference(self):
        correct = 0
        self.model.eval()
        with torch.no_grad():
            for data, label in self.test_data_loader:
                X = Variable(data)
                Y = Variable(label)
                if self.gpu:
                    self.model = self.model.cuda()
                    X = X.cuda()
                    Y = Y.cuda()
                out = self.model(X)
                pred = out.data.max(1, keepdim=True)[1]
                predicted = pred.eq(Y.data.view_as(pred))
                correct += predicted.sum()
            correct = correct.cpu()
            return correct.numpy() / self.test_size


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
    # for epoch in trange(MAX_EPOCH):
    #     for data, _ in tqdm(data_loader):
    #         train_unsupervised(net, data, 3)
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
            _, pot = net(data_in, 4)
            pot=pot.cpu().numpy()
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

    # kernels = [ utils.DoGKernel(3,1,2), utils.DoGKernel(3,2,1),
    #             utils.OnCenter(3), utils.OffCenter(3)]

    kernels = [utils.DoGKernel(3,1,2), utils.DoGKernel(3,2,1)]


    filter = utils.Filter(kernels, padding = 6, thresholds = 50)

    transform = InputTransform(filter)

    data_root = 'data/'

    MNIST_train = utils.CacheDataset(MNIST(root=data_root, train=True, download=True, transform=transform)) # 60000 x 30 x 30
    MNIST_test = utils.CacheDataset(MNIST(root=data_root, train=True, download=True, transform=transform)) # 10000 x 30

    MNIST_loader = DataLoader(MNIST_train, batch_size=1000, shuffle=True)
    MNIST_test_loader = DataLoader(MNIST_test, batch_size=1000, shuffle=False)


    net = CTNN()
    clf = Model()
    

    net = train(net, MNIST_loader)
    torch.save(net.state_dict(), "./MNISTcheckpoint.pt")
    net.state_dict(torch.load("./MNISTcheckpoint.pt"))
    train_outputs, train_y = inference(net, MNIST_loader)
    test_outputs, test_y  = inference(net, MNIST_test_loader)
    train_outputs, test_outputs = preprocess(train_outputs, test_outputs)

    train_data_set = (train_outputs, train_y)
    test_data_set = (test_outputs, test_y)
    data_set = [train_data_set, test_data_set]
    clf.get_dataset(data_set)
    clf.run()
    test_acc = clf.inference()

    print ("Training Accuracy is %.3f" % acc)

    acc = clf.score(test_outputs, test_y)
    print ("Test Accuracy is %.3f" % acc)
