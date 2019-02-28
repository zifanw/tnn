import math
import numpy as np
import matplotlib.pyplot as plt
from layer import Inhibitory_Layer as IL
from layer import Excitatory_Layer as EL
from sklearn.datasets import fetch_mldata
import firstlayer as firstlayer
import layer as layer
import time
import gzip
import pickle as pkl

#mnist = fetch_mldata('MNIST original')
path = 'mnist.pkl.gz'
train_X = None
train_y = None
with gzip.open(path, 'rb') as f:
    (train_X, train_y), (val_X, val_y), (test_X, test_y) = pkl.load(f, encoding='latin1')
            
N, _ = train_X.shape

# Reshape the data to be square
np.random.seed(2)

shuffledindex = np.random.permutation([i for i in (range(N))])
mnistdata = train_X[shuffledindex,:]
mnisttarget = train_y[shuffledindex]

square_data = mnistdata.reshape(N,28,28)
num_training_imgs = 10000

layer1 = firstlayer.FirstLayer(1)
layer2 = IL(layer_id=1,
            prev_layer=layer1,
            threshold=3,
            receptive_field=12)

layer3 = EL(input_dim=8,
            output_dim=16,
            layer_id=3,
            prev_layer=layer2,
            threshold=2,  
            initial_weight=1)
layer4 = LL(layer_id=4,
            prev_layer=layer3,
            threshold=None,
            receptive_field=None)
#layer1_train = firstlayer.FirstLayer(1, square_data[0:num_training_imgs - 1], mnisttarget[0:num_training_imgs - 1])

x1 = layer1.forward(mnist.square_data[-10:], 12)
x2 = layer2.forward(x1, mode='Exact')
x3 = layer3.forward(x2)
x4 = layer4.forward(data=x3)
print (x1)
print (x2)
print (x3)
print (x4)

#Update Weights of a certain layer:
layer3 = stdp_update_rule(layer3, x4)


layer3.write_spiketimes('spiketimes.csv', layer3.output)

#num_testing_imgs = 10000




