import math
import numpy as np
import matplotlib.pyplot as plt
from layer import Inhibitory_Layer as IL
from layer import Excitatory_Layer as EL
from layer import LateralInhibiton_Layer as LL

from sklearn.datasets import fetch_mldata
import firstlayer as firstlayer
import layer as layer
import time


mnist = fetch_mldata('MNIST original')
            
N, _ = mnist.data.shape

# Shuffle data
np.random.seed(2)
shuffledindex = np.random.permutation([i for i in (range(N))])
mnistdata = mnist.data[shuffledindex,:]
mnisttarget = mnist.target[shuffledindex]

# Reshape the data to be square
mnist.square_data = mnistdata.reshape(N,28,28)

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

num_training_imgs = 60000
start = time.time()
x1 = layer1.forward(mnist.square_data[0:num_training_imgs], 12)
x2 = layer2.forward(x1, mode='Exact')
x3 = layer3.forward(x2)
x4 = layer4.forward(data=x3)

layer3 = layer.stdp_update_rule(layer3, x4)
end = time.time()
print("Total training time", round(end - start, 1))

num_testing_imgs = 10000
start = time.time()

x1 = layer1.forward(mnist.square_data[0:num_testing_imgs], 12)
x2 = layer2.forward(x1, mode='Exact')
x3 = layer3.forward(x2)
x4 = layer4.forward(data=x3)
end = time.time()
print("Total testing time", round(end - start, 1))





