import math
import numpy as np

from layer import Inhibitory_Layer as IL
from layer import Excitatory_Layer as EL
from layer import LateralInhibiton_Layer as LL, stdp_update_rule

from sklearn.datasets import fetch_mldata
import firstlayer as firstlayer
import layer as layer
import time


mnist = fetch_mldata('MNIST original')

N, _ = mnist.data.shape

# Shuffle data
np.random.seed(2)
shuffledindex = np.random.permutation([i for i in (range(70000))])
mnistdata = mnist.data[shuffledindex]
mnisttarget = mnist.target[shuffledindex]

# Reshape the data to be square
mnistdata = mnistdata.reshape(N,28,28)

layer1 = firstlayer.FirstLayer(1)
layer2 = IL(layer_id=1,
            prev_layer=layer1,
            threshold=6,
            receptive_field=0)
layer3 = EL(input_dim=8,
            output_dim=16,
            layer_id=3,
            prev_layer=layer2,
            threshold=4,
            initial_weight=5)
layer4 = LL(layer_id=4,
            prev_layer=layer3,
            threshold=None,
            receptive_field=None)


num_training_imgs = 10000
num_testing_imgs = 10000
batch_size = 1
num_batch = num_training_imgs // batch_size
test_data = mnistdata[-num_testing_imgs:]
test_label = mnisttarget[-num_testing_imgs:]
print ("Weights before training")
for neuron in layer3.neurons:
    print (neuron.weight)
start = time.time()
offset = 0
train_data = mnistdata[offset:num_training_imgs+offset]
#for _ in range(3):
for i in range(num_batch):
    batch_data = train_data[i*batch_size:(i+1)*batch_size]
    x1 = layer1.forward(batch_data, 0)
    x2 = layer2.forward(x1, mode='LowPass')
    #print(x2)
    x3 = layer3.forward(x2)
    x4 = layer4.forward(x3)
    layer3 = stdp_update_rule(layer3, x4)
    #print('[TRAIN], batch_num: %d' % i )

end = time.time()
print("Total training time", round(end - start, 1))


start = time.time()

x1 = layer1.forward(test_data, 0)
x2 = layer2.forward(x1, mode='LowPass')
x3 = layer3.forward(x2)
x4 = layer4.forward(x3)

end = time.time()
print("Total testing time", round(end - start, 1))
print ("Weights after training")
for neuron in layer3.neurons:
    print (neuron.weight)
metrics = np.zeros((16, 10))
pred = x4 != -1
print("Test Results")
print (x4)
for sample, label_id in zip(pred,test_label):
    if True in sample:
        cluster_id = np.where(sample==True)[0][0]
        metrics[cluster_id][int(label_id)] += 1
print (metrics)

np.save('test_result.npy', metrics)
