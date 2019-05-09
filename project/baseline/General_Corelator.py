import math
import numpy as np

from layer import Inhibitory_Layer as IL
from layer import Excitatory_Layer as EL
from layer import LateralInhibiton_Layer as LL, stdp_update_rule

from sklearn.datasets import fetch_mldata
import firstlayer as firstlayer
import layer as layer
import time

def calculate_max(input):
    output = []
    for i in input:
        output.append(max(i))
    return output
def calculate_sum(input):
    output = []
    for i in input:
        output.append(np.sum(i))
    return output
def calculate_purity(max_outputs, sum_outputs):
        purity = np.sum(max_outputs)/np.sum(sum_outputs)
        return purity
def calculate_coverage(sum_outputs, total_instances):
    coverage = np.sum(sum_outputs)/total_instances
    return coverage

rf_size = 11
class BandCorelator():
    def __init__(self, threshold_lp, layer1, dim):
        self.layer2 = IL(layer_id=1,
             prev_layer=layer1,
             threshold=threshold_lp,
             receptive_field=0)
        self.layer3 = EL(input_dim=dim,
             output_dim=16,
             layer_id=3,
             prev_layer=self.layer2,
             threshold=6,
             initial_weight=7)
        self.layer4 = LL(layer_id=4,
             prev_layer=self.layer3,
             threshold=None,
             receptive_field=None)
        
    def forward(self, x1, train=True):
        x2 = self.layer2.forward(x1, mode='LowPass')
        x3 = self.layer3.forward(x2)
        x4 = self.layer4.forward(x3)
        if (train):
            self.layer3 = stdp_update_rule(self.layer3, x4)
        return x4

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
bc = []
for r in range(0, 4):    
    bc.append(BandCorelator(rf_size, layer1, rf_size*rf_size*2))    

num_training_imgs = 60000
num_testing_imgs = 10000
batch_size = 1
num_batch = num_training_imgs // batch_size
test_data = mnistdata[-num_testing_imgs:]
test_label = mnisttarget[-num_testing_imgs:]


#start = time.time()
offset = 0
train_data = mnistdata[offset:num_training_imgs+offset]
final_bc = BandCorelator(rf_size, layer1, 16)

print ("Weights before training")
for neuron in final_bc.layer3.neurons:
     print (neuron.weight)
for _ in range(1):
    for i in range(num_batch):
        final_train = []
        batch_data = train_data[i*batch_size:(i+1)*batch_size]
        x1 = layer1.forward(batch_data, 0, rf_size)
        for r in range(0, 4):
            final_train.append(bc[r].forward(x1).flatten().tolist())
        final_bc.forward(np.array(final_train))

print ("Weights after training")
for neuron in final_bc.layer3.neurons:
     print (neuron.weight)
end = time.time()
#print("Total training time", round(end - start, 1))


#start = time.time()


test_result = []
for _ in range(1):
    for i in range(num_testing_imgs):
        final_test = []
        batch_data = test_data[i*batch_size:(i+1)*batch_size]
        x1 = layer1.forward(batch_data, 0, rf_size)
        for r in range(0, 4):
            final_test.append(bc[r].forward(x1, train=False).flatten().tolist())
        test_result.append(final_bc.forward(np.array(final_test), train=False))
        #print(final_bc.forward(np.array(final_test), train=False))

#end = time.time()
metrics = np.zeros((16, 10))
pred = np.array(test_result) != -1


for sample, label_id in zip(pred,test_label):
    if True in sample:
        cluster_id = np.where(sample==True)[0][0]
        metrics[cluster_id][int(label_id)] += 1

test_max = calculate_max(metrics)[::-1]
test_sum = calculate_sum(metrics)[::-1]
index = np.argsort(calculate_max(metrics))[::-1]
metrics = metrics[index]
np.save('test_result.npy', metrics)
print (metrics)

print ("The max values are:")
print (test_max)
print ("The total values are:")
print (test_sum)
print ("The purity is:")
print (calculate_purity(test_max, test_sum))
print ("The coverage is:")
print (calculate_coverage(test_sum, num_testing_imgs ))






