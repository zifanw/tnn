import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_mldata
import firstlayer as firstlayer
import layer as layer

mnist = fetch_mldata('MNIST original')
N, _ = mnist.data.shape

# Reshape the data to be square
mnist.square_data = mnist.data.reshape(N,28,28)
layer1 = firstlayer.FirstLayer(1, mnist.square_data, mnist.target)

# You will need to change the instantiation of this layer 
# in order to properly initialize a new layer
layer2 = layer.Layer(2, layer1, None, None) 
