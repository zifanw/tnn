import math
import numpy as np
import matplotlib.pyplot as plt

def count_pos(data):
    return sum(n > 0 for n in data)

class Layer():
    def __init__(self, layer_id, prev_layer, threshold, receptive_field):
        self.layer_id = layer_id
        self.prev_layer = prev_layer
        self.threshold = threshold
        self.rf = receptive_field
        self.N,_,_ = self.prev_layer.raw_data.shape
        self.weights = np.zeros(18)

    def reset(self):
        self.weights.fill(0) 
        # Reset the network, clearing out any accumulator variables, etc

    def process_image(self):
        """
        This function will control the different processing steps for a
        single image

        Notice that you can get to values in the previous layer through
        self.prev_layer
        """
        pass


    def write_spiketimes(self, path, receptive_field):
        # create a file with: image_number, spike_position, spike_time
        pass

class Inhibitory_Layer(Layer):
    def __init__(self, layer_id, prev_layer, threshold, receptive_field):
        super(Inhibitory_Layer, self).__init__(layer_id, prev_layer, threshold, receptive_field)
        self.input = None
        self.output = None

    def process_image(self, mode='LowPass'):
        """
        Low Pass Filter funcitoning as Inhibitory Cell
        """
        input_spikes = self.prev_layer.forward(self.rf)
        self.input = input_spikes.copy()
        if mode == 'LowPass':
            for i in range(input_spikes.shape[0]):
                if count_pos(input_spikes[i]) > (self.threshold -1):
                    input_spikes[i].fill(-1) #NO SPIKE
        elif mode == 'HighPass':
                if count_pos(input_spikes[i]) < (self.threshold -1):
                    input_spikes[i].fill(-1) #NO SPIKE
        else:
            print("Current mode is not supported")

        self.output = input_spikes[:,None]
        return self.output
