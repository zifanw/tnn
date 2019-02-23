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
        #self.N,_,_ = self.prev_layer.raw_data.shape
        self.weights = np.zeros(18)

    def reset(self):
        self.weights.fill(0)
        # Reset the network, clearing out any accumulator variables, etc
        pass

    def process_image(self):
        """
        This function will control the different processing steps for a
        single image

        Notice that you can get to values in the previous layer through
        self.prev_layer
        """
        pass


    def write_spiketimes(self, path, spikes):
        # create a file with: image_number, spike_position, spike_time
        i = 0
        f = open(path, "a")
        f.write('img_number'+','+'spike position'+','+'spike time\n')
        for x in spikes:
            st =''
            for spike_time in x:
                st += str(spike_time) if spike_time != -1 else '-'
            f.write(str(i)+','+"(12 12)to(14 14)"+','+st+'\n')
            i += 1
        f.close()
        print ("Finish writing spike times to "+path)
        
class Winner_Take_All_Layer(Layer):
    def __init__(self, layer_id, prev_layer, threshold, receptive_field):
        super(Winner_Take_All_Layer, self).__init__(layer_id, prev_layer, threshold, receptive_field)
        self.input = None
        self.output = []
    def process_image(self, mode = 'LowPass'):
        """
        Winner take all functionality
        """
        input_spikes = self.prev_layer.output
        self.input = input_spikes.copy()
        for sample in self.input:
            min_spike = np.min(sample)
            for x in sample:
                if (x == min_spike):
                    output = min_spike
                else:
                    output = -1
            self.output.append(output)
        return self.output 
    
class Inhibitory_Layer(Layer):
    def __init__(self, layer_id, prev_layer, threshold, receptive_field):
        super(Inhibitory_Layer, self).__init__(layer_id, prev_layer, threshold, receptive_field)
        self.input = None
        self.raw_data = self.input
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
            for i in range(input_spikes.shape[0]):
                if count_pos(input_spikes[i]) < (self.threshold -1):
                    input_spikes[i].fill(-1) #NO SPIKE
        else:
            print("Current mode is not supported")

        self.output = input_spikes
        return self.output


class Excitatory_Layer(Layer):
    def __init__(self, layer_id, prev_layer, threshold, receptive_field):
        super(Excitatory_Layer, self).__init__(layer_id, prev_layer, threshold, receptive_field)
        self.input = None
        self.output = []

    def process_image(self, mode='LowPass'):
        """
        Step funtion funtionality
        """
        input_spikes = self.prev_layer.output
        self.input = input_spikes.copy()
        for sample in self.input:
            self.reset()
            for x in sample:
                if x != -1:
                    self.weights[x:] += 1
            if self.weights.sum() > 0 and self.threshold in self.weights:
                output = np.where(self.weights==self.threshold)[0][0]
            else:
                output = -1
            self.output.append(output)
        self.output = np.asarray(self.output)[:,None]
        return self.output
