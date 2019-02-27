import math
import numpy as np
import matplotlib.pyplot as plt

WMAX = 7


def count_pos(data):
    return sum(n > -1 for n in data)

def stdp_update_rule(layer, winning_spiketime, update_probability=1/32):
    '''
    Arguments:
    Input: 
        layer: weights of which layer you wanna do stdp update
        winning_spiketime: spikestimes after WTA
        update_probability: weight update probability
    Return:
        updated_layer: The layer whose weights are updated
    '''

    prev_layer_ouput = layer.prev_layer.output
    neuron_spiketime = layer.output
    num_of_imput = prev_layer_ouput.shape[0]

    for i in range(num_of_imput): # update for each input
        for j in range(len(layer.neurons)): # iterate across each neuron
            input_spikes = prev_layer_ouput[i]
            output_spikes = True if neuron_spiketime[i][j] > -1 else False
            pre_inhibition = True if winning_spiketime[i][j] > -1 else False

            if output_spikes:
                for k in range(len(layer.neurons[j].weight)): # update each weight, respectively
                    if input_spikes[k] > -1: #if input
                        if winning_spiketime[i][j] >= input_spikes[k]: 
                            layer.neurons[j].weight[k] = WMAX
                        else:
                            layer.neurons[j].weight[k] = 0
                    else:
                        layer.neurons[j].weight[k] = 0
            else:
                for k in range(len(layer.neurons[j].weight)): # update each weight, respectively
                    if pre_inhibition and input_spikes[k] > -1:
                        layer.neurons[j].weight[k] = 0
                    elif not pre_inhibition:
                        if input_spikes[k] > -1 and np.random.random_sample()<update_probability:
                            layer.neurons[j].weight[k] += 1
    for neuron in layer.neurons:
        print (neuron.weight)
    return layer


class Layer():
    def __init__(self, layer_id, prev_layer, threshold, receptive_field):
        self.layer_id = layer_id
        self.prev_layer = prev_layer
        self.threshold = threshold
        self.rf = receptive_field
        #self.N,_,_ = self.prev_layer.raw_data.shape

    def reset(self):
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


class Inhibitory_Layer(Layer):
    def __init__(self, layer_id, prev_layer, threshold, receptive_field):
        super(Inhibitory_Layer, self).__init__(layer_id, prev_layer, threshold, receptive_field)
        self.input = None
        self.raw_data = self.input
        self.output = None

    def process_image(self, input_spikes, mode):
        """
        Low Pass Filter funcitoning as Inhibitory Cell
        """
        self.input = input_spikes
        if mode == 'LowPass':
            for i in range(input_spikes.shape[0]):
                if count_pos(input_spikes[i]) > (self.threshold -1):
                    input_spikes[i].fill(-1) #NO SPIKE
        elif mode == 'HighPass':
            for i in range(input_spikes.shape[0]):
                if count_pos(input_spikes[i]) < (self.threshold -1):
                    input_spikes[i].fill(-1) #NO SPIKE
        elif mode == 'Exact':
            for i in range(input_spikes.shape[0]):
                if count_pos(input_spikes[i]) != self.threshold:
                    input_spikes[i].fill(-1) #NO SPIKE
        else:
            print("Current mode is not supported")

        self.output = input_spikes
        return self.output

    def forward(self, data, mode):
        return self.process_image(data.copy(), mode)


class Excitatory_Nueron(Layer):
    def __init__(self, input_dim, layer_id, prev_layer, threshold, receptive_field, initial_weight):
        super(Excitatory_Nueron, self).__init__(layer_id, prev_layer, threshold, receptive_field)
        self.input_dim = input_dim
        self.weight = np.zeros(input_dim) + initial_weight
        self.wave = np.zeros(input_dim)

    def reset(self):
        self.wave = np.zeros(self.input_dim)
        self.FireFlag = 0

    def process_image(self, data):
        """
        Step funtion funtionality
        """
        self.input = data
        self.output = []
        for sample in self.input:
            self.reset()
            sample = np.sort(sample)
            for i, x in enumerate(sample):
                if x != -1:
                    self.wave[x:] += self.weight[i]
                diff = self.wave > self.threshold
                if True in diff:
                    self.output.append(np.where(diff == True)[0][0])
                    self.FireFlag += 1
                    break
            if not self.FireFlag:
                self.output.append(-1)
        self.output = np.asarray(self.output)[:,None]
        return self.output


class Excitatory_Layer(Layer):
    def __init__(self, input_dim, output_dim, layer_id, prev_layer, threshold, initial_weight,receptive_field=None):
        super(Excitatory_Layer, self).__init__(layer_id, prev_layer, threshold, receptive_field)
        self.input_dim = input_dim
        #self.weight = initial_weight
        self.output = None
        self.neurons = [Excitatory_Nueron(input_dim=input_dim,
                                          layer_id=layer_id,
                                          prev_layer=self.prev_layer,
                                          threshold=threshold,
                                          receptive_field=receptive_field,
                                          initial_weight=initial_weight)] * output_dim

    def reset(self):
        for i in range(len(self.neurons)):
            self.neurons[i].reset()

    def process_image(self, data):
        output = []
        for i in range(len(self.neurons)):
            output.append(self.neurons[i].process_image(data))
        self.output = np.concatenate(output, axis=1)
        return self.output

    def forward(self, data):
        return self.process_image(data.copy())


class LateralInhibiton_Layer(Layer):
    def __init__(self, layer_id, prev_layer, threshold, receptive_field):
        super(LateralInhibiton_Layer, self).__init__(layer_id, prev_layer, threshold, receptive_field)
        self.input = None
        self.output = []

    def process_image(self, data):
        """
        Step funtion funtionality
        """
        self.output = data
        for i, sample in enumerate(self.output):
            sign = sample > -1
            if True in sign:
                pos = sample[sample>-1]
                fire_neuron = np.where(sample == pos.min())[0]
                self.output[i] += 1
                mask = np.zeros(sample.shape[0])
                mask[fire_neuron] = 1
                self.output[i] = mask * sample
                self.output[i] -= 1
        return self.output
    
    def forward(self, data):
        return self.process_image(data.copy())


