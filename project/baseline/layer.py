import math
import numpy as np
import matplotlib.pyplot as plt

WMAX = 8

def count_pos(data):
    return sum(n > -1 for n in data)

def stdp_update_rule(layer, winning_spiketime, update_probability=1/32):
    '''2
    Arguments:
    Input:
        layer: weights of which layer you wanna do stdp update
        winning_spiketime: spikestimes after WTA
        update_probability: weight update probability
    Return:
        updated_layer: The layer whose weights are updated
    '''
    input_spikes = layer.prev_layer.output
    pre_inhibit_spikes = layer.output
    num_of_input = input_spikes.shape[0]

    for i in range(num_of_input): # update for each input
        output_spikes = winning_spiketime[i].max()
        for j in range(len(layer.neurons)): # iterate across each neuron
            for k in range(len(layer.neurons[j].weight)): # update each weight, respectively
                if output_spikes == -1 or output_spikes != pre_inhibit_spikes[i][j]: #No output spike
                    if pre_inhibit_spikes[i][j] != -1 and input_spikes[i][k] != -1: #pre-inhibit and input spike
                        layer.neurons[j].weight[k] = 0
                    elif pre_inhibit_spikes[i][j] == -1 and input_spikes[i][k] != -1:
                        if np.random.random_sample()<update_probability:
                            layer.neurons[j].weight[k] = min(WMAX, layer.neurons[j].weight[k]+1)
                else:
                    if input_spikes[i][k] != -1 and input_spikes[i][k] <= output_spikes:
                        layer.neurons[j].weight[k] = WMAX
                    else:
                        layer.neurons[j].weight[k] = 0

    # print ("Weight after updating")
    # for neuron in layer.neurons:
    #     print (neuron.weight)
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
        #print(input_spikes)
        if mode == 'LowPass':
            for i in range(input_spikes.shape[0]):
                #print(count_pos(input_spikes[i]))
                if count_pos(input_spikes[i]) > (self.threshold -1):
                    input_spikes[i].fill(-1) #NO SPIKE
                #print(count_pos(input_spikes[i]))
                #print(input_spikes)
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
    def __init__(self, input_dim, layer_id, prev_layer, threshold, initial_weight, receptive_field):
        super(Excitatory_Nueron, self).__init__(layer_id, prev_layer, threshold, receptive_field)
        self.input_dim = input_dim
        self.weight = np.random.randint(low=0, high=initial_weight, size=input_dim)
        #self.weight = np.ones(8)
        self.wave = np.zeros(input_dim)

    def reset(self):
        self.wave = np.zeros(self.input_dim)

    def process_image(self, data):
        """
        Step funtion funtionality
        """
        self.input = data
        self.output = []
        for sample in self.input:
            self.reset()
#            sample = np.sort(sample)
            for i, x in enumerate(sample):
                if x != -1:
                    self.wave[x:] += self.weight[i]
            diff = self.wave > self.threshold
            if True in diff:
                self.output.append(np.where(diff == True)[0][0]) 
            else:
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
                                          initial_weight=initial_weight) for _ in range(output_dim)]

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
                fire_neuron = np.random.choice(fire_neuron) #random choose one to fire if multiple
                mask[fire_neuron] = 1
                self.output[i] = mask * sample
                self.output[i] -= 1
        return self.output

    def forward(self, data):
        return self.process_image(data.copy())