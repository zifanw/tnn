import math
import numpy as np
from scipy import signal

#Layer may not be both the first layer and an output layer

def count_pos(data):
    return sum(n > -1 for n in data)

class FirstLayer:
    def __init__ (self, layer_id):
        self.layer_id = layer_id
        self.output = None

    def OnCenterFilter(self, x):
        #Edge condition is dealing within firstlayer.py
        cnt = x[:,1,1]
        srd = x.sum(axis=(1,2)) - cnt
        ave_srd = np.around(srd/8).astype('int')
        return ave_srd - cnt

    def OffCenterFilter(self, x):
        #Edge condition is dealing within firstlayer.py
        cnt = x[:,1,1]
        srd = x.sum(axis=(1,2)) - cnt
        ave_srd = np.around(srd/8).astype('int')
        return cnt - ave_srd

    def preprocess(self, my_filter, raw_data, num_bits=3):
        a = []
        for i in range(2 ** num_bits):
            a += [i] * int(256/(2 ** num_bits))
        a = np.asarray(a)
        scaled_data = a[raw_data]
        filtered_data = np.zeros_like(scaled_data)
        padded_data = np.pad(scaled_data,
                            ((0,0),(1, 1), (1, 1)),
                            'constant',
                            constant_values=0) #deal with the edge by padding around
        for i in range(1, padded_data.shape[1]-1):
            for j in range(1, padded_data.shape[2]-1):
                filtered_data[:,i-1, j-1] = my_filter(padded_data[:,i-1:i+2, j-1:j+2])
        return filtered_data

    def generate_spikes(self, rp, threshold):
        N, W, H = rp.shape
        rp_flatten = rp.flatten()
        zero_mask = np.zeros_like(rp_flatten)
        spikes = np.maximum(zero_mask, rp_flatten)
        spikes = [7 - i if i >0 else -1 for i in spikes] #-1 for no spike time
        spikes = np.asarray(spikes).reshape((N, W*H))
        return spikes

    def output_spikes(self, img, receptive_field, rp_size=2):
        rp = img[:,receptive_field:receptive_field+rp_size, receptive_field:receptive_field+rp_size]
        On_spikes = self.generate_spikes(rp, 7)
        Off_spikes = self.generate_spikes(-rp, 7)
        self.output = np.concatenate((On_spikes, Off_spikes), axis=1)
        for i in range(self.output.shape[0]):
            if count_pos(self.output[i]):
                temp = self.output[i]
                temp2 = temp[temp>-1]
                self.output[i][self.output[i]>-1] -= temp2.min()
        return self.output

    def forward(self, data, receptive_field, rp_size):
        return self.output_spikes(self.preprocess(self.OnCenterFilter, data), receptive_field, rp_size=rp_size)

    def write_spiketimes(self, img, path, receptive_field, rp_size=3):
        rp = img[:,receptive_field:receptive_field+rp_size, receptive_field:receptive_field+rp_size]
        On_spikes = self.generate_spikes(rp, 7)
        Off_spikes = self.generate_spikes(-rp, 7)
        i = 0
        f = open(path, "a")
        f.write('img_number'+','+'spike position'+','+'spike time(left 9 are On, right 9 are Off)\n')
        for x, y in zip(On_spikes, Off_spikes):
            st1 =''
            st2 = ''
            for on, off in zip(x, y):
                st1 += str(on) if on != -1 else '-'
                st2 += str(off) if off != -1 else '-'
                st = st1 + ' ' + st2
                f.write(str(i)+','+'(12,12), (12,13) ... (14, 14)'+','+st+'\n')
                i += 1
        f.close()
        print ("Finish writing "+path)
