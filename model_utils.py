'''General Utility Functions'''
'''
Adapted from Stanford's CS320 code examples
Params, set_logger, save_dict_to_json
'''

import json
import logging
import obspy
from obspy.clients.fdsn import Client
from scipy.signal import lfilter, butter, decimate, hann
import numpy as np

#import tensorflow as tf
#import numpy as np
#import matplotlib.pyplot as plt

def get_model_name(pdict, remove_keys=['folder', 'model', 'prefix', 'catalog', 'name'], keep_keys=[]):
    my_keys = pdict.keys()
    
    if remove_keys:
        my_keys = [key for key in my_keys if not any(ele in key for ele in remove_keys)]
        
    if keep_keys:
        my_keys = [key for key in my_keys if any(ele in key for ele in keep_keys)]
        
    name = []
    
    for key in my_keys:
        if type(pdict[key]) is list:
            name.append(f'{key}:{"x".join(map(str, pdict[key]))}')
        else:
            name.append(f'{key}:{pdict[key]}')
    return '|'.join(name)

class Params():
    '''
    Class that load hyerparameters from a json file.
    Example:
        
    params = Params(json_path)
    print(params.learning_rate)
    parms.learning_rate = 0.001
    '''
    
    def __init__(self, json_path):
        self.update(json_path)
    
    def save(self, json_path):
        """ Saves parameters to json file """
        with open(json_path, 'w') as f:
            json.dump(self.__dict__, f, indent=4)
    
    def update(self, json_path):
        """ Loads parameters from json file """
        with open(json_path) as f:
            params=json.load(f)
            self.__dict__.update(params)
    
    @property
    def model_name(self):
        self.__dict__['model_name'] = get_model_name(self.__dict__)
        return self.__dict__['model_name']
    
    @property
    def dict(self):
        return self.__dict__
    
    def __str__(self):
        for key in self.dict.keys():
            print(f'{key:15}: {self.dict[key]}')
        return ''
    def __repr__(self):
        print(self)
        return ''
    
def set_logger(log_path):
    logger = logging.getLogger()
    logger.setLevel(logging.ERROR)
    
    if not logger.handlers:
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
        logger.addHandler(file_handler)
        
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(stream_handler)
        
def save_dict_to_json(d, json_path):
    with open(json_path, 'w') as f:
        d['model_name'] = get_model_name(d)
        json.dump(d, f, indent=4)
        
def datetime2epoch(timestamp):
    return (timestamp - datetime.datetime(1970, 1, 1)).total_seconds()

def get_wav(sta, st_time, en_time, pdict, source):
    
    X = []
    
    if sta[-2:] == 'AR':
        sta = sta[:-2] + '31'
        
    stream = source.get_waveforms(network="*", location="", station=sta, channel="BH?",
                             starttime=obspy.UTCDateTime(st_time), endtime=obspy.UTCDateTime(en_time))
    for cmpt in pdict.cmpts:
        X.append(stream.select(component=cmpt)[0].data)
         
    X = np.stack(X, axis=-1)
    if X.shape >= (int(pdict.w_len * pdict.s_rate), len(pdict.cmpts)):
        return X
    else:
        raise Exception()

def DAT_normalize(X):
    X = X - np.expand_dims(np.mean(X,1),1)
    X = X / (np.expand_dims(np.expand_dims(np.abs(X).max(1).max(1), 1), 1) + .001)
    return X

def butter_bandpass(lowcut, highcut, fs, order=8):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq

    b, a = butter(order, [low, high], btype='band')
    return b, a

def DAT_filter(X, pdict, order=3):
    lowcut = pdict.f_low
    highcut = pdict.f_high
    fs = pdict.s_rate
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    X_filt = lfilter(b, a, X, axis=1)
    

    if pdict.zp is True:
        X_filt = lfilter(b, a, X_filt[:,::-1,:])[::-1]
        
    return X_filt 

def DAT_taper(X, taper_percentage=.1):
    npts = X.shape[1]
    taper_len = int(npts * taper_percentage)
    taper_sides = hann(2 * taper_len + 1)
    taper = np.hstack((taper_sides[:taper_len], np.ones(npts - taper_len)))
    return X * np.reshape(taper,(1,-1,1))

