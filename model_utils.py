'''General Utility Functions'''
'''
Adapted from Stanford's CS320 code examples
Params, set_logger, save_dict_to_json
'''

import json
import logging
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
