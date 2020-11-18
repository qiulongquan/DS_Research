import os
import numpy as np
import pickle

def pickle_load(path):
    with open(path, mode='rb') as f:
        data = pickle.load(f)
        return data

