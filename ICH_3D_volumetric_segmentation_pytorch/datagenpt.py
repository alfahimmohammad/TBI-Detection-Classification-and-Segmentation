#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 28 12:49:00 2020

@author: sindhura
"""
import torch
import numpy as np
from torch.utils.data import Dataset

class DataGenerator(Dataset):
    'Generates data for PyTorch'
    def __init__(self, list_IDs, ct_scans, masks, batch_size=1, dim=(512,512,None)):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.ct_scans = ct_scans
        self.masks = masks
        self.list_IDs = list_IDs


    def __len__(self):
        'Denotes the number of batches per epoch'
        return len(self.list_IDs)
        

    def __getitem__(self, index):
        'Generate one batch of data'

        # Find list of IDs
        ID = self.list_IDs[index]

        # Generate data
        X = self.ct_scans[str(ID)] #dtype = unit8
        X = torch.from_numpy(X)
        X = X.permute(2,0,1).unsqueeze(0)
        X = X.type(torch.FloatTensor)

        Y = self.masks[str(ID)] #dtype = float64
        Y = Y/255
        Y = torch.from_numpy(Y)
        Y = Y.permute(2,0,1).unsqueeze(0)
        Y = Y.type(torch.FloatTensor)
        
        return X, Y
