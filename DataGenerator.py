# -*- coding: utf-8 -*-
"""
Created on Sun Aug  9 20:16:05 2020

@author: sidhant
"""
import os
import numpy as np
from PIL import Image
import random as rd

import torch
from torch.utils.data import Dataset

class DatasetGenerator (Dataset):
    
    #-------------------------------------------------------------------------------- 
    
    def __init__ (self, pathImageDirectory, transform):
    
        self.listImagePaths = []
        self.listImageLabels = []
        self.files=[]
        self.transform = transform
        
        
        
        directory=os.listdir(pathImageDirectory)
        
        for file in directory:
            filepath=os.path.join(pathImageDirectory , file)
            
            self.listImagePaths.append(filepath)
            self.files.append(file)
            
            
        
            
    
       
        
        
    
    #-------------------------------------------------------------------------------- 
    
    def __getitem__(self, index):
        
        imagePath = self.listImagePaths[index]
        
        imageData = Image.open(imagePath).convert('RGB')
        
        filelist=self.files[index]
        
        if self.transform != None: imageData = self.transform(imageData)
        
        return imageData, filelist
        
    #-------------------------------------------------------------------------------- 
    
    def __len__(self):
        
        return len(self.listImagePaths)