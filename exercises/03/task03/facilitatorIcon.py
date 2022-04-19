#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 19 11:39:36 2022

@author: Sascha Holzhauer, Uni Kassel
"""

from PIL import Image 
import numpy as np

im = Image.open('../../../abmodel/fire_evacuation/resources/human.png')
im = im.convert('RGBA')

data = np.array(im)   # "data" is a height x width x 4 numpy array
red, green, blue, alpha = data.T # Temporarily unpack the bands for readability

# Replace white with red... (leaves alpha values alone...)
white_areas = (red == 22) & (blue == 2) & (green == 11)
data[..., :-1][white_areas.T] = (255, 0, 0) # Transpose back needed

im2 = Image.fromarray(data)
im2.save('../../../abmodel/fire_evacuation/resources/facilitator.png')