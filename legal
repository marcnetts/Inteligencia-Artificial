# -*- coding: utf-8 -*-
"""
Created on Wed Oct 19 20:54:04 2022

@author: Akira
"""

from __future__ import division
from collections import Counter
from functools import partial
from linear_algebra import dot
import math, random

def step_function(x):
    return 1 if x >= 0 else 0

def neuron_output (weights, inputs):
    calculation = dot(weights, inputs) 
    return step_function(calculation)

def feed_forward(neural_network, input_vector):
    
    outputs = []
    
    for layer in neural_network:
        
        input_with_bias = input_vector + [1]                 # add a bias input
        output = [neuron_output(neuron, input_with_bias)    # compute the output 
                  for neuron in layer]                      # for this layer 
        outputs.append(output)                              # and remember it
        
        # the input to the next layer is the output of this one
        input_vector = output
    return outputs

xor_network = [# camada oculta
               [[1, 1, -1.5],
                [1, 1, -0.5]],
               # output layer
               [[-2, 1, -0.5]]]

for x in [0, 1]:
    for y in [0, 1]: 
        print (x,"EXCLUSIVO",y," = ",feed_forward(xor_network, [x, y])[-1])

