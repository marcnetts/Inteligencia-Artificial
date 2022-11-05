# -*- coding: utf-8 -*-
"""
Created on Wed Oct 26 21:08:29 2022

@author: marcnetts
"""

from __future__ import division
from collections import Counter
from functools import partial
from linear_algebra import dot
import matplotlib.pyplot as plt
import numpy as np
import math, random

def sigmoid(t):
    return ((2 / (1 + math.exp(-t))) - 1)

def neuron_output(weights, inputs):
    return sigmoid(dot(weights, inputs))

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

alpha = 0.08
def backpropagate(network, input_vector, target):
    # feed _foward calcula a saida dos neurônios usando sigmóide
    # não a entrada
    hidden_outputs, outputs = feed_forward(network, input_vector)
    # 0.5 alpha... 
    # (output - target[i]) = erro
    # alpha é taxa de aprendizagem
    output_deltas = [0.5 * (1 + output) * (1 - output) * (output - target[i]) * alpha for i, output in enumerate(outputs)]
    
    # ajuste dos pesos simplicos para camada de saída (network[-1])
    for i, output_neuron in enumerate(network[-1]):
        for j, hidden_output in enumerate(hidden_outputs + [1]):
            output_neuron[j] -= output_deltas[i] * hidden_output
            
    # hidden_deltas = [0.5 * alpha * (1 + output) * (1 - output)] calculo da derivada da sign5ide
    # retro-propagacio do erro para camadas intermediarias
    print(hidden_outputs)
    hidden_deltas = [0.5 * alpha * (1 + hidden_output) * (1 - hidden_output) * dot(output_deltas, [n[i] for n in network[-1]]) for i, hidden_output in enumerate(hidden_outputs)]
    
    # ajuste dos pesos sinipticos para camadas intermediarias netmork[0])
    for i, hidden_neuron in enumerate(network[0]):
        for j, input in enumerate(input_vector + [1]):
            hidden_neuron[j] -= hidden_deltas[i] * input 

def seno(x): # funcao a ser aproximada pela Rede Neural
    seno = [(math.sin(math.pi/180*x)*math.sin(2*math.pi/180*x))] # seno é uma lista
    # [(0.8+(math.sin(math.pi/180*x)*math.sin(2*math.pi/180*x)))*0.5)
    return [seno]

def predict(inputs):
    return feed_forward(network, inputs)[-1]

inputs = []
targets = []
for x in range(360):
    seno_a = seno(x) 
                                                                                                                                                                                      
random.seed(0) # valores iniciais de pesos sinápticos
input_size = 1 # dimensão do vetor de entrada
num_hidden = 6 # numero de neuronios na camada intermediaria
output_size = 1 # dimensão da camada de saida = 1 neuronio 

"""
inserindo manualmente os vetores relativos à camada intermediária e à saída da Rede Neural
hidden_layer = ((-0.085, -0.091, (-0.033, -0.08J, (-0.074, 4.063], (-0.075, -0.0651, (-0.088, •0.076], (•0.077, •0.072)
output Layer = ((0.082, -0.09, 0.064, -0.08, 0.084, -0.075, 0.099)) "' 
"""

# cada neuronio da camada intermediaria tem um peso sinaptico associado à entrdada
# e adicionando o peso do bias
hidden_layer = [[random.random() for __ in range(input_size + 1)] for __ in range(num_hidden)] 

# print(hidden_layer)
# neuronio de saida tem um peso sineptic sociado a cada neuronio da camada intermediiria
# e adicionando o peso do bias
output_layer = [[random.random() for __ in range(num_hidden + 1)] for __ in range(output_size)]
# a rede inicializa com pesos sinapticos randomicos

network = [hidden_layer, output_layer]
#print(network) 

for __ in range(300): #numero de ciclos da treinamento
    for x in range(360):
        inputs = seno(x)
        targets = seno(x)
        for input_vector, target_vector in zip(inputs, targets):
            backpropagate(network, input_vector, target_vector)
#TREINANENTO DA REDE NEURAL 

#formação do gráfico
fig,ax = plt.subplots()
ax.set(xlabel='ângulo (º)', ylabel='função sen(x)*sen(2x)', title='Aproximocao Funcional')
ax.grid()
t = np.arange(0, 360, 1) 

# teste da rede atraves de predict( )
saida = []
for x in range(360):
    inputs = seno(x)
    targets = seno(x)
    for input_vector, target_vector in zip(inputs, targets):
        sinal_saida = predict(input_vector)
        saida.extend(sinal_saida) 

entrada = []
for x in range(360):
    entrada += seno(x) # criando o arranjo da função de entrada para o graficp
ax.plot(t, entrada)
ax.plot(t, saida)
plt.show() 

print ("camada entrodo", hidden_layer)
print ("comada saida", output_layer) 

