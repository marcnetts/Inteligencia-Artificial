# -*- coding: utf-8 -*-
"""
Created on Wed Oct 26 21:08:29 2022

@author: marcnetts
"""

from __future__ import division
from linear_algebra import dot
import matplotlib.pyplot as plt
import numpy as np
import math, random

def sigmoid(t):
    return 1/(1 + math.exp(-t))
    #return ((2 / (1 + math.exp(-t))) - 1)

def neuron_output(weights, inputs):
    return sigmoid(dot(weights, inputs))

def feed_forward(neural_network, input_vector):
    outputs = []

    for layer in neural_network:
        input_with_bias = input_vector + [1]
        output = [neuron_output(neuron, input_with_bias)
                  for neuron in layer]
        outputs.append(output)
        input_vector = output

    return outputs

alpha = 0.08
def backpropagate(network, input_vector, target):
    # feed _foward calcula a saida dos neurônios usando sigmóide
    # não a entrada
    hidden_outputs, outputs = feed_forward(network, input_vector)
    # (output - target[i]) = erro
    # alpha é taxa de aprendizagem
    #output_deltas = [0.5 * (1 + output) * (1 - output)
    #                 * (output - target[i]) * alpha
    #                 for i, output in enumerate(outputs)]
    output_deltas = [output * (1 - output)
                     * (output - target[i]) * alpha
                     for i, output in enumerate(outputs)]

    # ajuste dos pesos sinapticos para camada de saída (network[-1])
    for i, output_neuron in enumerate(network[-1]):
        for j, hidden_output in enumerate(hidden_outputs + [1]):
            output_neuron[j] -= output_deltas[i] * hidden_output

    # hidden_deltas = [0.5 * alpha * (1 + output) * (1 - output)] calculo da derivada da signoide
    # retro-propagacio do erro para camadas intermediarias
    print(hidden_outputs)
    hidden_deltas = [hidden_output * (1 - hidden_output) * alpha *
                     dot(output_deltas, 
                     [n[i] for n in network[-1]])
                     for i, hidden_output in enumerate(hidden_outputs)]
    #hidden_deltas = [0.5 * alpha * (1 + hidden_output)
    #                 * (1 - hidden_output) * dot(output_deltas, 
    #                 [n[i] for n in network[-1]])
    #                 for i, hidden_output in enumerate(hidden_outputs)]

    # ajuste dos pesos sinapticos para camadas intermediarias netmork[0])
    for i, hidden_neuron in enumerate(network[0]):
        for j, input in enumerate(input_vector + [1]):
            hidden_neuron[j] -= hidden_deltas[i] * input

def seno(x): # funcao a ser aproximada pela Rede Neural
    #seno = [(math.sin(math.pi/180*x)*math.sin(2*math.pi/180*x))] # seno é uma lista
    seno = [(0.8+(math.sin(math.pi/180*x)*math.cos(2*math.pi/180*x)))*0.5]
    return [seno]

def predict(inputs):
    return feed_forward(network, inputs)[-1]

inputs = []
targets = []

random.seed(0) # valores iniciais de pesos sinápticos
input_size = 1 # dimensão do vetor de entrada
num_hidden = 6 # numero de neuronios na camada intermediaria
output_size = 1 # dimensão da camada de saida = 1 neuronio

"""
inserindo manualmente os vetores relativos à camada intermediária e à saída da Rede Neural
hidden_layer = ((-0.085, -0.091, (-0.033, -0.08J, (-0.074, 4.063], (-0.075, -0.0651, (-0.088, •0.076], (•0.077, •0.072)
output Layer = ((0.082, -0.09, 0.064, -0.08, 0.084, -0.075, 0.099)) "'
"""

hidden_layer = [[random.random() for __ in range(input_size + 1)]
    for __ in range(num_hidden)]

output_layer = [[random.random() for __ in range(num_hidden + 1)]
    for __ in range(output_size)]
# a rede inicializa com pesos sinapticos randomicos

network = [hidden_layer, output_layer]

#TREINANENTO DA REDE NEURAL
for __ in range(300): #numero de ciclos da treinamento
    for x in range(360):
        inputs = seno(x)
        targets = seno(x)
        for input_vector, target_vector in zip(inputs, targets):
            backpropagate(network, input_vector, target_vector)

#formação do gráfico
fig,ax = plt.subplots()
ax.set(xlabel='ângulo (º)',
       ylabel='função sen(x)*sen(2x)',
       title='Aproximacao Funcional')
ax.grid()

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
    entrada += seno(x) # criando o arranjo da função de entrada para o grafico
t = np.arange(0, 360, 1)
ax.plot(t, entrada)
ax.plot(t, saida)
plt.show()

print ("camada entrada", hidden_layer)
print ("comada saida", output_layer)
