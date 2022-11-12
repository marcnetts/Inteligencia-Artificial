# -*- coding: utf-8 -*-
"""
Created on Wed Nov  9 21:09:39 2022

@author: marcnetts
"""

from linear_algebra import dot
import math, random

def step_function(x):
    return 1 if x >= 0 else 0

def perceptron_output(weights, bias, x):
    """invoca função degrau = step_function"""
    return step_function(dot(weights, x) + bias)

def sigmoid(t):
    return 1 / (1 + math.exp(-t))

def neuron_output(weights, inputs):
    return sigmoid(dot(weights, inputs))

def feed_forward(neural_network, input_vector):
    "manipula uma Rede Neural e retorna a saída do algoritmo forward-propagating a partir da entrada"
    
    outputs = [] 
    for layer in neural_network:
        input_with_bias = input_vector + [1]
        output = [neuron_output(neuron, input_with_bias)
              for neuron in layer]
        outputs.append(output)
        input_vector = output
    
    return outputs

alpha = 0.8
def backpropagate(network, input_vector, target): 
    hidden_outputs, outputs = feed_forward(network, input_vector)
    
    # the output *  (1 - output) is from the derivative of sigmoid
    output_deltas = [output * (1 - output) * (output - target[i]) * alpha
        for i, output in enumerate(outputs)]
    
    # adjust weights for output layer (network[-1])
    for i, output_neuron in enumerate(network[-1]):
        for j, hidden_output in enumerate(hidden_outputs + [1]):
            output_neuron[j] -= output_deltas[i] * hidden_output
    
    # back-propagate errors to hidden layer
    hidden_deltas = [hidden_output * (1 - hidden_output) *
        dot(output_deltas, [n[i] for n in network[-1]])
        for i, hidden_output in enumerate(hidden_outputs)]
    
    # adjust weights for hidden layer (network[0])
    for i, hidden_neuron in enumerate(network[0]):
        for j, input in enumerate(input_vector + [1]):
            hidden_neuron[j] -= hidden_deltas[i] * input

if __name__ == "__main__":
    raw_digits = [
        """11111
           1...1
           11111
           1...1
           1...1""",
           
        """11111
           1....
           11111
           1....
           11111""",
           
        """..1..
           ..1..
           ..1..
           ..1..
           ..1..""",
           
        """11111
           1...1
           1...1
           1...1
           11111""",
           
        """1...1
           1...1
           1...1
           1...1
           11111"""
        ]
    
    def make_digit(raw_digit):
        return [1 if c == '1' else 0
            for row in raw_digit.split("\n")
            for c in row.strip()]
        
    inputs = list(map(make_digit, raw_digits))
    targets = [[1 if i == j else 0 for i in range(10)] for j in range(10)]
        
    random.seed(0)
    input_size = 25 # dimensão dos vetores relacionados às 10 entradas
    num_hidden = 5 # quantidade de neurõnios na camada intermediária
    output_size = 5 # 10 saídas, cada uma relacionada à uma entrada
    
    hidden_layer = [[random.random() for __ in range(input_size + 1)]
        for __ in range(num_hidden)]
    
    output_layer = [[random.random() for __ in range(num_hidden + 1)]
        for __ in range(output_size)]
    
    network = [hidden_layer, output_layer] 

    # 10000 CICLOS DE TREINAMENTO
    for __ in range(10000):
        for input_vector, target_vector in zip(inputs, targets):
            backpropagate(network, input_vector, target_vector)
    
    def predict(input):
        return feed_forward(network, input)[-1]
    
    # TESTE DAS ENTRADAS TREINADAS
    for i, input in enumerate(inputs):
        outputs = predict(input)
        print(i, [round(p,2) for p in outputs]) #resultado dos neuronios de saida c 2 casas decimais
       
    # TESTE DE DÍGITOS NUMÉRICOS QUE NÃO FORAM TREINADOS
    print(
"""11111
1...1
11111
1...1
1...1""")
    print([round(x, 2) for x in
           predict( [1,1,1,1,1,
                     1,0,0,0,1,
                     1,1,1,1,1,
                     1,0,0,0,1,
                     1,0,0,0,0])])
    
    print(
"""11111
1....
11111
1....
11111""")
    print([round(x, 2) for x in
           predict( [1,1,1,1,1,
                     1,0,0,0,0,
                     1,1,1,1,1,
                     1,0,0,0,0,
                     1,1,1,1,1])])
    
    print(
"""..1..
..1..
..1..
..1..
..1..""")
    print([round(x, 2) for x in
           predict( [0,0,1,0,0,
                     0,0,0,0,0,
                     0,0,1,0,0,
                     0,0,1,0,0,
                     0,0,1,0,0])])
    
    print(
"""11111
1...1
1...1
1...1
11111""")
    print([round(x, 2) for x in
           predict( [1,1,1,1,1,
                     1,0,0,0,1,
                     1,0,0,0,0,
                     1,0,0,0,1,
                     1,1,1,1,1])])
    
    print(
"""1...1
1...1
1...1
1...1
11111""")
    print([round(x, 2) for x in
           predict( [1,0,0,0,1,
                     1,0,0,0,1,
                     1,0,0,0,0,
                     1,0,0,0,1,
                     1,1,1,1,1])])
    
    print("VARIAÇÔES")
    
    print(
""".111.
1...1
11111
1...1
1...1""")
    print([round(x, 2) for x in
           predict( [0,1,1,1,0,
                     1,0,0,0,1,
                     1,1,1,1,1,
                     1,0,0,0,1,
                     0,0,0,0,0])])
    
    print(
""".111.
..1..
..1.
..1..
.111.""")
    print([round(x, 2) for x in
           predict( [0,1,1,1,0,
                     0,0,1,0,0,
                     0,0,1,0,0,
                     0,0,1,0,0,
                     0,1,1,1,0])])
    
    print(
""".111.
1...1
1...1
1...1
.111.""")
    print([round(x, 2) for x in
           predict( [0,1,1,1,0,
                     1,0,0,0,1,
                     1,0,0,0,0,
                     1,0,0,0,1,
                     0,1,1,1,0])])

    