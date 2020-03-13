import numpy as np
from activation_functions import heaviside_step_function

class Adaline():
    
    def __init__(self, input_size, act_func=heaviside_step_function, epochs=100, learning_rate=0.0025, precisao=0.000001):
        self.act_func = act_func
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.weights = np.random.rand(input_size + 1)
        self.precisao = precisao
        
    def predict(self, inputs):
        inputs = np.append(-1, inputs)
        u = np.dot(inputs, self.weights)
        return self.act_func(u)
    
    def EQM(self, training_inputs, labels):
        EQM = 0
        p = len(labels)
        for inputs, label in zip(training_inputs, labels):
            u = self.predict(inputs)
            inputs = np.append(-1, inputs)
            EQM = EQM + (label - self.act_func(u))**2
        EQM = EQM/p
        return EQM
    
    def matriz_conf(self, training_inputs, labels):
        matcon=[0,0,0,0]
        for inputs, label in zip(training_inputs, labels):
            prediction = self.predict(inputs)
            if (prediction == label and label == 0):
                print('verdadeiro negativo')
                matcon[0] += 1
            if prediction != label and label == 0:
                print('falso negativo')
                matcon[1] += 1
            if prediction == label and label == 1:
                print('verdadeiro positivo')
                matcon[2] += 1
            if prediction != label and label == 1:
                print('falso positivo')
                matcon[3] += 1
        print(matcon)
        return matcon
    
    def train(self, training_inputs, labels):
        EQManterior = 0
        EQMatual = 0
        for e in range(self.epochs):
            print(f'>>> Start epoch {e + 1}')
            print(f'Actual weights {self.weights}')
            EQManterior = self.EQM(training_inputs, labels)
            for inputs, label in zip(training_inputs, labels):
                print(f'Input {inputs}')
                predicton = self.predict(inputs)
                inputs = np.append(-1, inputs)
                self.weights = self.weights + self.learning_rate * (label - predicton) * inputs
                print(f'New weights {self.weights}')
            EQMatual = self.EQM(training_inputs, labels)
            if abs(EQMatual - EQManterior) <= self.precisao:
                    break
            if e == self.epochs:
                break
            
            
    def operacao(self, inputs, labels):
        predicton = self.act_func(self.predict(inputs))
        if predicton == 0:
            print(f'Input {inputs}')
            print(f'É Rocha')
        else:
            print(f'Input {inputs}')
            print(f'É Mina')
        
        
# -*- coding: utf-8 -*-

