import numpy as np
from activation_functions import heaviside_step_function

class Adalinetext():
    
    def __init__(self, input_size, act_func=heaviside_step_function, epochs=1000, learning_rate=0.0025, precisao=0.0001):
        self.act_func = act_func
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.weights = np.random.rand(input_size + 1)
        self.precisao = precisao
        self.erro = []
        
    def predict(self, inputs):
        inputs = np.append(-1, inputs)
        u = np.dot(inputs, self.weights)
        return u
    
    def erro(self):
        return self.erro
    
    def pesos(self):
        return self.weights
    
    def EQM(self, inputs, labels):
        EQM = 0
        for input, label in zip(inputs, labels):
            u = self.predict(input)
            EQM += (label - u)**2
        EQM = EQM/len(labels)
        return EQM
    
    def matriz_conf(self, training_inputs, labels):
        matcon=[0,0,0,0]
        for inputs, label in zip(training_inputs, labels):
            prediction = self.act_func(self.predict(inputs))
            if (prediction == label and label == 0):
                #print('verdadeiro negativo')
                matcon[0] += 1
            if prediction != label and label == 0:
                #print('falso negativo')
                matcon[1] += 1
            if prediction == label and label == 1:
                #print('verdadeiro positivo')
                matcon[2] += 1
            if prediction != label and label == 1:
                #print('falso positivo')
                matcon[3] += 1
        print(matcon)
        return matcon
        
    
    def train(self, training_inputs, labels):
        EQManterior = 0
        EQMatual = 0
        EQM = []
        #print(f'Actual weights:\n {self.weights}')
        for e in range(self.epochs):
            print(f'>>> Start epoch {e + 1}')
            #print(f'Actual weights {self.weights}')
            EQManterior = self.EQM(training_inputs, labels)
            for inputs, label in zip(training_inputs, labels):
                #print(f'Input {inputs}')
                predicton = self.predict(inputs)
                #print({self.act_func(predicton)})
                inputs = np.append(-1, inputs)
                self.weights += self.learning_rate * (label - predicton) * inputs
                #print(f'New weights {self.weights}')
            EQMatual = self.EQM(training_inputs, labels)
            #print(EQManterior)
            print(abs(EQMatual- EQManterior))
            EQM.append(abs(EQMatual-EQManterior))
            #self.erro.append(abs(EQMatual - EQManterior))
            if abs(EQMatual - EQManterior) <= self.precisao:
                #print(f'New weights {self.weights}')
                #print(EQMatual)
                break
        #print(f'New weights:\n {self.weights}')
        return EQM
            
    def operacao(predicton):
        if predicton == 0:
            print(f'Input {inputs}')
            print(f'É 0')
            return 0
        else:
            print(f'Input {inputs}')
            print(f'É 1')
            return 1
        
