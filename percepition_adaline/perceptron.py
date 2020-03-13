import numpy as np
from activation_functions import heaviside_step_function

class Perceptron():
    
    def __init__(self, input_size, act_func=heaviside_step_function, epochs=100, learning_rate=0.0025):
        self.act_func = act_func
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.weights = np.random.rand(input_size + 1) 
        
    
    def predict(self, inputs):
        inputs = np.append(-1, inputs)
        u = np.dot(inputs, self.weights)
        return self.act_func(u)
    
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
        error = True
        #print(f'Pesos Atuais:\n {self.weights}')
        #print(*self.weights, sep='\n')
        for e in range(self.epochs):
            error = False
            print(f'>>> Iniciar Época - {e + 1} <<<\n')
            print(f'Pesos Atuais:\n {self.weights}')
            for inputs, label in zip(training_inputs, labels):
                print(f'Entrada:\n {inputs}')
                predicton = self.predict(inputs)
                if predicton != label:
                    print(f'\nERRO >> Esperado: {label}, Obtido: [{predicton}]\n\nIniciar Treinamento!')
                    inputs = np.append(-1, inputs)
                    self.weights = self.weights + self.learning_rate * (label - predicton) * inputs
                    pesos = self.weights
                    print(f'Novos Pesos:\n {pesos}')
                    error = True
                    break
                else:
                    print(f'Tudo OK!\n\n')
        #print(f'Novos Pesos:\n {pesos}')
        #print('\nnovos pesos')
        #print(*pesos, sep='\n')         
            print('')
            if not error:
                break
            
            
