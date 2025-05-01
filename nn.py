'''
Sicheng Zhong
Data 527: Perdictive Modeling
April 12th
Spring 2025
'''

import json
import os
import random
import numpy
import math


class Activate:
    @staticmethod
    def sigmoid(val):
        return 1 / (1 + math.exp(-1 * val))

    @staticmethod
    def reLu(val):
        if val <= 0:
            return 0
        else:
            return val
    
    @staticmethod
    def tanh(val):
        alpha = 1
        s = 1
        return alpha * ((numpy.e ** ( 2 * val * s ) - 1) / (numpy.e ** ( 2 * val * s ) + 1))

    @staticmethod
    def pd_activation(out, func):
        if func == Activate.reLu:
            if out < 0:
                return 0
            else:
                return 1
        elif func == Activate.sigmoid:
            return out * ( 1 - out )
        else:
            return 1.0

class Dense:
    def __init__(self, nodes, activation=None):
        '''
        Fullly Connectted Layer
        :param activation: function pointer
        '''
        super().__init__()
        self.nodes = nodes
        self.activation = activation



class FeedForwardNetwork:
    def __init__(self, numFeatures):
        self.RMSE = []
        self.val_loss = []
        # hidden layers
        self.layers = [
            Dense(32, Activate.sigmoid)  # 32
        ]
        self.weights = []  # hidden layers weights 2D (layers * weights in each layer)
        self.output_weights = []  # output weights
        self.hidden_output = []  # 2D (layers * node/output)
        self.output_activation = None

        for index, layer in enumerate(self.layers):
            layer_weight = []
            for i in range(layer.nodes):
                if index == 0:
                    # first layer
                    for i in range(numFeatures):
                        layer_weight.append(random.random())
                else:
                    # other layers
                    for i in range(self.layers[index - 1].nodes):
                        layer_weight.append(random.random())

            self.weights.append(layer_weight)
            self.hidden_output.append(list())

        for out_weight in range(self.layers[-1].nodes):
            self.output_weights.append(random.random())


    def train(self, inputs, targets,
              epochs=20, learning_rate = 0.01,
              training_type='batch',
              validation = None):

        for i in range(epochs):
            outputs = []

            if training_type == 'stochastic':
                indexID = random.randint(0, inputs.shape[0] - 1)
                outputs.append(self.__forward(inputs[indexID]))
                # calculate RMSE
                error = targets[indexID] - outputs[0]
                self.RMSE.append(math.sqrt(error ** 2))
                self.__backProp(inputs[indexID], outputs[0], error, learning_rate)

            elif training_type == 'batch':

                batch_error = 0
                for j, value in enumerate(inputs):
                    outputs.append(self.__forward(value))

                    # calculate MAE
                    error = targets[j] - outputs[j]

                    batch_error +=  error ** 2
                    self.__backProp(inputs[j], outputs[j], error, learning_rate)

                self.RMSE.append( math.sqrt(batch_error / inputs.shape[0] ))

            else:
                pass

            with open(f'NNTraining{epochs}{learning_rate}MSE.txt', 'a+', encoding='UTF-8') as fp:
                fp.write(f'Epoch: {i} Loss(RMSE): {self.RMSE[i]}\n')

            if validation and isinstance(validation, list) and len(validation) == 2 and len(validation[0]) == len(validation[1]):
                val_out = []

                error = 0
                for j in range(len(validation[0])):
                    val_out.append(self.__forward(validation[0][j]))
                    error += (validation[1][j] - val_out[j]) ** 2


                self.val_loss.append(math.sqrt(error / len(validation)))
            
            # if self.RMSE[i] >= self.RMSE[0] and i >= 10:
            #     # after 10 epoch and the RMSE goes rising up
            #     print('Training goes bad!!')
            #     return -1
        return 0


    def predict(self, inputs):
        predicts = []

        for i, value in enumerate(inputs):
            # set binary output
            predicts.append(self.__forward(value))

        return numpy.array(predicts)

    def getRMAE(self):
        return self.RMSE

    def getValLoss(self):
        return self.val_loss

    def __forward(self, input):
        for hidden in self.hidden_output:
            hidden.clear()

        # hidden layers
        for i, layer in enumerate(self.layers):
            if i == 0:
                # first layer
                for node in range(layer.nodes):
                    weightSum = 0
                    for j, item in enumerate(input):
                        weightSum += item * self.weights[i][node * len(input) + j]
                    self.hidden_output[i].append(self.layers[i].activation(weightSum))
            else:
                # other layers
                for node in range(layer.nodes):
                    weightSum = 0
                    for j, item in enumerate(self.hidden_output[i - 1]):
                        weightSum += item * self.weights[i][node * len(self.hidden_output[i - 1]) + j]
                    self.hidden_output[i].append(self.layers[i].activation(weightSum))

        # output layers
        weightSum_out = 0
        for j in range(self.layers[-1].nodes):
            weightSum_out += self.hidden_output[-1][j] * self.output_weights[j]

        if self.output_activation:
            return self.output_activation(weightSum_out)
        else:
            return weightSum_out  # no activation

    def __backProp(self, input, output, error, learning_rate):
        error_list = []
        for i in range(len(self.layers)):
            error_list.append(list())

        for i in range(self.layers[-1].nodes):
            pdErr = Activate.pd_activation(output, self.output_activation) * error
            deltaWeight = learning_rate * pdErr * self.hidden_output[-1][i]

            self.output_weights[i] += deltaWeight
            error_list[len(self.layers) - 1].append(pdErr)

        index = len(self.layers) - 1
        while index >= 0:
            current_weight_index = 0

            for j in range(self.layers[index].nodes):
                if index == 0:
                    # only one layer
                    pdErr = Activate.pd_activation(self.hidden_output[index][j], self.layers[index].activation) * self.output_weights[j] * error_list[index][j]

                    for k in range(len(input)):
                        deltaWeight = learning_rate * pdErr * input[k]
                        self.weights[index][current_weight_index] += deltaWeight

                        current_weight_index += 1
                else:
                    if index == 0:
                        # first layer
                        pdErr = Activate.pd_activation(self.hidden_output[index][j], self.layers[index].activation) * self.weights[index + 1][j] * error_list[index][j]

                        for k in range(len(input)):
                            deltaWeight = learning_rate * pdErr * input[k]
                            self.weights[index][current_weight_index] += deltaWeight
                            current_weight_index += 1
                    elif index == len(self.layers) - 1:
                        # last layer
                        pdErr = Activate.pd_activation(self.hidden_output[index][j], self.layers[index].activation) * self.output_weights[j] * error_list[index][j]

                        for k in range(self.layers[index - 1].nodes):
                            deltaWeight = learning_rate * pdErr * self.hidden_output[index - 1][k]
                            self.weights[index][current_weight_index] += deltaWeight
                            current_weight_index += 1
                    else:
                        pdErr = Activate.pd_activation(self.hidden_output[index][j], self.layers[index].activation) * self.weights[index + 1][j] * error_list[index][j]

                        for k in range(self.layers[index - 1].nodes):
                            deltaWeight = learning_rate * pdErr * self.hidden_output[index - 1][k]
                            self.weights[index][current_weight_index] += deltaWeight
                            current_weight_index += 1

            index -= 1