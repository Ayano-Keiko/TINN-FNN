import numpy
import time
import pandas
import nn
import matplotlib.pyplot as plt
import argparse
from sklearn.model_selection import train_test_split
import random
import json
import data_prepocess

if __name__ == '__main__':

    random_state = 123
    train_size = 0.8

    numpy.random.seed(random_state)
    random.seed(random_state)

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--iteration', help='Neural Network iteration',
                        type=int, required=True)
    parser.add_argument('-l', '--learning_rate', help='Neural Network learning rate',
                        type=float, required=True)
    parser.add_argument('-g', '--train_type', help='Gradient Descent type',
                        choices=['stochastic', 'batch'], required=True)

    args = parser.parse_args()

    iteration = args.iteration
    learning_rate = args.learning_rate
    train_type = args.train_type
    print('Data Loading......')
    # data propocessing

    data = data_prepocess.feature_selection('./input_fl_12477')

    inputs, targets, scaler = data_prepocess.normalization(data)
    x_train, x_test, y_train, y_test = train_test_split(inputs, targets,
                                                        random_state=random_state,
                                                        train_size=train_size)

    num_features = inputs.shape[1]
    total_size = inputs.shape[0]


    print('model training......')
    net = nn.FeedForwardNetwork(num_features)
    start = time.time()
    net.train(x_train.values, y_train.values, validation=[x_test.values, y_test.values])
    finish = time.time()

    print(f'Program Running Time: {finish - start}s')

    loss = net.getRMAE()
    val_loss = net.getValLoss()

    print(f'Number of iterations: {len(loss)}\nactivation function: {getattr(net.layers[0].activation, "__name__", str(net.layers[0].activation))}\nlearning rate: {learning_rate}\ntraining type: {train_type}')

    # save parameter
    parameter = {
        'Learning rate': learning_rate,
        'iterations': len(loss),
        'Train type': train_type,
        'activation function': 'sigmoid',
        'loss': loss[-1]
    }
    with open('NNModelParameters.json', 'w', encoding='UTF-8') as fp:
        json.dump(parameter, fp)

    plt.figure()
    plt.title(f'RMSE epochs {iteration} lr {learning_rate} target pitch')
    plt.plot(range(1, len(loss) + 1), loss, color='#0000FF', label='train loss')
    plt.plot(range(1, len(val_loss) + 1), val_loss, color='#FF0000', label='test loss', linestyle='--')
    plt.legend()
    plt.savefig('RMSE')

    predicted = net.predict(x_test.values)
    idxs = [idx for idx, item in enumerate(predicted)]

    plt.figure(figsize=(64, 16))
    plt.title(f'Actual vs Predict\nepochs {iteration} lr {learning_rate} target pitch')
    plt.plot(range(len(predicted)), y_test.values, color='#FF0000', label='Actual')
    plt.plot(range(len(predicted)), predicted, color='#0000FF', label='Predict')
    plt.legend()
    plt.savefig('Demo')
