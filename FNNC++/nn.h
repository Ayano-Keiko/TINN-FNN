#pragma once
#include <vector>
#include <cmath>
#include <array>

namespace FNN {
	class FullConnectLayer;
	class NeuralNetwork;

	class NeuralNetwork {
		/*
		Class of Neural Network
		Full Connectted
		*/
	private:
		const size_t number_features;  // number of features in input
		const size_t class_number;  // number of output nodes ( the classification class number )

		std::vector<FullConnectLayer> layers;  // layers list
		
		//int number_of_weights{};  // number of weights
		//int number_out_weights{}; // number of output weights
		std::vector<std::vector<double>> hidden_outputs; // all output of each hidden layers -- 2D (number of layer * output in each layer)

		std::vector<std::vector<double>> weights;  // all weights in hidden layers
		std::vector<double> output_weights; // output weights
		std::vector<double> bias; // all bias

		// RMSE
		std::vector<double> RMSE{};

		double (*output_activation) (double);
		double forward(const std::vector<double>&);
		void backPropogation(const std::vector<double>&, double, double, double);
	public:
		NeuralNetwork(size_t);
		~NeuralNetwork();
		
		int train(const std::vector<std::vector<double>>& inputs, const std::vector<double>& tergets,
			int epochs = 20, double learning_rate = 0.01);
		std::vector<double> predict(const std::vector<std::vector<double>>&);

		std::vector<double> getRMSE();
	};

	class FullConnectLayer {
	public:
			const size_t nodes;
			double (*activation) (double);
		
			FullConnectLayer(size_t, double (*) (double));
	};

	
}

namespace activation {
	inline double sigmoid(double value) {
		/*
		sigmoid function
		*/
		return 1 / (1 + std::exp(value * -1));
	}

	inline double relu(double value) {
		if (value < 0)
		{
			return 0;
		}
		else
		{
			return value;
		}
	}

	inline double tanh(double value)
	{
		return (std::exp(value) - std::exp(-value)) / (std::exp(value) + std::exp(-value));
	}

	inline double pdActivation(double value, double (* activation)(double))
	{
		/*
		partial derivative of activation function
		*/
		if (activation == relu)
		{
			// partial derivative of relu
			// x = 0 is undefined and by convention partial derivative is 0 when x = 0
			if (value <= 0)
			{				
				return 0;
			}
			else
			{
				return 1;
			}
		}
		else if (activation == sigmoid)
		{
			// partial derivative of sigmoid
			return value * (1 - value);
		}
		else if (activation == tanh)
		{
			return 1 - std::pow((std::exp(value) - std::exp(-value)) / (std::exp(value) + std::exp(-value)), 2);
		}
		else if (activation == NULL)
		{
			// null activation
			return 1;
		}
		else
		{
			return 0.0;
		}
	}
}