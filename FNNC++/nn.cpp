#include "nn.h"
#include <cstdlib>
#include <cstdio>
#include <cstring>

namespace FNN {
	NeuralNetwork::NeuralNetwork(size_t number_features) :
		number_features { number_features },  // input features
		class_number{ 1 }  // output nodes number
	{
		// initialize layer -- size should be equal to number of hidden layers number + number of output layers (1)
		// a layer list
		this->layers.push_back(FullConnectLayer(16LL, *activation::sigmoid));
		
		
		for (size_t i = 0; i < this->layers.size(); ++i) {

			if (i == 0) {
				// first hidden layer
				this->weights.push_back(std::vector<double>(number_features * this->layers[i].nodes));
			}
			else {
				// other layers
				this->weights.push_back(std::vector<double>(this->layers[i].nodes * this->layers[i - 1].nodes));

			}
			
			// initial weights
			for (size_t j = 0; j < this->weights[i].size(); ++j)
			{
				this->weights[i][j] = (double)std::rand() / (double)RAND_MAX;
			}
			// create hidden outout of each layer and filled with 0
			this->hidden_outputs.push_back(std::vector<double>(this->layers[i].nodes, 0.0));
			
		}

		// bias
		this->bias = std::vector<double>(this->layers.size() + 1);

		// output weights
		this->output_weights = std::vector<double>(this->layers[this->layers.size() - 1].nodes);
		for (size_t i = 0; i < this->output_weights.size(); ++i) {
			this->output_weights[i] = (double)std::rand() / (double)RAND_MAX;
		}
		

		// give each item of bias random value
		for (size_t i = 0; i < this->layers.size() + 1; ++i) {
			this->bias[i] = (double)std::rand() / (double)RAND_MAX;
			
		}
			

		// output weights
		output_activation = NULL;
		
	}

	NeuralNetwork::~NeuralNetwork() {
		
	}

	int NeuralNetwork::train(const std::vector<std::vector<double>>& inputs, const std::vector<double>& targets,
		int epochs, double learning_rate)
	{
		/*
		The training algorithm
		inputs --> input values (input features 4096 (64 * 64) )
		targets --> targets (eg. 15 class for Chinese Minist)
		training and update weights via Gradient Descent
		iteration --> total iterations
		learning_rate --> learning rate/ initialized learning rate
		return value: int, the status code ( 0 --> success, -1 --> error )
		*/
		// Learning rate annealing is essentially a strategy to reduce the learning rate as training progresses -- used to update learning rate
		// mse
		
		
		if (inputs.size() != targets.size()) {
			std::printf("Inputs should be same size as Targets!!");
			return -1;  // fail
		}

		// const double anneal = 0.99f;
		
		for (int i = 0; i < epochs; ++i)
		{
			double loss_sum = 0.0;

			
			for (size_t j = 0; j < inputs.size(); ++j)
			{
				// forward prop
				double predicted = this->forward(inputs[j]);
				// this->final_output.push_back(predicted);

				// predicted - actual
				loss_sum += std::pow(predicted - targets[j], 2);

				// backprop
				this->backPropogation(inputs[j], predicted, targets[j], learning_rate);

			}
			loss_sum /= (double)inputs.size();
				

			// update learn rate
			// learning_rate *= anneal;

			this->RMSE.push_back(std::sqrt(loss_sum));
		}

		return 0;
	}

	std::vector<double> NeuralNetwork::predict(const std::vector<std::vector<double>>& inputs)
	{
		std::vector<double> output{};

		for (size_t i = 0; i < inputs.size(); ++i)
		{
			output.push_back(this->forward(inputs[i]));
		}
		
		return output;
	}

	double NeuralNetwork::forward(const std::vector<double>& input)
	{

		// from first hidden layer to the last hidden layer
		for (size_t i = 0; i < this->layers.size(); ++i) {
			// current index
			int current_index = 0;

			// Calculate hidden layer node values.
			for (int j = 0; j < this->layers[i].nodes; ++j) {
				double sum = 0.0;

				if (i == 0)
				{
					// first layer
					for (size_t k = 0; k < input.size(); ++k)
					{
						sum += input[k] * this->weights.at(i).at(current_index);
						++current_index;  // incrment the current wieght index
					}

					// save hidden output value
					this->hidden_outputs[i][j] = this->layers[i].activation(sum + this->bias[i]);
					
				}
				else
				{

					// remain layer
					for (int k = 0; k < this->hidden_outputs[i - 1].size(); ++k)
					{
						sum += this->hidden_outputs[i - 1][k] * this->weights.at(i).at(current_index);
						++current_index;  // incrment the current wieght index
					}

					// save hidden output value
					this->hidden_outputs[i][j] = this->layers[i].activation(sum + this->bias[i]);
					
				}
			}

		}

		// out weight index
		// Calculate output layer neuron values.
		
		double out_sum = 0.0;
			
		for (size_t i = 0, out_weights_idx = 0; i < this->hidden_outputs[this->hidden_outputs.size() - 1].size(); ++i, ++out_weights_idx)
		{
			// because we clear each hidden out before enter each hidden layer except first, so the vector save the last hidden layer output
			out_sum += this->hidden_outputs[this->hidden_outputs.size() - 1][i] * this->output_weights.at(out_weights_idx);
		}

		
		if (this->output_activation != NULL)
		{
			double final_value = this->output_activation(out_sum + this->bias[this->layers.size()]);

			return final_value;
		}
		else
		{
			double final_value = out_sum + this->bias[this->layers.size()];

			return final_value;
		}
	}

	void NeuralNetwork::backPropogation(const std::vector<double>& inputs, double predicted, double target, double learning_rate) {
		std::vector<std::vector<double> > errors(this->layers.size());

		// update output layer neuron weights
		// weight_idx --> current weight index of output layer
		
		for (int j = 0, weight_idx = 0; j < this->layers[this->layers.size() - 1].nodes; ++j, ++weight_idx) {
			// partial derivative of error function
			
			double err =  activation::pdActivation(predicted, this->output_activation) * (target - predicted);
			
			
			double deltaWeight = learning_rate * err * this->hidden_outputs[this->layers.size() - 1][j];
			this->output_weights[j] += deltaWeight;

			errors[this->layers.size() - 1].push_back(err);
			
		}
		
		
		// from first hidden layer to the last hidden layer
		for (int i = this->layers.size() - 1; i >= 0; --i) {
			// current wieght index
			int curr_wieght_idx = 0;
			
			// Calculate hidden layer node weights			
			for (int j = 0; j < this->layers[i].nodes; ++j) {
				if (this->layers.size() == 1)
				{
					// only one layer
					double err = activation::pdActivation(this->hidden_outputs[i][j], this->layers[i].activation) * this->output_weights[j] * errors[i][j];
					
					for (int k = 0; k < this->number_features; ++k)
					{
						// update weight
						double deltaWeight = learning_rate * err * inputs[k];

						this->weights[i][curr_wieght_idx++] += deltaWeight;
					}
				}
				else {

					if (i == 0)
					{
						// first layer
						double err = activation::pdActivation(this->hidden_outputs[i][j], this->layers[i].activation) * this->weights[i + 1][j] * errors[i][j];

						for (int k = 0; k < this->number_features; ++k)
						{
							// update weight
							double deltaWeight = learning_rate * err * inputs[k];

							this->weights[i][curr_wieght_idx++] += deltaWeight;
						}

					}
					else if (i == this->layers.size() - 1)
					{


						// last layer					
						double err = activation::pdActivation(this->hidden_outputs[i][j], this->layers[i].activation) * this->output_weights[j] * errors[i][j];

						errors[i - 1].push_back(err);


						for (int k = 0; k < this->layers[i - 1].nodes; ++k)
						{
							// update weight

							double deltaWeight = learning_rate * err * this->hidden_outputs[i - 1][k];

							this->weights[i][curr_wieght_idx++] += deltaWeight;
						}

					}
					else
					{
						// others
						double err = activation::pdActivation(this->hidden_outputs[i][j], this->layers[i].activation) * this->weights[i + 1][j] * errors[i][j];
						errors[i - 1].push_back(err);

						for (int k = 0; k < this->layers[i - 1].nodes; ++k)
						{
							// update weight
							double deltaWeight = learning_rate * err * this->hidden_outputs[i - 1][k];

							this->weights[i][curr_wieght_idx++] += deltaWeight;
						}


					}
				}
			}
			
		}
		
	}

	FullConnectLayer::FullConnectLayer(size_t nodes, double (*activation) (double)) :
		nodes {nodes}, activation {activation}
	{

	}

	std::vector<double> NeuralNetwork::getRMSE()
	{
		return this->RMSE;
	}
}