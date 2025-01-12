#pragma once
#include <iostream>

class Layer
{
public:

	typedef struct layerparams
	{
		double LeakyReluAlpha = 0.01;
	} layerparams;

	typedef enum layertype {
		Input,
		Sigmoid,
		Relu,
		LeakyRelu,
		Tanh,
	} layertype;

	int number;
	double* values;
	double* pre_activation_values;

	layerparams parameters;
	layertype type;
	std::string neurontype;

	Layer(int num,std::string neuronname);
};