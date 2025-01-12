#include "layer.h"

Layer::Layer(int num,std::string neuronname)
{

	if (!neuronname.compare("Sigmoid"))
		type = Sigmoid;
	else if (!neuronname.compare("Relu"))
		type = Relu;
	else if (!neuronname.compare("LRelu"))
		type = LeakyRelu;
	else if (!neuronname.compare("Tanh"))
		type = Tanh;
	else if (!neuronname.compare("Input"))
		type = Input;
	else
		type = Sigmoid;

	number = num;
	neurontype = neuronname;
	values = (double*)malloc(sizeof(double) * number);
	pre_activation_values = (double*)malloc(sizeof(double) * number);
}