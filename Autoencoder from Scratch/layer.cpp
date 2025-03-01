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
	else if (!neuronname.compare("Input2D"))
		type = Input2D;
	else if (!neuronname.compare("Pool2D"))
	{
		type = Pool2D;
		padding = num;
		return;
	}
	else
		type = Sigmoid;

	number = num;
	neurontype = neuronname;
	values = (double*)malloc(sizeof(double) * number);
	pre_activation_values = (double*)malloc(sizeof(double) * number);
}

Layer::Layer(int kernalnumber, int size,std::string layertype)
{
	type = Conv;
	neurontype = "Relu";
	kernelnumber = kernalnumber;
	kernelsize = size;
}

Layer::Layer(int kernalnumber, int size,int kerneldilation, std::string layertype)
{
	type = Conv;
	neurontype = "Relu";
	kernelnumber = kernalnumber;
	kernelsize = size;
	dilation = kerneldilation;
}