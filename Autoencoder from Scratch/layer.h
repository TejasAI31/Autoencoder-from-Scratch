#pragma once
#include <iostream>
#include <vector>
using namespace std;


class Layer
{
public:

	typedef struct layerparams
	{
		double LeakyReluAlpha = 0.01;
	} layerparams;

	typedef enum layertype {
		Input2D,
		Input,
		Sigmoid,
		Relu,
		LeakyRelu,
		Tanh,
		Conv,
		Pool2D
	} layertype;


	vector<vector<vector<vector<double>>>> kernels;
	vector<vector<vector<double>>> pre_activation_values2D;
	vector<vector<vector<double>>> values2D;
	vector<vector<vector<double>>> values2Dderivative;
	vector<vector<vector<vector<double>>>> deltakernel;

	bool flattenweights = false;

	int padding;
	int kernelnumber;
	int kernelsize;
	int number;
	double* values;
	double* pre_activation_values;

	layerparams parameters;
	layertype type;
	std::string neurontype;

	Layer(int num,std::string neuronname);
	Layer(int kernalnumber, int size,std::string layertype);
};