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
		Softmax,
		Conv,
		Pool2D,
		Dropout,
	} layertype;


	vector<vector<vector<vector<double>>>> kernels;
	vector<vector<vector<double>>> pre_activation_values2D;
	vector<vector<vector<double>>> values2D;
	vector<vector<vector<double>>> values2Dderivative;
	vector<vector<vector<vector<double>>>> deltakernel;
	vector<vector<vector<vector<double>>>> momentum2D;
	vector<vector<vector<vector<double>>>> rmsp2D;


	bool flattenweights = false;

	int dilation = 1;
	int padding=0;
	int kernelnumber=0;
	int kernelsize=0;
	int number=0;
	
	double softmaxsum = 0;
	double dropout = 0.0;
	double* values;
	double* pre_activation_values;

	layerparams parameters;
	layertype type;
	std::string neurontype;

	Layer(int num,std::string neuronname);
	Layer(int kernalnumber, int size,std::string layertype);
	Layer(int kernalnumber, int size, int dilation, std::string layertype);
	Layer(double drop,std::string layertype);
};