#pragma once
#include "layer.h"
#include <random>
#include <thread>

using namespace std;

class Network
{
public:

	//Deep Learning
	typedef enum gradtype {
		Stochastic,
		Batch,
		Mini_Batch,
	}gradtype;

	typedef enum losstype {
		Mean_Squared,
		Mean_Absolute,
		Mean_Biased,
		Root_Mean_Squared
	}losstype;

	typedef enum optimizer {
		No_Optimizer,
		Momentum,
		RMSProp,
		Adam,
	}optimizer;

	typedef enum weightinitializer {
		Glorot,
		He,
		Random
	}weightinitializer;

	typedef enum showparams {
		Visual,
		Text
	}showparams;

	gradtype gradient_descent_type;
	losstype model_loss_type;
	showparams displayparameters;

	optimizer Optimizer;
	weightinitializer WeightInitializer;

	vector<Layer> layers;

	double*** weights;
	double** biases;
	double*** momentum1D;
	double*** rmsp1D;
	double* errors;
	double* derrors;


	double alpha = 0.1;
	double momentumbeta = 0.9;
	double rmspropbeta = 0.999;
	double rmspropepsilon = 10e-10;

	int batchsize;
	int totalinputsize;
	int totalepochs;
	int batchnum=1;

	//Image
	double MatrixAverage(vector<vector<double>>* mat);
	vector<vector<double>> PixelDistances(vector<vector<double>>* mat1, vector<vector<double>>* mat2);
	vector<vector<vector<double>>> SobelEdgeDetection(vector<vector<vector<double>>>* images);
	vector<vector<vector<double>>> PrewittEdgeDetection(vector<vector<vector<double>>>* images);
	vector<vector<vector<double>>> NNInterpolation(vector<vector<vector<double>>>* image,int finalwidth,int finalheight);
	vector<vector<vector<double>>> BilinearInterpolation(vector<vector<vector<double>>>* image, int finalwidth, int finalheight);
	vector<vector<vector<double>>> EmptyUpscale(vector<vector<vector<double>>>* image, int finalwidth, int finalheight);

	//Network
	double Activation(double x,int i);
	double DActivation(double x, int i);
	void AddLayer(Layer l);
	void SetDisplayParameters(string s);
	void Summary();
	void PrintParameters();
	void Compile(string type,int batch_size);
	void Compile(string type);
	void Initialize();
	double WeightInitialization(int fan_in, int fan_out);
	void Train(vector<vector<double>> *inputs,vector<vector<double>> *actual,vector<vector<double>>* predicted,int epochs,string loss);
	void Train(vector<vector<vector<double>>>* inputs, vector<vector<double>>* actual, vector<vector<double>>* predicted, int epochs, string loss);
	void ShowTrainingStats(vector<vector<double>>* inputs, vector<vector<double>>* actual,int i);
	void ForwardPropogation(vector<vector<double>> sample, vector<double> actualvalue, vector<vector<double>>* predicted);
	void ErrorCalculation(vector<double> actualvalue);
	void AccumulateErrors();
	double DError(double predictedvalue, double actualvalue, int neuronnum);
	void CleanErrors();
	void BackPropogation();
	void LeakyReluParameters(double i, double a);
	void SetOptimizer(string opt);
	void SetInitializer(string init);
	vector<vector<double>> FullConvolve2D(vector<vector<double>>* input, vector<vector<double>>* kernel);
	vector<vector<double>> Convolve2D(vector<vector<double>> *input, vector<vector<double>> *kernel);
	vector<vector<double>> Rotate(vector<vector<double>>* input);
	vector<vector<double>> Zero2DMatrix(int x, int y);
	void AddVectors(vector<vector<double>>* v1, vector<vector<double>>* v2);
	void UpdateKernel(vector<vector<double>>* v1, vector<vector<double>>* v2, vector<vector<double>>* momentumkernel, vector<vector<double>>* rmspkernel);
	vector<vector<double>> InitializeKernel(int kernelsize, int dilation);
	vector<vector<double>> Relu2D(vector<vector<double>>* input);
	void MaxPooling2D(vector<vector<double>>* input, short int padnum, vector<vector<double>>* outputdest, vector<vector<double>>* chosendest);
	void CleanLayers();
};