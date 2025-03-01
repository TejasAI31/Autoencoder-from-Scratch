#pragma once
#include "layer.h"
#include <vector>
#include <thread>

using namespace std;

class Network
{
public:
	typedef enum gradtype {
		Stochastic,
		Batch,
		Mini_Batch,
	}gradtype;

	typedef enum losstype {
		Mean_Squared,

	}losstype;

	typedef enum showparams {
		None,
		Visual,
		Text
	}showparams;

	gradtype gradient_descent_type;
	losstype model_loss_type;
	showparams displayparameters;

	vector<Layer> layers;

	double*** weights;
	double* errors;
	double* derrors;
	double alpha = 0.1;

	int batchsize;
	int totalinputsize;
	int totalepochs;
	int batchnum=1;

	double Activation(double x,int i);
	double DActivation(double x, int i);
	void AddLayer(Layer l);
	void SetDisplayParameters(string s);
	void Summary();
	void PrintParameters();
	void Compile(string type,int batch_size);
	void Compile(string type);
	void Initialize();
	void Train(vector<vector<double>> *inputs,vector<vector<double>> *actual,vector<vector<double>>* predicted,int epochs,string loss);
	void Train(vector<vector<vector<double>>>* inputs, vector<vector<double>>* actual, vector<vector<double>>* predicted, int epochs, string loss);
	void ShowTrainingStats(vector<vector<double>>* inputs, vector<vector<double>>* actual,int i);
	void ForwardPropogation(vector<double> sample,vector<double> actualvalue,vector<vector<double>>* predicted);
	void ConvForwardPropogation(vector<vector<double>> sample, vector<double> actualvalue, vector<vector<double>>* predicted);
	void ErrorCalculation(vector<double> actualvalue);
	double DError(double predictedvalue,double actualvalue);
	void CleanErrors();
	void BackPropogation();
	void ConvBackPropogation();
	void LeakyReluParameters(double i, double a);
	vector<vector<double>> FullConvolve2D(vector<vector<double>>* input, vector<vector<double>>* kernel);
	vector<vector<double>> Convolve2D(vector<vector<double>> *input, vector<vector<double>> *kernel);
	vector<vector<double>> Rotate(vector<vector<double>>* input);
	void AddVectors(vector<vector<double>>* v1, vector<vector<double>>* v2);
	void UpdateKernel(vector<vector<double>>* v1, vector<vector<double>>* v2);
	vector<vector<double>> InitializeKernel(int kernelsize, int dilation);
	vector<vector<double>> Relu2D(vector<vector<double>>* input);
	void MaxPooling2D(vector<vector<double>>* input, short int padnum, vector<vector<double>>* outputdest, vector<vector<double>>* chosendest);
};