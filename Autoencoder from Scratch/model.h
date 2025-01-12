#pragma once
#include "layer.h"
#include <vector>

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
	double alpha = 0.1;

	int batch_size;

	double Activation(double x,int i);
	double DActivation(double x, int i);
	void AddLayer(Layer l);
	void SetDisplayParameters(string s);
	void Summary();
	void PrintParameters();
	void Compile(string type,int batch_size);
	void Compile(string type);
	void Initialize();
	void Train(vector<vector<double>> *inputs,vector<vector<double>> *actual,vector<vector<double>>* predicted,int epochs,string losstype);
	void ShowTrainingStats(vector<vector<double>>* inputs, vector<vector<double>>* actual,int i);
	void ForwardPropogation(vector<double> sample,vector<double> actualvalue,vector<vector<double>>* predicted);
	void ErrorCalculation(vector<double> actualvalue);
	double DError(double predictedvalue,double actualvalue);
	void BackPropogation(vector<double> actualvalue);
	void LeakyReluParameters(double i, double a);
};