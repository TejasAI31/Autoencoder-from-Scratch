#include "model.h"

using namespace std;

void Network::AddLayer(Layer l)
{
	layers.push_back(l);
}

void Network::PrintParameters()
{
	cout << "\n\n";
	for (int x = 0; x < layers.size() - 1; x++)
	{
		cout << "Layer " << x + 1 << " Weights:\n===============\n";
		int counter = 1;
		for (int y = 0; y < layers[x].number; y++)
		{
			for (int z = 0; z < layers[x + 1].number; z++)
			{
				//Prints Weights
				cout << counter++ << ". " << weights[x][y][z] << endl;
			}
		}
		cout << endl;
	}
}

void Network::Summary()
{
	//Network Summary
	cout << "Network SUMMARY\n=============\n\n";
	for (int x = 0; x < layers.size(); x++)
	{
		cout << "LAYER: " << layers[x].neurontype << "\t\tNUMBER: " << layers[x].number << "\n\n";
	}
	cout << "\n";
}

void Network::Initialize()
{
	//Parameter Initialisation
	weights = (double***)malloc(sizeof(double**) * layers.size() - 1);
	for (int x = 0; x < layers.size() - 1; x++)
	{
		weights[x] = (double**)malloc(sizeof(double*) * layers[x].number);
		for (int y = 0; y < layers[x].number; y++)
		{
			weights[x][y] = (double*)malloc(sizeof(double) * layers[x + 1].number);
			for (int z = 0; z < layers[x + 1].number; z++)
			{
				weights[x][y][z] = (double)rand() / (double)RAND_MAX;
			}
		}
	}

	errors = (double*)malloc(sizeof(double) * layers[layers.size() - 1].number);
}

void Network::Compile(string type)
{
	if (!type.compare("Stochastic"))
	{
		gradient_descent_type = Stochastic;
	}
	else if(!type.compare("Batch"))
	{
		gradient_descent_type = Batch;
	}
	else
	{
		cerr << "Mini Batch Gradient Descent Requires A Defined Batch Size" << endl;
	}
	Initialize();
}

void Network::Compile(string type, int input_batch_size)
{
	gradient_descent_type = Mini_Batch;
	batch_size = input_batch_size;
	Initialize();
}

double Network::DActivation(double x, int i)
{
	switch (layers[i].type)
	{
	case Layer::Sigmoid:
		return Activation(x, i) * (1 - Activation(x, i));
	case Layer::Relu:
		return (x > 0) ? 1 : 0;
	case Layer::LeakyRelu:
		return (x > 0) ? 1 : layers[i].parameters.LeakyReluAlpha;
	case Layer::Tanh:
		return 1 - pow(tanh(x), 2);
	}
}

double Network::Activation(double x,int i)
{
	switch (layers[i].type)
	{
	case Layer::Sigmoid:
		return 1 / (double)(1 + exp(-x));
	case Layer::Relu:
		return (x > 0) ? x : 0;
	case Layer::LeakyRelu:
		return (x > 0) ? x : layers[i].parameters.LeakyReluAlpha*x;
	case Layer::Tanh:
		return tanh(x);
	}
}

void Network::ErrorCalculation(vector<double> actualvalue)
{
	switch (model_loss_type)
	{
	case Mean_Squared:
		for (int i = 0; i < layers[layers.size() - 1].number; i++)
		{
			errors[i] = pow(layers[layers.size() - 1].values[i] - actualvalue[i], 2)/2;
		}
		break;
	}
}

double Network::DError(double predictedvalue,double actualvalue)
{
	int finallayer = layers.size() - 1;
	switch (model_loss_type)
	{
	case Mean_Squared:
		return actualvalue-predictedvalue;
	}
}

void Network::ForwardPropogation(vector<double> sample,vector<double> actualvalue,vector<vector<double>>* predicted)
{
	//Insert Sample Into Input Layer
	for (int i = 0; i < layers[0].number; i++)
	{
		layers[0].values[i] = sample[i];
	}


	//Calculate Forward Prop
	for (int i = 1; i < layers.size(); i++)
	{
		for (int j = 0; j < layers[i].number; j++)
		{
			double sum = 0;
			for (int k = 0; k < layers[i - 1].number; k++)
			{
				sum += weights[i - 1][k][j] * layers[i - 1].values[k];
			}
			layers[i].pre_activation_values[j] = sum;
			layers[i].values[j] = Activation(sum, i);
		}
	}

	//Add Output To The List Of Outputs
	vector<double> output;
	for (int x = 0; x < layers[layers.size() - 1].number; x++)
		output.push_back(layers[layers.size() - 1].values[x]);
	predicted->push_back(output);

	//Calculate Error
	ErrorCalculation(actualvalue);
}

void Network::BackPropogation(vector<double> actualvalue)
{
	int final_layer = layers.size() - 1;
	for (int i = 0; i < layers[final_layer].number; i++)
	{
		layers[final_layer].values[i] = DActivation(layers[final_layer].pre_activation_values[i],final_layer)*DError(layers[final_layer].values[i],actualvalue[i]);
		for (int j = 0; j < layers[final_layer - 1].number; j++)
		{
			weights[final_layer - 1][j][i] += alpha * layers[final_layer - 1].values[j] * layers[final_layer].values[i];
		}
	}

	for (int i = final_layer-1; i>0; i--)
	{
		for (int j = 0; j < layers[i].number;j++)
		{
			double sum = 0;
			for (int k = 0; k < layers[i + 1].number; k++)
			{
				sum += layers[i + 1].values[k]*weights[i][j][k];
			}
			layers[i].values[j] =DActivation(layers[i].pre_activation_values[j],i)*sum;

			for (int k = 0; k < layers[i - 1].number; k++)
			{
				weights[i - 1][k][j] += alpha * layers[i].values[j] * layers[i - 1].values[k];
			}
		}
	}
}

void Network::LeakyReluParameters(double i, double a)
{
	if (i<1 || i>layers.size())
	{
		cout << "Layer Number Out Of Range" << endl;
		return;
	}
	layers[i-1].parameters.LeakyReluAlpha = a;
}

void Network::ShowTrainingStats(vector<vector<double>>* inputs, vector<vector<double>>* actual,int i)
{
	//cout << "Inputs: ";
	//for (int j = 0; j < (*inputs)[i].size(); j++)
	//{
	//	cout << (*inputs)[i][j] << " ";
	//}

	cout << "\tPredicted: ";
	for (int j = 0; j < layers[layers.size() - 1].number; j++)
	{
		cout << layers[layers.size() - 1].values[j] << " ";
	}

	cout << "\tActual: ";
	for (int j = 0; j < (*actual)[i].size(); j++)
	{
		cout << (*actual)[i][j] << " ";
	}

	cout << "\tErrors: ";
	for (int j = 0; j < layers[layers.size() - 1].number; j++)
	{
		cout << errors[j] << " ";
	}

	cout << endl;
}

void Network::SetDisplayParameters(string s)
{
	if (!s.compare("Visual"))
		displayparameters = Visual;
	else if (!s.compare("Text"))
		displayparameters = Text;
	else
		displayparameters = None;
}

void Network::Train(vector<vector<double>> *inputs, vector<vector<double>> *actual,vector<vector<double>>* predicted, int epochs, string losstype)
{
	int inputsize = inputs->size();

	if (!losstype.compare("Mean_Squared"))
	{
		losstype = Mean_Squared;
	}

	cout << "Training\n========\n" << endl;

	for(int l=0;l<epochs;l++)
	for (int i = 0; i < inputsize; i++)
	{
		//Taking the sample
		vector<double> sample = (*inputs)[i];
		vector<double> actualvalue = (*actual)[i];

		ForwardPropogation(sample,actualvalue,predicted);

		if(displayparameters==Text)
			ShowTrainingStats(inputs, actual,i);
		BackPropogation(actualvalue);

		
	}
}