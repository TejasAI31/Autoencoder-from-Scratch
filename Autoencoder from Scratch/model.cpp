#include "model.h"

using namespace std;

//Convolution Functions
vector<vector<double>> Network::Convolve2D(vector<vector<double>> *input, vector<vector<double>> *kernel)
{
	short int kernelsize = kernel->size();
	short int columns = input->size();
	short int rows = (*input)[0].size();
	short int rowpadding = kernelsize-rows % kernelsize;
	short int columnpadding = kernelsize-columns % kernelsize;

	vector<vector<double>> output;

	for (int y = 0; y <= columns -kernelsize+columnpadding; y++)
	{
		vector<double> row;
		for (int x = 0; x <= rows - kernelsize+ rowpadding; x++)
		{
			double value = 0;
			for (int i = 0; i < kernelsize; i++)
			{
				for (int j = 0; j < kernelsize; j++)
				{
					if (y + i >= columns || x + j >= rows)
						continue;
					else
						value += (*kernel)[i][j] * (*input)[y + i][x + j];
				}
			}
			row.push_back(value);
		}
		output.push_back(row);
	}

	return output;
}

vector<vector<double>> Network::FullConvolve2D(vector<vector<double>>* input, vector<vector<double>>* kernel)
{
	short int kernelsize = kernel->size();
	short int columns = input->size();
	short int rows = (*input)[0].size();
	short int rowpadding = kernelsize - rows % kernelsize;
	short int columnpadding = kernelsize - columns % kernelsize;

	vector<vector<double>> output;

	for (int y = 1-columns; y < 2*columns - kernelsize + columnpadding; y++)
	{
		vector<double> row;
		for (int x = 1-rows; x < 2*rows - kernelsize + rowpadding; x++)
		{
			double value = 0;
			for (int i = 0; i < kernelsize; i++)
			{
				for (int j = 0; j < kernelsize; j++)
				{
					if (y + i >= columns || x + j >= rows||y+i<0||x+j<0)
						continue;
					else
						value += (*kernel)[i][j] * (*input)[y + i][x + j];
				}
			}
			row.push_back(value);
		}
		output.push_back(row);
	}

	return output;
}

vector<vector<double>> Network::Relu2D(vector<vector<double>>* input)
{
	vector<vector<double>> output;
	for (int i = 0; i < input->size(); i++)
	{
		vector<double> row;
		for (int j = 0; j < (*input)[0].size(); j++)
		{
			row.push_back((*input)[i][j] > 0 ? (*input)[i][j] : 0);
		}
		output.push_back(row);
	}
	return output;
}

vector<vector<double>> Network::Rotate(vector<vector<double>>* input)
{
	vector<vector<double>> rotated;
	for (int i = input->size() - 1; i >= 0; i--)
	{
		vector<double> row;
		for (int j = (*input)[i].size() - 1; j >= 0; j--)
		{
			row.push_back((*input)[i][j]);
		}
		rotated.push_back(row);
	}
	return rotated;
}

void Network::AddVectors(vector<vector<double>>* v1, vector<vector<double>>* v2)
{
	if (v1->size() == 0)
	{
		(*v1) = (*v2);
		return;
	}
	else
	{
		for (int i = 0; i < v1->size(); i++)
		{
			for (int j = 0; j < (*v1)[0].size(); j++)
			{
				(*v1)[i][j] += (*v2)[i][j];
			}
		}
	}
}

void Network::MaxPooling2D(vector<vector<double>>* input, short int padnum, vector<vector<double>>* outputdest, vector<vector<double>>* chosendest)
{
	int columns = input->size();
	int rows = (*input)[0].size();
	short int rowpadding = padnum-rows % padnum;
	short int columnpadding = padnum-columns % padnum;

	vector<vector<double>> output;
	vector<vector<double>> chosenvalues(columns,vector<double>(rows));
	
		
	for (int y = 0; y <= columns + columnpadding; y+=padnum)
	{
		vector<double> row;
		for (int x = 0; x <= rows + rowpadding; x+=padnum)
		{
			short int chosenx = 0;
			short int choseny = 0;
			double maxval = 0;
			for (int i = 0; i < padnum; i++)
			{
				for (int j = 0; j < padnum; j++)
				{
					if (y + i >= columns || x + j >= rows)
						continue;
					else if ((*input)[y+i][x+j] > maxval)
					{
						maxval = (*input)[y+i][x+j];
						chosenx = x + j;
						choseny = y + i;
					}
				}
			}
			chosenvalues[choseny][chosenx] = 1;
			row.push_back(maxval);
		}
		output.push_back(row);
	}

	*outputdest = output;
	*chosendest = chosenvalues;
}

void Network::UpdateKernel(vector<vector<double>>* v1, vector<vector<double>>* v2)
{
	for (int i = 0; i < v1->size(); i++)
	{
		for (int j = 0; j < (*v1)[0].size(); j++)
		{
			(*v1)[i][j] += (*v2)[i][j]*alpha;
		}
	}
	return;
}

void Network::ConvForwardPropogation(vector<vector<double>> sample, vector<double> actualvalue, vector<vector<double>>* predicted)
{
	if (layers[0].type != Layer::Input2D)
	{
		cout << "\n\nLayer 0 is not Input2D!!\n\n";
		return;
	}

	//Insert Sample Into Input Layer
	layers[0].values2D[0]=sample;
	short int convend = 0;

	//Calculate Convolutions
	for (int i = 1; i < layers.size(); i++)
	{
		if (layers[i].type == Layer::Conv)
		{
			//For all previous convolutions
			for (int j = 0; j < layers[i - 1].values2D.size(); j++)
			{
				for (int k = 0; k < layers[i].kernelnumber; k++)
				{
					vector<vector<double>> convolution = Convolve2D(&layers[i - 1].values2D[j], &layers[i].kernels[k]);
					layers[i].pre_activation_values2D[j*layers[i].kernelnumber+k]=convolution;
					layers[i].values2D[j * layers[i].kernelnumber + k]=Relu2D(&convolution);
				}
			}
		}
		else if (layers[i].type == Layer::Pool2D)
		{
			//For all previous convolutions
			for (int j = 0; j < layers[i - 1].values2D.size(); j++)
			{
				MaxPooling2D(&layers[i - 1].values2D[j], layers[i].padding, &layers[i].values2D[j], &layers[i].pre_activation_values2D[j]);
			}
		}

		else
		{
			//FLATTEN
			layers[i - 1].values = (double*)malloc(sizeof(double) * layers[i - 1].values2D.size() * layers[i - 1].values2D[0].size() * layers[i - 1].values2D[0][0].size());
			layers[i - 1].number = layers[i - 1].values2D.size() * layers[i - 1].values2D[0].size() * layers[i - 1].values2D[0][0].size();
			unsigned long long int counter = 0;
			for (int j = 0; j < layers[i - 1].values2D.size(); j++)
			{
				for (int k = 0; k < layers[i - 1].values2D[j].size(); k++)
				{
					for (int l = 0; l < layers[i - 1].values2D[j][0].size(); l++)
					{
						layers[i - 1].values[counter] = layers[i - 1].values2D[j][k][l];
						counter++;
					}
				}
			}

			//Weight Initialization
			if (!layers[i-1].flattenweights)
			{
				weights[i - 1] = (double**)malloc(sizeof(double*) * layers[i - 1].number);
				for (int j = 0; j < layers[i - 1].number; j++)
				{
					weights[i - 1][j] = (double*)malloc(sizeof(double) * layers[i].number);
					for (int k = 0; k < layers[i].number; k++)
					{
						weights[i - 1][j][k] = (double)rand() / (double)RAND_MAX;
					}
				}
				layers[i - 1].flattenweights = true;
			}

			convend = i;
			break;
		}
	}

	//Calculate Forward Prop
	for (int i = convend; i < layers.size(); i++)
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

void Network::ConvBackPropogation(vector<double> actualvalue)
{
	//Initial Error
	short int final_layer = layers.size() - 1;
	for (int i = 0; i < layers[final_layer].number; i++)
	{
		layers[final_layer].values[i] = DActivation(layers[final_layer].pre_activation_values[i], final_layer) * DError(layers[final_layer].values[i], actualvalue[i]);
		for (int j = 0; j < layers[final_layer - 1].number; j++)
		{
			weights[final_layer - 1][j][i] += alpha * layers[final_layer - 1].values[j] * layers[final_layer].values[i];
		}
	}

	int convstart = 0;
	for (int i = final_layer - 1; i > 0; i--)
	{
		//Check for Convolution Start
		if (layers[i].type == Layer::Conv || layers[i].type == Layer::Pool2D)
		{
			convstart = i;
			//Calculate derivate
			for (long long unsigned int j = 0; j < layers[i].number; j++)
			{
				long long int sum = 0;
				for (int k = 0; k < layers[i + 1].number; k++)
				{
					sum += layers[i + 1].values[k] * weights[i][j][k];
				}
				layers[i].values[j] = sum;
			}
			break;
		}

		//Derivative
		for (int j = 0; j < layers[i].number; j++)
		{
			double sum = 0;
			for (int k = 0; k < layers[i + 1].number; k++)
			{
				sum += layers[i + 1].values[k] * weights[i][j][k];
			}
			layers[i].values[j] = DActivation(layers[i].pre_activation_values[j], i) * sum;

			for (int k = 0; k < layers[i - 1].number; k++)
			{
				weights[i - 1][k][j] += alpha * layers[i].values[j] * layers[i - 1].values[k];
			}
		}
	}

	//UNFLATTEN
	unsigned long long int counter = 0;
	for (int i = 0; i < layers[convstart].values2D.size(); i++)
	{
		vector<vector<double>> derivative;
		for (int j = 0; j < layers[convstart].values2D[i].size(); j++)
		{
			vector<double> row;
			for (int k = 0; k < layers[convstart].values2D[i][j].size(); k++)
			{
				row.push_back(layers[convstart].values[counter]);
				counter++;
			}
			derivative.push_back(row);
		}
		layers[convstart].values2Dderivative[i]=derivative;
	}

	//2D Derivative
	for (int i = convstart; i > 0; i--)
	{
		//Convolution Case
		if (layers[i].type == Layer::Conv)
		{
			for (int j = 0; j < layers[i - 1].values2D.size(); j++)
			{
				for (int k = 0; k < layers[i].kernelnumber; k++)
				{
					//Calculate Change
					layers[i].deltakernel[j * layers[i].kernelnumber + k] = Convolve2D(&layers[i - 1].values2D[j], &layers[i].values2D[j * layers[i].kernelnumber + k]);
					vector<vector<double>> rotatedfilter = Rotate(&layers[i].kernels[j * layers[i].kernelnumber + k]);
					vector<vector<double>> delta2D = FullConvolve2D(&rotatedfilter, &layers[i].values2D[j * layers[i].kernelnumber + k]);

					//Update Change
					AddVectors(&layers[i - 1].values2Dderivative[j], &delta2D);
					UpdateKernel(&layers[i].kernels[j * layers[i].kernelnumber + k], &layers[i].deltakernel[j * layers[i].kernelnumber + k]);
					
				}
			}
		}

		//Pooling Case
		else if (layers[i].type == Layer::Pool2D)
		{
			for (int h = 0; h < layers[i].values2D.size(); h++)
			{
				if (layers[i - 1].values2Dderivative[h].size() == 0)
				{
					for (int m = 0; m < layers[i - 1].values2D[h].size(); m++)
					{
						vector<double> row(layers[i - 1].values2D[h][m].size());
						layers[i - 1].values2Dderivative[h].push_back(row);
					}
				}

				for (int j = 0; j < layers[i].values2D[0].size(); j++)
				{
					for (int k = 0; k < layers[i].values2D[0][0].size(); k++)
					{
						for (int l = 0; l < layers[i].padding; l++)
						{
							for (int m = 0; m < layers[i].padding; m++)
							{
								if (layers[i].pre_activation_values2D[h][j + l][k + m])
									layers[i - 1].values2Dderivative[h][j + l][k + m] = layers[i].values2D[h][j][k];
							}
						}
					}
				}
			}
		}
	}
}

//Other Functions
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
		if (layers[x].type == Layer::Pool2D)
			cout << "LAYER: " << "Pool2D" << "\t\tDIMENSIONS: " << layers[x].padding<<"\n\n";
		else if(layers[x].type==Layer::Conv)
			cout << "LAYER: " << "Conv2D" << "\t\tKERNEL SIZE: " << layers[x].kernelsize<<"\tKERNEL NUMBER: " << layers[x].kernelnumber<< "\n\n";
		else
			cout << "LAYER: " << layers[x].neurontype << "\t\tNUMBER: " << layers[x].number << "\n\n";
	}
	cout << "\n";
}

void Network::Initialize()
{
	//Layer 0 Initialisation
	vector<vector<double>> temp;
	layers[0].values2D.push_back(temp);
	layers[0].values2Dderivative.push_back(temp);

	//Parameter Initialisation
	weights = (double***)malloc(sizeof(double**) * layers.size() - 1);
	for (int x = 0; x < layers.size() - 1; x++)
	{
		//Check for Convolution Layer
		if (layers[x].type == Layer::Conv)
		{
			for(int j=0;j<layers[x-1].values2D.size();j++)
			for (int i = 0; i < layers[x].kernelnumber; i++)
			{
				vector<vector<double>> kernel;
				vector<vector<double>> deltaker;
				for (int j = 0; j < layers[x].kernelsize; j++)
				{
					vector<double> row;
					for (int k = 0; k < layers[x].kernelsize; k++)
					{
						row.push_back((float)rand() / (float)RAND_MAX);
					}
					kernel.push_back(row);
				}
				layers[x].kernels.push_back(kernel);
				layers[x].deltakernel.push_back(deltaker);
			}

			for (int i = 0; i < layers[x - 1].values2D.size(); i++)
			{
				for (int j = 0; j < layers[x].kernelnumber; j++)
				{
					vector<vector<double>> temp;
					layers[x].pre_activation_values2D.push_back(temp);
					layers[x].values2D.push_back(temp);
					layers[x].values2Dderivative.push_back(temp);
				}
			}
		}

		//Check for Pooling
		else if (layers[x].type == Layer::Pool2D)
		{
			for (int i = 0; i < layers[x - 1].values2D.size(); i++)
			{
				vector<vector<double>> temp;
				layers[x].pre_activation_values2D.push_back(temp);
				layers[x].values2D.push_back(temp);
				layers[x].values2Dderivative.push_back(temp);
			}
		}

		//Other Cases
		else
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
	static double avgerror = 0;
	static int counter = 0;
	static int totalcounter = 0;

	switch (model_loss_type)
	{
	case Mean_Squared:
		for (int i = 0; i < layers[layers.size() - 1].number; i++)
		{
			errors[i] = pow(layers[layers.size() - 1].values[i] - actualvalue[i], 2)/2;
			avgerror += errors[i];
		}
		counter++;
		totalcounter++;
		break;
	}
	
	if (counter % 100==0)
	{
		cout << totalcounter<<". Average Error : " << avgerror/(double)counter << endl;
		avgerror = 0;
		counter = 0;
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

void Network::Train(vector<vector<vector<double>>>* inputs, vector<vector<double>>* actual, vector<vector<double>>* predicted, int epochs, string losstype)
{
	int inputsize = inputs->size();

	if (!losstype.compare("Mean_Squared"))
	{
		losstype = Mean_Squared;
	}

	cout << "Training\n========\n" << endl;

	for (int l = 0; l < epochs; l++)
		for (int i = 0; i < inputsize; i++)
		{
			//Taking the sample
			vector<vector<double>> sample = (*inputs)[i];
			vector<double> actualvalue = (*actual)[i];

			ConvForwardPropogation(sample, actualvalue, predicted);
			//if (displayparameters == Text)
				//ShowTrainingStats(inputs, actual, i);
			ConvBackPropogation(actualvalue);
		}
}