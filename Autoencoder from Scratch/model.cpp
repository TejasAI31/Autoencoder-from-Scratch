#include "model.h"

using namespace std;

//Image Transformations
double Network::MatrixAverage(vector<vector<double>>* mat)
{
	double sum = 0;
	for (int x = 0; x < mat->size(); x++)
	{
		for (int y = 0; y < (*mat)[0].size(); y++)
		{
			sum += (*mat)[x][y];
		}
	}
	return sum / (double)(mat->size()*(*mat)[0].size());
}

vector<vector<double>> Network::PixelDistances(vector<vector<double>>* mat1, vector<vector<double>>* mat2)
{
	vector<vector<double>> distancemat;
	if (mat1->size() != mat2->size()||mat1->empty())
		return {};

	for (int x = 0; x < mat1->size(); x++)
	{
		vector<double> row;
		for (int y = 0; y < (*mat1)[0].size(); y++)
		{
			row.push_back(sqrt(pow((*mat1)[x][y],2) + pow((*mat2)[x][y],2)));
		}
		distancemat.push_back(row);
	}

	return distancemat;
}

vector<vector<vector<double>>> Network::EmptyUpscale(vector<vector<vector<double>>>* image, int finalwidth, int finalheight)
{
	vector<vector<vector<double>>> finalimage;

	for (int i = 0; i < image->size(); i++)
	{
		vector<vector<double>> upscaled;
		int rowsize = (*image)[i][0].size();
		int columnsize = (*image)[i].size();
		int rowstep = floor(finalheight / columnsize);
		int columnstep = floor(finalwidth / rowsize);

		int ydeficit = finalheight - columnsize;
		int ycounter = 0;
		for (int x = 0; x < finalheight; x++)
		{
			//Last Column
			if (ycounter == columnsize - 1)
			{
				for (int y = 0; y < finalheight - x - 1; y++)
				{
					vector<double> empty(finalwidth,123.456);
					upscaled.push_back(empty);
				}
				break;
			}

			//Enter Full Row
			vector<double> row(finalwidth,123.456);
			int xdeficit = finalwidth - rowsize;
			int xcounter = 0;
			for (int y = 0; y < finalwidth; y++)
			{
				if (xcounter == rowsize - 1)break;
				row[y] = (*image)[i][ycounter][xcounter];
				if (xdeficit > 0)y+=columnstep;
				xdeficit--;
				xcounter++;
			}
			row[finalwidth - 1] = (*image)[i][ycounter][rowsize - 1];
			upscaled.push_back(row);
			
			//Empty Row
			if (ydeficit > 0)
			{
				for (int z = 0; z < rowstep; z++)
				{
					vector<double> empty(finalwidth, 123.456);
					upscaled.push_back(empty);
				}
				x += rowstep;
			}
			ydeficit--;
			ycounter++;
		}

		//Final Row
		vector<double> row(finalwidth,123.456);
		int xdeficit = finalwidth - rowsize;
		int xcounter = 0;
		for (int y = 0; y < finalwidth; y++)
		{
			if (xcounter == rowsize - 1)break;
			row[y] = (*image)[i][columnsize-1][xcounter];
			if (xdeficit > 0)y++;
			xdeficit--;
			xcounter++;
		}
		row[finalwidth - 1] = (*image)[i][ycounter][rowsize - 1];
		upscaled.push_back(row);

		//Send Image
		finalimage.push_back(upscaled);
	}
	return finalimage;
}

vector<vector<vector<double>>> Network::SobelEdgeDetection(vector<vector<vector<double>>>* image)
{
	static vector<vector<double>> sobelx = {
		{-1,0,1},
		{-2,0,2},
		{-1,0,1}
	};

	static vector<vector<double>> sobely = {
		{1,2,1},
		{0,0,0},
		{-1,-2,-1}
	};

	vector<vector<vector<double>>> edgeimages;
	
	for (int x = 0; x < image->size(); x++)
	{
		vector<vector<double>> xedges = Convolve2D(&((*image)[x]), &sobelx);
		vector<vector<double>> yedges = Convolve2D(&((*image)[x]), &sobely);
		vector<vector<double>> magnitude = PixelDistances(&xedges, &yedges);
		
		double threshold = MatrixAverage(&magnitude);

		for (int x = 0; x < magnitude.size(); x++)
		{
			for (int y = 0; y < magnitude[0].size(); y++)
			{
				magnitude[x][y] = (magnitude[x][y] < threshold) ? 0 : 1;
			}
		}

		edgeimages.push_back(magnitude);
	}
	return edgeimages;
}

vector<vector<vector<double>>> Network::PrewittEdgeDetection(vector<vector<vector<double>>>* image)
{
	static vector<vector<double>> prewittx = {
		{-1,0,1},
		{-1,0,1},
		{-1,0,1}
	};

	static vector<vector<double>> prewitty = {
		{1,1,1},
		{0,0,0},
		{-1,-1,-1}
	};

	vector<vector<vector<double>>> edgeimages;

	for (int x = 0; x < image->size(); x++)
	{
		vector<vector<double>> xedges = Convolve2D(&((*image)[x]), &prewittx);
		vector<vector<double>> yedges = Convolve2D(&((*image)[x]), &prewitty);
		vector<vector<double>> magnitude = PixelDistances(&xedges, &yedges);

		double threshold = MatrixAverage(&magnitude);

		for (int x = 0; x < magnitude.size(); x++)
		{
			for (int y = 0; y < magnitude[0].size(); y++)
			{
				magnitude[x][y] = (magnitude[x][y] < threshold) ? 0 : 1;
			}
		}

		edgeimages.push_back(magnitude);
	}
	return edgeimages;
}

vector<vector<vector<double>>> Network::NNInterpolation(vector<vector<vector<double>>>* image,int finalwidth,int finalheight)
{
	vector<vector<vector<double>>> emptyupscaled = EmptyUpscale(image, finalwidth, finalheight);
	for (int i = 0; i < emptyupscaled.size(); i++)
	{
		//Row Interpolation
		for (int x = 0; x < emptyupscaled[i].size(); x++)
		{
			//Empty Row
			if (emptyupscaled[i][x][0] == 123.456)continue;

			double value = emptyupscaled[i][x][0];
			for (int y = 1; y < emptyupscaled[i][x].size(); y++)
			{
				if (emptyupscaled[i][x][y] == 123.456)
					emptyupscaled[i][x][y] = value;
				else
					value = emptyupscaled[i][x][y];
			}
		}

		//Column Interpolation
		for (int x = 0; x < emptyupscaled[i][0].size(); x++)
		{
			double value = emptyupscaled[i][0][x];
			for (int y = 1; y < emptyupscaled[i].size(); y++)
			{
				if (emptyupscaled[i][y][x] == 123.456)
					emptyupscaled[i][y][x] = value;
				else
					value = emptyupscaled[i][y][x];
			}
		}
	}
	return emptyupscaled;
}

vector<vector<vector<double>>> Network::BilinearInterpolation(vector<vector<vector<double>>>* image, int finalwidth, int finalheight)
{
	vector<vector<vector<double>>> emptyupscaled = EmptyUpscale(image, finalwidth, finalheight);
	for (int i = 0; i < emptyupscaled.size(); i++)
	{
		//Row Interpolation
		for (int x = 0; x < emptyupscaled[i].size(); x++)
		{
			double value = emptyupscaled[i][x][0];
			int valuecount = 0;
			for (int y = 1; y < emptyupscaled[i][x].size(); y++)
			{
				if (emptyupscaled[i][x][y] != 123.456)
				{
					int denom = y - valuecount;
					for (int z = 1; z < denom; z++)
					{
						emptyupscaled[i][x][valuecount+z] =value*(denom-z)/(double)denom + emptyupscaled[i][x][y]*z/(double)denom;
					}
					value = emptyupscaled[i][x][y];
					valuecount = y;
				}
			}
		}

		//Column Interpolation
		for (int x = 0; x < emptyupscaled[i][0].size(); x ++)
		{
			double value = emptyupscaled[i][0][x];
			int valuecount = 0;
			for (int y = 1; y < emptyupscaled[i].size(); y++)
			{
				if (emptyupscaled[i][y][x] != 123.456)
				{
					int denom = y - valuecount;
					for (int z = 1; z < denom; z++)
					{
						emptyupscaled[i][valuecount + z][x] = value * (denom - z) / (double)denom + emptyupscaled[i][y][x] * z / (double)denom;
					}
					value = emptyupscaled[i][y][x];
					valuecount = y;
				}
			}
		}
	}
	return emptyupscaled;
}

//Convolution Functions
vector<vector<double>> Network::Convolve2D(vector<vector<double>> *input, vector<vector<double>> *kernel)
{
	short int kernelsize = kernel->size();
	short int columns = input->size();
	short int rows = (*input)[0].size();

	vector<vector<double>> output;

	for (int y = 0; y <= columns -columns% kernelsize; y++)
	{
		vector<double> row;
		for (int x = 0; x <= rows - rows%kernelsize; x++)
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

vector<vector<double>> Network::InitializeKernel(int kernelsize, int dilation)
{
	vector<vector<double>> kernel;

	for (int j = 0; j < kernelsize + (dilation - 1) * (kernelsize - 1); j++)
	{
		vector<double> row;
		for (int k = 0; k < kernelsize + (dilation - 1) * (kernelsize - 1); k++)
		{
			if (k % dilation == 0 && k % dilation == 0)
				row.push_back((float)rand() / (float)RAND_MAX);
			else
				row.push_back(0);
		}
		kernel.push_back(row);
	}
	return kernel;
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
			for (int j = 0; j< layers[i].kernelnumber; j++)
			{
				for (int k = 0; k < layers[i - 1].values2D.size(); k++)
				{
					vector<vector<double>> convolution = Convolve2D(&layers[i - 1].values2D[k], &layers[i].kernels[j][k]);
					AddVectors(&layers[i].pre_activation_values2D[j], &convolution);
				}
				layers[i].values2D[j] = Relu2D(&layers[i].pre_activation_values2D[j]);
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

void Network::ConvBackPropogation()
{
	//Initial Error
	short int final_layer = layers.size() - 1;
	for (int i = 0; i < layers[final_layer].number; i++)
	{
		for (int j = 0; j < layers[final_layer - 1].number; j++)
		{
			weights[final_layer - 1][j][i] += alpha * layers[final_layer - 1].values[j] * derrors[i];
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
			for (int j = 0; j < layers[i].kernelnumber; j++)
			{
				for (int k = 0; k < layers[i - 1].values2D.size(); k++)
				{
					//Calculate Change
					layers[i].deltakernel[j][k] = Convolve2D(&layers[i - 1].values2D[k], &layers[i].values2D[j]);
					vector<vector<double>> rotatedfilter = Rotate(&layers[i].kernels[j][k]);
					vector<vector<double>> delta2D = FullConvolve2D(&rotatedfilter, &layers[i].values2D[j]);

					//Update Change
					AddVectors(&layers[i - 1].values2Dderivative[k], &delta2D);
					UpdateKernel(&layers[i].kernels[j][k], &layers[i].deltakernel[j][k]);
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

//Generic Functions
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
			cout << "LAYER: " << "Conv2D" << "\t\tKERNEL SIZE: " << layers[x].kernelsize+(layers[x].dilation-1)*(layers[x].kernelsize-1)<<"\tKERNEL NUMBER: " << layers[x].kernelnumber<< "\n\n";
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
			for (int i = 0; i < layers[x].kernelnumber; i++)
			{
				vector<vector<vector<double>>> kernelset;
				layers[x].kernels.push_back(kernelset);
				layers[x].deltakernel.push_back(kernelset);

				for (int j = 0; j < layers[x - 1].values2D.size(); j++)
				{
					vector<vector<double>> kernel=InitializeKernel(layers[x].kernelsize,layers[x].dilation);
					vector<vector<double>> deltaker;
					
					layers[x].kernels[i].push_back(kernel);
					layers[x].deltakernel[i].push_back(deltaker);
				}
			}

			for (int j = 0; j < layers[x].kernelnumber; j++)
			{
				vector<vector<double>> temp;
				layers[x].pre_activation_values2D.push_back(temp);
				layers[x].values2D.push_back(temp);
				layers[x].values2Dderivative.push_back(temp);
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
	for (int x = 0; x < layers[layers.size() - 1].number; x++)
	{
		errors[x] = 0;
	}

	derrors = (double*)malloc(sizeof(double) * layers[layers.size() - 1].number);
	for (int x = 0; x < layers[layers.size() - 1].number; x++)
	{
		derrors[x] = 0;
	}
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
	batchsize = input_batch_size;
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

	for (int i = 0; i < layers[layers.size() - 1].number; i++)
	{
		double error = 0;
		switch (model_loss_type)
		{
			case Mean_Squared:
				error = pow(layers[layers.size() - 1].values[i] - actualvalue[i], 2) / 2;
				break;
		}

		if (gradient_descent_type == Stochastic)
		{
			errors[i] = error;
			derrors[i] = DActivation(layers.back().pre_activation_values[i], layers.size() - 1) * DError(layers.back().values[i], actualvalue[i]);
		}
		else
		{
			errors[i] += error / totalinputsize;
			derrors[i] += DActivation(layers.back().pre_activation_values[i], layers.size() - 1) * DError(layers.back().values[i], actualvalue[i])/totalinputsize;
		}

		
		avgerror += errors[i];
	}
	counter++;
	totalcounter++;
	
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

void Network::CleanErrors()
{
	for (int x = 0; x < layers.back().number; x++)
	{
		errors[x] = 0;
		derrors[x] = 0;
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

void Network::BackPropogation()
{
	int final_layer = layers.size() - 1;
	for (int i = 0; i < layers[final_layer].number; i++)
	{
		for (int j = 0; j < layers[final_layer - 1].number; j++)
		{
			weights[final_layer - 1][j][i] += alpha * layers[final_layer - 1].values[j] * derrors[i];
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

	switch (gradient_descent_type)
	{
	case Stochastic:
		for (int l = 0; l < epochs; l++)
		{
			for (int i = 0; i < inputsize; i++)
			{
				//Taking the sample
				vector<double> sample = (*inputs)[i];
				vector<double> actualvalue = (*actual)[i];

				ForwardPropogation(sample, actualvalue, predicted);

				//if (displayparameters == Text)
					//ShowTrainingStats(inputs, actual, i);

				BackPropogation();
			}
		}
		break;


	case Batch:
		for (int l = 0; l < epochs; l++)
		{
			for (int i = 0; i < inputsize; i++)
			{
				//Taking the sample
				vector<double> sample = (*inputs)[i];
				vector<double> actualvalue = (*actual)[i];

				ForwardPropogation(sample, actualvalue, predicted);

				//if (displayparameters == Text)
					//ShowTrainingStats(inputs, actual, i);
			}
			BackPropogation();
			CleanErrors();
		}
		break;


	case Mini_Batch:
		batchnum = ceil(totalinputsize / (float)batchsize);
		cout << "Total Batches= " << batchnum << "\n\n";
		for (int l = 0; l < epochs; l++)
		{
			for (int j = 0; j < batchnum; j++)
			{
				for (int i = 0; i < batchsize; i++)
				{
					if (j * batchsize + i < inputs->size())
					{
						//Taking the sample
						vector<double> sample = (*inputs)[i];
						vector<double> actualvalue = (*actual)[i];

						ForwardPropogation(sample, actualvalue, predicted);

						//if (displayparameters == Text)
						//ShowTrainingStats(inputs, actual, i);
					}
					else break;
				}
				BackPropogation();
				CleanErrors();
			}
		}
		break;
	}
}

void Network::Train(vector<vector<vector<double>>>* inputs, vector<vector<double>>* actual, vector<vector<double>>* predicted, int epochs, string loss)
{
	int inputsize = inputs->size();
	totalinputsize = inputs->size();
	totalepochs = epochs;

	if (!loss.compare("Mean_Squared"))
	{
		model_loss_type = Mean_Squared;
	}

	cout << "Training\n========\n" << endl;

	
	switch (gradient_descent_type)
	{
		case Stochastic:
			for (int l = 0; l < epochs; l++)
			{
				for (int i = 0; i < inputsize; i++)
				{
					//Taking the sample
					vector<vector<double>> sample = (*inputs)[i];
					vector<double> actualvalue = (*actual)[i];

					ConvForwardPropogation(sample, actualvalue, predicted);

					//if (displayparameters == Text)
						//ShowTrainingStats(inputs, actual, i);

					ConvBackPropogation();
				}
			}
			break;


		case Batch:
			for (int l = 0; l < epochs; l++)
			{
				for (int i = 0; i < inputsize; i++)
				{
					//Taking the sample
					vector<vector<double>> sample = (*inputs)[i];
					vector<double> actualvalue = (*actual)[i];

					ConvForwardPropogation(sample, actualvalue, predicted);

					//if (displayparameters == Text)
						//ShowTrainingStats(inputs, actual, i);
				}
				ConvBackPropogation();
				CleanErrors();
			}
			break;


		case Mini_Batch:
			batchnum = ceil(totalinputsize / (float)batchsize);
			cout << "Total Batches= " << batchnum << "\n\n";
			for (int l = 0; l < epochs; l++)
			{
				for (int j = 0; j < batchnum; j++)
				{
					for (int i = 0; i < batchsize; i++)
					{
						if (j * batchsize + i < inputs->size())
						{
							//Taking the sample
							vector<vector<double>> sample = (*inputs)[i];
							vector<double> actualvalue = (*actual)[i];

							ConvForwardPropogation(sample, actualvalue, predicted);

							//if (displayparameters == Text)
							//ShowTrainingStats(inputs, actual, i);
						}
						else break;
					}
					ConvBackPropogation();
					CleanErrors();
				}
			}
			break;
	}
}
//