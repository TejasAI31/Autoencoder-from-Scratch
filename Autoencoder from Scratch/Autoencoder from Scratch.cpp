#include <fstream>
#include <random>
#include <raylib.h>

#include "model.h"

using namespace std;

void createSampleDataset(vector<vector<double>> *input,vector<vector<double>> *actual,int i)
{

	for (int x = 0; x < i; x++)
	{
		double randval = (double)(rand() / (double)RAND_MAX);

		vector<double> sampleinput;
		vector<double> sampleoutput;

		if (randval < 0.25)
		{
			sampleinput = { 0,0 };
			sampleoutput = { 0 };
		}
		else if (randval < 0.5)
		{
			sampleinput = { 0,1 };
			sampleoutput = { 1};
		}
		else if (randval < 0.75)
		{
			sampleinput = { 1,0 };
			sampleoutput = { 1 };
		}
		else
		{
			sampleinput = { 1,1 };
			sampleoutput = { 1 };
		}

		input->push_back(sampleinput);
		actual->push_back(sampleoutput);
	}
}

void CreateMnistDataset(vector<vector<double>>* input, vector<vector<double>>* actual,int i)
{
	if (i > 60000)
	{
		cout << "Only 60000 Instances Are Present" << endl;
		return;
	}
	cout << "LOADING DATASET" << endl;

	ifstream file("../Data/train-images.idx3-ubyte", ios::binary);

	vector<unsigned char> header(16);
	file.read((char*)(header.data()), 16);
	for (int x = 0; x < i; x++)
	{
		vector<unsigned char> temp(28 * 28);
		vector<double> sample;
		file.read((char*)(temp.data()), 28 * 28);
		for (int i = 0; i < 28 * 28; i++)
		{
			sample.push_back((double)(temp[i]));
		}

		actual->push_back(sample);
		input->push_back(sample);
	}
	file.close();

	cout << "DATASET LOADED\n\n";
}

void CreateMnistDataset2D(vector<vector<vector<double>>>* input, vector<vector<double>>* actual, int i)
{
	if (i > 60000)
	{
		cout << "Only 60000 Instances Are Present" << endl;
		return;
	}
	cout << "LOADING DATASET" << endl;

	ifstream file("../Data/train-images.idx3-ubyte", ios::binary);

	vector<unsigned char> header(16);
	file.read((char*)(header.data()), 16);
	for (int x = 0; x < i; x++)
	{
		vector<unsigned char> temp(28 * 28);
		vector<double> sample1D;
		vector<vector<double>> sample2D;
		file.read((char*)(temp.data()), 28 * 28);
		for (int i = 0; i < 28 * 28; i++)
		{
			sample1D.push_back((double)(temp[i])/(double)255);
		}

		for (int i = 0; i < 28; i++)
		{
			vector<double> row;
			for (int j = 0; j < 28; j++)
			{
				row.push_back((double)(temp[i * 28 + j])/(double)255);
			}
			sample2D.push_back(row);
		}

		actual->push_back(sample1D);
		input->push_back(sample2D);
	}
	file.close();

	cout << "DATASET LOADED\n\n";
}

void AddNoise(vector<vector<double>>* input)
{
	default_random_engine generator;
	normal_distribution<double> dist(0,120);

	for (int x = 0; x < input->size(); x++)
	{
		for (int y = 0; y < (*input)[x].size(); y++)
		{
			(*input)[x][y] += dist(generator);
			if ((*input)[x][y] < 0)
				(*input)[x][y] = 0;
		}
	}
}

void AddNoise2D(vector<vector<vector<double>>>* input)
{
	default_random_engine generator;
	normal_distribution<double> dist(0, 0.3);

	for (int x = 0; x < input->size(); x++)
	{
		for (int y = 0; y < (*input)[x].size(); y++)
		{
			for (int z = 0; z < (*input)[x][y].size(); z++)
			{
				(*input)[x][y][z] += dist(generator);
				if ((*input)[x][y][z] < 0)
					(*input)[x][y][z] = 0;
			}
		}
	}
}

void Display(vector<vector<double>>* input,vector<int> inputdims, vector<vector<double>>* predicted,vector<int> predicteddims ,vector<vector<double>>* actual)
{
	bool autotransition = false;

	const int windowwidth = 1200;
	const int windowheight = 800;
	const int boxside = 250;
	const int boxheight = 250;

	Rectangle InputBox = { 175,boxheight,boxside,boxside };
	Rectangle PredictedBox = { 475,boxheight,boxside,boxside };
	Rectangle ActualBox = { 775,boxheight,boxside,boxside };

	Rectangle NextButton = { 950,50,200,100 };
	Rectangle AutoButton = { 500,650,200,100 };
	Rectangle PrevButton = { 50,50,200,100 };

	InitWindow(windowwidth,windowheight, "Denoising Autoencoder");
	SetTargetFPS(144);

	Color LineWhite = { 100,100,100,100 };

	int samplenum =((int)rand())%input->size()-1;
	int framecounter = 0;

	Vector2 mousepos;

	while (!WindowShouldClose())
	{
		mousepos = GetMousePosition();
		framecounter++;
		if (framecounter == 60)
		{
			framecounter = 0;
			if(autotransition&&samplenum<input->size()-2)
			samplenum++;
		}

		BeginDrawing();

		ClearBackground({20,20,20,20});
		DrawRectangleRec(InputBox, BLACK);
		DrawRectangleLinesEx(InputBox,1,GREEN);
		DrawRectangleRec(PredictedBox, BLACK);
		DrawRectangleLinesEx(PredictedBox, 1, GREEN);
		DrawRectangleRec(ActualBox, BLACK);
		DrawRectangleLinesEx(ActualBox, 1, GREEN);

		DrawText("Input Image", InputBox.x+40, InputBox.y + boxside + 20, 30, WHITE);
		DrawText("Predicted Image", PredictedBox.x + 10, PredictedBox.y + boxside + 20, 30, WHITE);
		DrawText("Actual Image", ActualBox.x + 30, ActualBox.y + boxside + 20, 30, WHITE);

		DrawRectangleRec(NextButton, DARKGREEN);
		DrawRectangleRec(PrevButton, DARKGREEN);
		DrawRectangleRec(AutoButton, DARKGREEN);
		DrawRectangleLinesEx(AutoButton, 5, WHITE);
		DrawRectangleLinesEx(NextButton, 5,WHITE);
		DrawRectangleLinesEx(PrevButton,5, WHITE);

		DrawText("Next", NextButton.x + 55, NextButton.y + 30,40, WHITE);
		DrawText("Auto", AutoButton.x + 55, AutoButton.y + 30, 40, WHITE);
		DrawText("Previous", PrevButton.x + 25, PrevButton.y + 33, 35, WHITE);

		for (double i = 0; i <= boxside; i += boxside / (double)inputdims[0])
		{
			DrawLine(InputBox.x + i, InputBox.y, InputBox.x + i, InputBox.y + boxside, LineWhite);
			DrawLine(PredictedBox.x + i, PredictedBox.y, PredictedBox.x + i, PredictedBox.y + boxside, LineWhite);
			DrawLine(ActualBox.x + i, ActualBox.y, ActualBox.x + i, ActualBox.y + boxside, LineWhite);
		}

		for (double i = 0; i <= boxside; i += boxside / (double)inputdims[1])
		{
			DrawLine(InputBox.x, InputBox.y+i, InputBox.x + boxside, InputBox.y+i , LineWhite);
			DrawLine(PredictedBox.x , PredictedBox.y+i, PredictedBox.x + boxside, PredictedBox.y + i, LineWhite);
			DrawLine(ActualBox.x , ActualBox.y+i, ActualBox.x + boxside, ActualBox.y + i, LineWhite);
		}

		for (int i = 0; i < (*input)[0].size(); i++)
		{
			int row = i / inputdims[0];
			int column = i % inputdims[1];
			if ((*input)[samplenum][i] > 0)
			{
				Color tilecolor = { (*input)[samplenum][i] ,(*input)[samplenum][i] ,(*input)[samplenum][i] ,(*input)[samplenum][i] };
				DrawRectangle(InputBox.x + column * boxside / (double)inputdims[0], InputBox.y + row * boxside / (double)inputdims[1], boxside / (double)inputdims[0], boxside / (double)inputdims[1], tilecolor);
			}
		}

		for (int i = 0; i < (*predicted)[0].size(); i++)
		{
			int row = i / predicteddims[0];
			int column = i % predicteddims[1];
			if ((*predicted)[samplenum+1][i] > 0)
			{
				Color tilecolor = { (*predicted)[samplenum+1][i] ,(*predicted)[samplenum+1][i] ,(*predicted)[samplenum+1][i] ,(*predicted)[samplenum+1][i] };
				DrawRectangle(PredictedBox.x + column * boxside / (double)predicteddims[0], PredictedBox.y + row * boxside / (double)predicteddims[1], boxside / (double)predicteddims[0], boxside / (double)predicteddims[1], tilecolor);
			}
		}

		for (int i = 0; i < (*actual)[0].size(); i++)
		{
			int row = i / predicteddims[0];
			int column = i % predicteddims[1];
			if ((*actual)[samplenum][i] > 0)
			{
				Color tilecolor = { (*actual)[samplenum][i] ,(*actual)[samplenum][i] ,(*actual)[samplenum][i] ,(*actual)[samplenum][i] };
				DrawRectangle(ActualBox.x + column * boxside / (double)predicteddims[0], ActualBox.y + row * boxside / (double)predicteddims[1], boxside / (double)predicteddims[0], boxside / (double)predicteddims[1], tilecolor);
			}
		}

		if (CheckCollisionPointRec(mousepos, NextButton) && IsMouseButtonPressed(MOUSE_BUTTON_LEFT)&&samplenum<input->size()-2)
			samplenum+=1;
		if (CheckCollisionPointRec(mousepos, PrevButton) && IsMouseButtonPressed(MOUSE_BUTTON_LEFT)&&samplenum>0)
			samplenum -= 1;
		if (CheckCollisionPointRec(mousepos, AutoButton) && IsMouseButtonPressed(MOUSE_BUTTON_LEFT))
			autotransition = !autotransition;


		EndDrawing();
	}
}

void Display2D(vector<vector<vector<double>>>* input, vector<int> inputdims, vector<vector<double>>* predicted, vector<int> predicteddims, vector<vector<double>>* actual)
{
	bool autotransition = false;

	const int windowwidth = 1200;
	const int windowheight = 800;
	const int boxside = 250;
	const int boxheight = 250;

	Rectangle InputBox = { 175,boxheight,boxside,boxside };
	Rectangle PredictedBox = { 475,boxheight,boxside,boxside };
	Rectangle ActualBox = { 775,boxheight,boxside,boxside };

	Rectangle NextButton = { 950,50,200,100 };
	Rectangle AutoButton = { 500,650,200,100 };
	Rectangle PrevButton = { 50,50,200,100 };

	InitWindow(windowwidth, windowheight, "Denoising Autoencoder");
	SetTargetFPS(144);

	Color LineWhite = { 100,100,100,100 };

	int samplenum = 1;
	//int samplenum = ((int)rand()) % input->size();
	int framecounter = 0;

	Vector2 mousepos;

	while (!WindowShouldClose())
	{
		mousepos = GetMousePosition();
		framecounter++;
		if (framecounter == 20)
		{
			framecounter = 0;
			if (autotransition && samplenum < input->size()-2)
				samplenum++;
		}

		BeginDrawing();

		ClearBackground({ 20,20,20,20 });
		DrawRectangleRec(InputBox, BLACK);
		DrawRectangleLinesEx(InputBox, 1, GREEN);
		DrawRectangleRec(PredictedBox, BLACK);
		DrawRectangleLinesEx(PredictedBox, 1, GREEN);
		DrawRectangleRec(ActualBox, BLACK);
		DrawRectangleLinesEx(ActualBox, 1, GREEN);

		DrawText("Input Image", InputBox.x + 40, InputBox.y + boxside + 20, 30, WHITE);
		DrawText("Predicted Image", PredictedBox.x + 10, PredictedBox.y + boxside + 20, 30, WHITE);
		DrawText("Actual Image", ActualBox.x + 30, ActualBox.y + boxside + 20, 30, WHITE);

		DrawRectangleRec(NextButton, DARKGREEN);
		DrawRectangleRec(PrevButton, DARKGREEN);
		DrawRectangleRec(AutoButton, DARKGREEN);
		DrawRectangleLinesEx(AutoButton, 5, WHITE);
		DrawRectangleLinesEx(NextButton, 5, WHITE);
		DrawRectangleLinesEx(PrevButton, 5, WHITE);

		DrawText("Next", NextButton.x + 55, NextButton.y + 30, 40, WHITE);
		DrawText("Auto", AutoButton.x + 55, AutoButton.y + 30, 40, WHITE);
		DrawText("Previous", PrevButton.x + 25, PrevButton.y + 33, 35, WHITE);

		for (double i = 0; i <= boxside; i += boxside / (double)inputdims[0])
		{
			DrawLine(InputBox.x + i, InputBox.y, InputBox.x + i, InputBox.y + boxside, LineWhite);
			DrawLine(PredictedBox.x + i, PredictedBox.y, PredictedBox.x + i, PredictedBox.y + boxside, LineWhite);
			DrawLine(ActualBox.x + i, ActualBox.y, ActualBox.x + i, ActualBox.y + boxside, LineWhite);
		}

		for (double i = 0; i <= boxside; i += boxside / (double)inputdims[1])
		{
			DrawLine(InputBox.x, InputBox.y + i, InputBox.x + boxside, InputBox.y + i, LineWhite);
			DrawLine(PredictedBox.x, PredictedBox.y + i, PredictedBox.x + boxside, PredictedBox.y + i, LineWhite);
			DrawLine(ActualBox.x, ActualBox.y + i, ActualBox.x + boxside, ActualBox.y + i, LineWhite);
		}

		for (int i = 0; i < (*input)[0].size(); i++)
		{
			for (int j = 0; j < (*input)[0][0].size(); j++)
			{
				if ((*input)[samplenum][i][j] > 0)
				{
					Color tilecolor = { (*input)[samplenum][i][j]*255 ,(*input)[samplenum][i][j]*255 ,(*input)[samplenum][i][j]*255 ,(*input)[samplenum][i][j]*255};
					DrawRectangle(InputBox.x + j * boxside / (double)inputdims[0], InputBox.y + i * boxside / (double)inputdims[1], boxside / (double)inputdims[0], boxside / (double)inputdims[1], tilecolor);
				}
			}
		}

		for (int i = 0; i < (*predicted)[0].size(); i++)
		{
			int row = i / predicteddims[0];
			int column = i % predicteddims[1];
			if ((*predicted)[samplenum + 1][i] > 0)
			{
				Color tilecolor = { (*predicted)[samplenum + 1][i] * 255 ,(*predicted)[samplenum + 1][i] * 255,(*predicted)[samplenum + 1][i] * 255 ,(*predicted)[samplenum + 1][i] * 255 };
				DrawRectangle(PredictedBox.x + column * boxside / (double)predicteddims[0], PredictedBox.y + row * boxside / (double)predicteddims[1], boxside / (double)predicteddims[0], boxside / (double)predicteddims[1], tilecolor);
			}
		}

		for (int i = 0; i < (*actual)[0].size(); i++)
		{
			int row = i / predicteddims[0];
			int column = i % predicteddims[1];
			if ((*actual)[samplenum][i] > 0)
			{
				Color tilecolor = { (*actual)[samplenum][i] * 255 ,(*actual)[samplenum][i] * 255 ,(*actual)[samplenum][i] * 255 ,(*actual)[samplenum][i] * 255 };
				DrawRectangle(ActualBox.x + column * boxside / (double)predicteddims[0], ActualBox.y + row * boxside / (double)predicteddims[1], boxside / (double)predicteddims[0], boxside / (double)predicteddims[1], tilecolor);
			}
		}

		if (CheckCollisionPointRec(mousepos, NextButton) && IsMouseButtonPressed(MOUSE_BUTTON_LEFT) && samplenum < input->size()-2)
			samplenum += 1;
		if (CheckCollisionPointRec(mousepos, PrevButton) && IsMouseButtonPressed(MOUSE_BUTTON_LEFT) && samplenum > 0)
			samplenum -= 1;
		if (CheckCollisionPointRec(mousepos, AutoButton) && IsMouseButtonPressed(MOUSE_BUTTON_LEFT))
			autotransition = !autotransition;


		EndDrawing();
	}
}

int main()
{
	srand(time(NULL));

	vector<vector<vector<double>>> input;
	vector<vector<double>> actual;
	vector<vector<double>> predicted;

	CreateMnistDataset2D(&input, &actual, 1000);
	AddNoise2D(&input);

	//Model
	Network model;
	model.AddLayer(Layer(28 * 28, "Input2D"));
	model.AddLayer(Layer(12, 3, "Conv2D"));
	model.AddLayer(Layer(2, "Pool2D"));
	model.AddLayer(Layer(32, "Tanh"));
	model.AddLayer(Layer(96, "Tanh"));
	model.AddLayer(Layer(28 * 28, "Relu"));

	model.Compile("Stochastic");
	model.Summary();

	model.alpha=0.005;
	model.Train(&input, &actual,&predicted, 1, "Mean_Squared");

	Display2D(&input, { 28,28 },&predicted,{28,28}, &actual);
}