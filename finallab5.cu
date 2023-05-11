/*
Author: Supriya Sundar
Class: ECE 6122
Last Date Modified: 11/21/2021
Description: Steady state temperature distribution of a thin plate was analysed and computed using CUDA programming.
References Used:
1.http://joshiscorner.com/files/src/blog/laplace-cuda-code.html
2.https://stackoverflow.com/questions/11994679/solving-2d-diffusion-heat-equation-with-cuda
3.https://stackoverflow.com/questions/32847449/alternatives-to-syncthreads-in-cuda
4.https://enccs.github.io/OpenACC-CUDA-beginners/2.01_cuda-introduction
5.https://codingbyexample.com/2019/02/06/error-handling-for-gpu-cuda
6.https://stackoverflow.com/questions/19600879/how-to-compare-arrays-of-char-in-cuda-c
*/

#include<cuda_runtime.h>
#include<cuda.h>
#include<device_launch_parameters.h>
#include<iostream>
#include<math.h>
#include<string.h>
#include<fstream>
#include<stdlib.h>
#include<stdio.h>
#include<string>
#include<cstring>
#include<stdlib.h>
#include<ctype.h>
#include<iomanip>

using namespace std;

__global__ void laplaceIterationGPU(double* oldValues, double* newValues, int N) // function to update array values
{
	int X = blockIdx.x * blockDim.x + threadIdx.x;//computing x and y coordinate of every node point
	int Y = blockIdx.y * blockDim.y + threadIdx.y;
	int point = X + Y * (N + 2); // attaining index value of point and neighbouring points
	int north = X + (Y + 1) * (N + 2);
	int south = X + (Y - 1) * (N + 2);
	int east = (X + 1) + Y * (N + 2);
	int west = (X - 1) + Y * (N + 2);
	if (X > 0 && X < (N + 1) && Y > 0 && Y < (N + 1)) //updating interior node points
	{
		newValues[point] = 0.25 * (oldValues[north] + oldValues[south] + oldValues[east] + oldValues[west]);
	}
	__syncthreads();
}

void createPlate(double* plateTemperature, int N)
{
	for (int i = 0; i < (N + 2); i++)
	{
		plateTemperature[i*(N+2)] = 20.0; //left edge of plate
		plateTemperature[(N + 1) * (N + 2) + i] = 20.0; //right edge of plate
		plateTemperature[i * (N + 2) + (N + 1)] = 20.0;// downside of plate
		if ((i > 0.3 * (N + 2)) && (i < 0.7 * (N + 2)))//upper edge of plate
		{
			plateTemperature[i] = 100.0;
		}
		else
		{
			plateTemperature[i] = 20.0;
		}
	}
	for (int i = 1; i <= N; i++)
	{
		for (int j = 1; j <= N; j++)
		{
			plateTemperature[i+j*(N+2)] = 20.0;
		}
	}
}


int main(int argc, char* argv[])
{
	int N = 0;
	int I = 0;
	if (argc < 2)
	{
		cout << "Invalid parameters, please check your values." << endl; //checking for invalid inputs
		return 0;
	}
	for (int i = 1; i < argc; i++) //checking for invalid inputs
	{
		if (strcmp(argv[i], "-N") == 0)
		{
			N = atoi(argv[i]);
			if (++i >= argc)
			{
				cout << "Invalid parameters, please check your values." << endl;
				return 0;
			}
			//cout << N << endl;
			if (N < 0)
			{
				cout << "Invalid parameters, please check your values." << endl;
				return 0;
			}
		}
		else if (strcmp(argv[i], "-I") == 0)
		{
			I = atoi(argv[i]);
			if (++i >= argc)
			{
				cout << "Invalid parameters, please check your values." << endl;
				return 0;
			}
			//cout << I << endl;
			if (I < 0)
			{
				cout << "Invalid parameters, please check your values." << endl;
				return 0;

			}
		}
	}


	size_t size = (N + 2) * (N + 2) * sizeof(double);

	//cpu temperature distribution
	double* cpuTemperature = (double*)malloc(size);
	double* cpuNewTemperature = (double*)malloc(size);
	createPlate(cpuTemperature,N);

	//gpu tempearture distribution
	double* gpuTemperature =NULL;
	cudaMalloc((void**)&gpuTemperature,size);
	double* gpuOldTemperature = NULL;
	cudaMalloc((void**)&gpuOldTemperature,size);

	//copying values from cpu to gpu
    cudaMemcpy(gpuTemperature,cpuTemperature,size,cudaMemcpyHostToDevice);
	cudaMemcpy(gpuOldTemperature,cpuTemperature,size,cudaMemcpyHostToDevice);

	//grid size declaration
	dim3 dimBlock(16, 16);
	dim3 dimGrid(16, 16);

	//dim3 dimBlock(32, 32);
	//dim3 dimGrid((N / 32) + 1, (N / 32) + 1);

	float time;

	//time computation for iteration duration
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	for (int i = 0; i < I; i++) //Jacobi iteration
	{
		laplaceIterationGPU << <dimGrid, dimBlock >> > (gpuTemperature, gpuOldTemperature, N);
		laplaceIterationGPU << <dimGrid, dimBlock >> > (gpuOldTemperature, gpuTemperature, N);
	}
 
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time, start, stop);
	printf("%.2f\n", time);

	//copy values from gpu to cpu
	cudaMemcpy(cpuNewTemperature,gpuTemperature,size,cudaMemcpyDeviceToHost);

	fstream file;
	file.open("finalTemperatures.csv", ios::out | ios::trunc);//writing output to file
	for (int i = 0; i < (N + 2); i++)
	{
		for (int j = 0; j < (N + 2); j++)
		{
			file << fixed << setprecision(6) << cpuNewTemperature[i * (N + 2)+j] << ",";
		}
		file << "\n";
	}
	file.close();
	//releasing memory
	free(cpuTemperature);
	free(cpuNewTemperature);
    cudaFree(gpuTemperature);
    cudaFree(gpuOldTemperature);
	return 0;
}
