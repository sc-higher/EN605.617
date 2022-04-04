/*
Sean Connor - March 2022
605.617 Module 8 Assignment
*/

/* ========================================================================== */

#include <stdio.h>
#include <iostream>
#include <numeric>
#include <iterator>
#include <chrono>
#include <random>
#include <math.h>
#include <cuda.h>
#include <cufft.h>
#include <cublas.h>
#include <curand.h>

#define IDX2C(i,j,ld) (((j)*(ld))+(i))

static std::random_device rd;
static std::mt19937 rng{rd()};
static std::uniform_int_distribution<int> case2_val(0,3);
static std::uniform_int_distribution<int> case3_val(0,100);

/* ========================================================================== */

/**
 * @brief 
 * 
 * @param array Point to array to be filled.
 * @param len Length of array to be filled.
 * @param type Type 1 = Increment 0 - len; Type 2 = 0-3 random; Type 3 = random
 */
void in_generator(int * array, int len, int type) {

	switch(type) {
		case 1:
			for (int i = 0; i < len; i++) {
				array[i] = i;
			}
			break;
		case 2:
			for (int i = 0; i < len; i++) {
				array[i] = case2_val(rng);
			}
			break;
		case 3:
			for (int i = 0; i < len; i++) {
				array[i] = case3_val(rng);
			}
			break;
		default:
			std::cout << "INVALID" << std::endl;
	}
	
}

/* ========================================================================== */

/**
 * @brief 
 * 
 * @param argc 
 * @param argv 
 */
void parse_cmdline(int argc, char **argv, int * numTh, int * blSz, int * dataSz)
{
	if (argc >= 2) {
		*numTh = atoi(argv[1]);
	}
	if (argc >= 3) {
		*blSz = atoi(argv[2]);
	}
	if (argc >= 4) {
		*dataSz = atoi(argv[3]);
	}

	int numBlocks = *numTh / *blSz;	

	// validate command line arguments
	if (*numTh % *blSz != 0) {
		++numBlocks;
		*numTh = numBlocks * (*blSz);
		printf("Warning: Total thread count is not evenly divisible by the block size\n");
		printf("The total number of threads will be rounded up to %d\n", *numTh);
	}
	printf("================\n");
	printf("Total Threads: %d\n", *numTh);
	printf("Block Size: %d\n", *blSz);
	printf("Data Size: %d\n", *dataSz);
	printf("================\n");
}

/* ========================================================================== */

cufftReal* generate_sine(int samples) {

	float pi = 3.14159;
	float amplitude = 1.0;
	float frequency = 10.0;
	float phase = 0.0;
	float sampleRate = 1000.0;

	cufftReal *arr = new cufftReal[samples];

	for (int i = 0; i < samples; i++)
  	{
		arr[i] = amplitude * sin(2.0 * pi * frequency * (i/sampleRate) + phase);
  	}

	return arr;

}

/* ========================================================================== */

void execute_fft_gpu(float * pTimer, int * pDataSize) {
	
	int NX = *pDataSize;

	// set up the cuFFT plan
	int batch = 1;
	cufftHandle plan;
    cufftPlan1d(&plan, NX, CUFFT_R2C, batch);

	// copy the data to device
	cufftReal *h_data = generate_sine(NX);
	cuComplex *r_data = (cuComplex*) malloc(sizeof(cuComplex)*NX);
	memset(r_data, 0, sizeof(cuComplex)*NX);

	// set up CUDA timing
	// https://developer.nvidia.com/blog/how-implement-performance-metrics-cuda-cc/
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	// execute and time!
	cudaEventRecord(start);
	
	cufftReal *d_data_in;
	cuComplex *d_data_out;	
	cudaMalloc((void **) &d_data_in, sizeof(cufftReal)*NX);
	cudaMalloc((void **) &d_data_out, sizeof(cuComplex)*NX);
	cudaMemcpy(d_data_in, h_data, sizeof(cufftReal)*NX, cudaMemcpyHostToDevice);

	// execute the cuFFT
	cufftExecR2C(plan, d_data_in, d_data_out);

	// copy results back to host
	cudaMemcpy(r_data, d_data_out, sizeof(cuComplex)*NX, cudaMemcpyDeviceToHost);

    // stop timer and record
    cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(pTimer, start, stop);	

	// // print statement to verify accuracy of results
	// float *res = new float[NX/2] {0.0};
	// for (int j = 1; j <= (NX/2); j++) {
	// 	res[j-1] = sqrt((r_data[j].x * r_data[j].x)+ (r_data[j].y * r_data[j].y));
	// 	if (res[j-1] > 5) {
	// 		printf("fft[%d] = %f\n", j-1, res[j-1]);
	// 	}
	// }

	// clean up
	cufftDestroy(plan);
	cudaFree(d_data_in); cudaFree(d_data_out);
	free(r_data);
	delete [] h_data;

}

/* ========================================================================== */

void execute_cublas_gpu_test() {

    cublasHandle_t handle;

	// int NX = *pDataSize;
	int M = 3;
	int N = 3;
	float * matrix = new float [M*N];

	for (int j = 0; j < N; j++) {
        for (int i = 0; i < M; i++) {
            matrix[IDX2C(i,j,M)] = 5.0;
        }
    }

	float *d_in1, *d_in2, *d_out1;
	cudaMalloc((void **) &d_in1, sizeof(float)*M*N);
	cudaMalloc((void **) &d_in2, sizeof(float)*M*N);
	cudaMalloc((void **) &d_out1, sizeof(float)*M*N);
	cublasCreate_v2(&handle);
	cublasSetMatrix (M, N, sizeof(float), matrix, M, d_in1, M);
	cublasSetMatrix (M, N, sizeof(float), matrix, M, d_in2, M);
	cublasSgemm('n', 'n', M, N, M, 1, d_in1, M, d_in2, M, 0, d_out1, M);
	cublasGetMatrix(M, N, sizeof(float), d_out1, M, matrix, M);

	for (int j = 0; j < N; j++) {
        for (int i = 0; i < M; i++) {
            printf("cublas[%d] = %f\n", i, matrix[IDX2C(i,j,M)]);
        }
    }

	// clean up
	cudaFree (d_in1);
	cudaFree (d_in2);
	cudaFree (d_out1);
    // cublasDestroy_v2(handle);
	delete[] matrix;

}

/* ========================================================================== */

void execute_cublas_gpu(float * pTimer, int * pDataSize) {

    cublasHandle_t handle;

	int NX = *pDataSize;
	int M, N = 0;
	M = sqrt(NX);
	N = M;
	int size = M*N;
	float * matrix = new float [size];
	

	// setup cuRAND and generate random matrix
	curandGenerator_t gen;
	curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_MTGP32);
	curandSetPseudoRandomGeneratorSeed(gen, 1234ULL);
	float * devRand;
	cudaMalloc((void **) &devRand, sizeof(float)*M*N);	
	curandGenerateUniform(gen, devRand, size);
	cudaMemcpy(matrix, devRand, sizeof(float)*size, cudaMemcpyDeviceToHost);

	// // print matrix
	// for (int j = 0; j < N; j++) {
    //     for (int i = 0; i < M; i++) {
    //         // matrix[IDX2C(i,j,M)] = 5.0;
	// 		printf("cublas[%d] = %f\n", i, matrix[IDX2C(i,j,M)]);
    //     }
    // }

	// set up CUDA timing
	// https://developer.nvidia.com/blog/how-implement-performance-metrics-cuda-cc/
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	// execute and time!
	cudaEventRecord(start);

	float *d_in1, *d_in2, *d_out1;
	cudaMalloc((void **) &d_in1, sizeof(float)*M*N);
	cudaMalloc((void **) &d_in2, sizeof(float)*M*N);
	cudaMalloc((void **) &d_out1, sizeof(float)*M*N);
	cublasCreate_v2(&handle);
	cublasSetMatrix (M, N, sizeof(float), matrix, M, d_in1, M);
	cublasSetMatrix (M, N, sizeof(float), matrix, M, d_in2, M);
	cublasSgemm('n', 'n', M, N, M, 1, d_in1, M, d_in2, M, 0, d_out1, M);
	cublasGetMatrix(M, N, sizeof(float), d_out1, M, matrix, M);

    // stop timer and record
    cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(pTimer, start, stop);	

	// // print statement to verify accuracy of results
	// for (int j = 0; j < N; j++) {
    //     for (int i = 0; i < M; i++) {
    //         printf("cublas[%d] = %f\n", i, matrix[IDX2C(i,j,M)]);
    //     }
    // }

	// clean up
	cudaFree (d_in1);
	cudaFree (d_in2);
	cudaFree (d_out1);
    // cublasDestroy_v2(handle);
	delete[] matrix;
	curandDestroyGenerator(gen);

}

/* ========================================================================== */

/**
 * @brief 
 * 
 * @param pTotalThreads 
 * @param pBlockSize 
 * @param pDataSize 
 * @param pTimer 
 */
void execute_gpu(int * pTotalThreads, int * pBlockSize, int * pDataSize, 
    float * pTimer) {

	// // test correctness of cuBLAS code
	// execute_cublas_gpu_test();

    // test harness
	int iterations = 10;
	float** res = new float*[2];
	for(int i = 0; i < 2; ++i) {
		res[i] = new float[iterations];
	}

	for (int i = 0; i < iterations; i++) {

		// CUFFT test
		execute_fft_gpu(pTimer, pDataSize);
		res[0][i] = *pTimer;

        // CUBLAS test
        execute_cublas_gpu(pTimer, pDataSize);
		res[1][i] = *pTimer;

	}

	// write results array to file
	FILE * pFile;
	pFile = fopen("results.txt","w");

	float sum = 0.0;
	for(int i = 1; i < iterations; i++) {
        sum += res[0][i];
		fprintf(pFile, "cuFFT[%d] = %f\n", i, res[0][i]);
    }
	printf("cuFFT Average = %f\n", (sum/iterations));
	fprintf(pFile, "cuFFT Average = %f\n", (sum/iterations));

    sum = 0.0;
	for(int i = 1; i < iterations; i++) {
        sum += res[1][i];
		fprintf(pFile, "cuBLAS[%d] = %f\n", i, res[1][i]);
    }
	printf("cuBLAS Average = %f\n", (sum/iterations));
	fprintf(pFile, "cuBLAS Average = %f\n", (sum/iterations));

	fclose(pFile);
    
}

/* ========================================================================== */

int main(int argc, char** argv) {
	
    using namespace std;
	
	// read command line arguments
	int totalThreads = 4096;
	int blockSize = 256;
	int dataSize = 4096;
	float timer = 0.0;
	int *pTotalThreads = &totalThreads;
	int *pBlockSize = &blockSize;
	int *pDataSize = &dataSize;
	float *pTimer = &timer;

	parse_cmdline(argc, argv, pTotalThreads, pBlockSize, pDataSize);
	execute_gpu(pTotalThreads, pBlockSize, pDataSize, pTimer);
	
}

/* ========================================================================== */

/* APPENDIX */

// CUDA ERROR CHECKING 

	// cudaError_t err = cudaGetLastError();
    // if ( err != cudaSuccess ) {
    // 	printf("CUDA Error: %s\n", cudaGetErrorString(err)); 
    // 	// Possibly: exit(-1) if program cannot continue....
    // }