/*
Sean Connor - February 2022
605.617 Module 4 Assignment
*/

/* ========================================================================== */

#include <stdio.h>
#include <iostream>
#include <numeric>
#include <iterator>
#include <chrono>
#include <random>

static std::random_device rd;
static std::mt19937 rng{rd()};
static std::uniform_int_distribution<int> case2_val(0,3);
static std::uniform_int_distribution<int> case3_val(0,100);

__constant__ int c_in2[4096];

/* ========================================================================== */

/**
 * @brief 
 * 
 * @param n 
 * @param a 
 * @param b 
 * @param c 
 * @return __global__ 
 */
 __global__ 
 void add(int n, int *a, int *b, int *c) {
	
	for (int i = (blockIdx.x * blockDim.x) + threadIdx.x;
		 i < n;
		 i += blockDim.x * gridDim.x)
	{
		c[i] = a[i] + b[i];
	}
	
}

// overload function for constant memory usage
__global__ 
 void add(int n, int *a, int *c) {
	
	for (int i = (blockIdx.x * blockDim.x) + threadIdx.x;
		 i < n;
		 i += blockDim.x * gridDim.x)
	{
		c[i] = a[i] + c_in2[i];
	}
	
}

// overload function for shared memory usage
__global__ 
 void add(int n, int blockSize, int *a, int *b, int *c) {
	
	extern __shared__ int s[];
		
	for (int i = (blockIdx.x * blockDim.x) + threadIdx.x;
		 i < n;
		 i += blockDim.x * gridDim.x)
	{
		s[i] = b[i];
		__syncthreads();
		c[i] = a[i] + s[i];
	}
	
}

/* ========================================================================== */

/**
 * @brief 
 * 
 * @param n 
 * @param a 
 * @param b 
 * @param c 
 * @return __global__ 
 */
__global__ 
void subtract(int n, int *a, int *b, int *c) {
	
	for (int i = (blockIdx.x * blockDim.x) + threadIdx.x;
		 i < n;
		 i += blockDim.x * gridDim.x)
	{
		c[i] = a[i] - b[i];
	}
	
}

// overload function for constant memory usage
__global__ 
void subtract(int n, int *a, int *c) {
	
	for (int i = (blockIdx.x * blockDim.x) + threadIdx.x;
		 i < n;
		 i += blockDim.x * gridDim.x)
	{
		c[i] = a[i] - c_in2[i];
	}
	
}

// overload function for shared memory usage
__global__ 
void subtract(int n, int blockSize, int *a, int *b, int *c) {
	
	extern __shared__ int s[];
		
	for (int i = (blockIdx.x * blockDim.x) + threadIdx.x;
		 i < n;
		 i += blockDim.x * gridDim.x)
	{
		s[i] = b[i];
		__syncthreads();
		c[i] = a[i] - s[i];
	}
	
}

/* ========================================================================== */

/**
 * @brief 
 * 
 * @param n 
 * @param a 
 * @param b 
 * @param c 
 * @return __global__ 
 */
__global__ 
void multiply(int n, int *a, int *b, int *c) {
	
	for (int i = (blockIdx.x * blockDim.x) + threadIdx.x;
		 i < n;
		 i += blockDim.x * gridDim.x)
	{
		c[i] = a[i] * b[i];
	}
	
}

// overload function for constant memory usage
__global__ 
void multiply(int n, int *a, int *c) {
	
	for (int i = (blockIdx.x * blockDim.x) + threadIdx.x;
		 i < n;
		 i += blockDim.x * gridDim.x)
	{
		c[i] = a[i] * c_in2[i];
	}
	
}

// overload function for shared memory usage
__global__ 
void multiply(int n, int blockSize, int *a, int *b, int *c) {
	
	extern __shared__ int s[];
		
	for (int i = (blockIdx.x * blockDim.x) + threadIdx.x;
		 i < n;
		 i += blockDim.x * gridDim.x)
	{
		s[i] = b[i];
		__syncthreads();
		c[i] = a[i] * s[i];
	}
	
}

/* ========================================================================== */

/**
 * @brief 
 * 
 * @param n 
 * @param a 
 * @param b 
 * @param c 
 * @return __global__ 
 */
__global__ 
void modulo(int n, int *a, int *b, int *c) {
	
	for (int i = (blockIdx.x * blockDim.x) + threadIdx.x;
		 i < n;
		 i += blockDim.x * gridDim.x)
	{
		c[i] = a[i] % b[i];
	}
	
}

// overload function for constant memory usage
__global__ 
void modulo(int n, int *a, int *c) {
	
	for (int i = (blockIdx.x * blockDim.x) + threadIdx.x;
		 i < n;
		 i += blockDim.x * gridDim.x)
	{
		c[i] = a[i] % c_in2[i];
	}
	
}

// overload function for shared memory usage
__global__ 
 void modulo(int n, int blockSize, int *a, int *b, int *c) {
	
	extern __shared__ int s[];
		
	for (int i = (blockIdx.x * blockDim.x) + threadIdx.x;
		 i < n;
		 i += blockDim.x * gridDim.x)
	{
		s[i] = b[i];
		__syncthreads();
		c[i] = a[i] % s[i];
	}
	
}

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
		// *dataSz = atoi(argv[3]);
		*dataSz = 4096;
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

/**
 * @brief 
 * 
 * @param numBlocks 
 * @param blockSize 
 * @param dataSize 
 * @param out1 
 * @param d_in1 
 * @param d_in2 
 * @param d_out1 
 * @param type 
 */
void do_action(int numBlocks, int blockSize, int dataSize, int * d_in1, 
	int * d_in2, int * d_out1, int type) {
	
	using namespace std;

	char name[20];
	switch(type) {
		case 1:
			strcpy_s(name, 20,"add.txt");
			// add<<<numBlocks, blockSize>>>(dataSize, d_in1, d_in2, d_out1);
			break;
		case 2:
			strcpy_s(name, 20, "sub.txt");
			subtract<<<numBlocks, blockSize>>>(dataSize, d_in1, d_in2, d_out1);
			break;
		case 3:
			strcpy_s(name, 20, "mul.txt");
			multiply<<<numBlocks, blockSize>>>(dataSize, d_in1, d_in2, d_out1);
			break;
		case 4:
			strcpy_s(name, 20, "mod.txt");
			modulo<<<numBlocks, blockSize>>>(dataSize, d_in1, d_in2, d_out1);
			break;
		default:
			std::cout << "INVALID" << std::endl;
			exit(0);
	}

}

/* ========================================================================== */

/**
 * @brief 
 * 
 * @param in1 
 * @param in2 
 * @param out1 
 * @param pTotalThreads 
 * @param pBlockSize 
 * @param pDataSize 
 */
void constant_test(int * pTotalThreads, int * pBlockSize, int * pDataSize, 
	float * pTimer) {

	int size = *pDataSize * sizeof(int);
	int numBlocks = *pTotalThreads / *pBlockSize;

	// allocate host data arrays
	int *in1 = new int[*pDataSize] {0};
	int *in2 = new int[*pDataSize] {0};
	int *out1 = new int[*pDataSize] {0};
	
	// generate data and allocate device data (global)
	int *d_in1, *d_in2, *d_out1;
	in_generator(in1,*pDataSize,1);
	in_generator(in2,*pDataSize,2);
	cudaMalloc((void **) &d_in1, size);
	cudaMalloc((void **) &d_in2, size);
	cudaMalloc((void **) &d_out1, size);
	cudaMemcpy(d_in1, in1, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_in2, in2, size, cudaMemcpyHostToDevice);
		
	// copy device global --> constant
	cudaMemcpyToSymbol(c_in2, d_in2, size, 0, cudaMemcpyDeviceToDevice);

	// set up CUDA timing
	// https://developer.nvidia.com/blog/how-implement-performance-metrics-cuda-cc/
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	// execute kernels and time
	cudaEventRecord(start);	
	add<<<numBlocks, *pBlockSize>>>(*pDataSize, d_in1, d_out1);
	subtract<<<numBlocks, *pBlockSize>>>(*pDataSize, d_in1, d_out1);
	multiply<<<numBlocks, *pBlockSize>>>(*pDataSize, d_in1, d_out1);
	modulo<<<numBlocks, *pBlockSize>>>(*pDataSize, d_in1, d_out1);
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(pTimer, start, stop);	
	
	// // print statement to verify accuracy of results
	// cudaMemcpy(out1, d_out1, size, cudaMemcpyDeviceToHost);
	// for (int j = 0; j < *pDataSize; j+=128) {
	// 	printf("out[%d] = %d\n", j, out1[j]);
	// }
		
	// clean up
	cudaFree(d_in1); cudaFree(d_in2); cudaFree(d_out1);
	delete [] in1; 	delete [] in2; delete [] out1;
			
}

/* ========================================================================== */

/**
 * @brief 
 * 
 * @param in1 
 * @param in2 
 * @param out1 
 * @param pTotalThreads 
 * @param pBlockSize 
 * @param pDataSize 
 */
 void shared_test(int * pTotalThreads, int * pBlockSize, int * pDataSize, 
	float * pTimer) {

	int size = *pDataSize * sizeof(int);
	int numBlocks = *pTotalThreads / *pBlockSize;

	// allocate host data arrays
	int *in1 = new int[*pDataSize] {0};
	int *in2 = new int[*pDataSize] {0};
	int *out1 = new int[*pDataSize] {0};
	
	// generate data and allocate device data (global)
	int *d_in1, *d_in2, *d_out1;
	in_generator(in1,*pDataSize,1);
	in_generator(in2,*pDataSize,2);
	cudaMalloc((void **) &d_in1, size);
	cudaMalloc((void **) &d_in2, size);
	cudaMalloc((void **) &d_out1, size);
	cudaMemcpy(d_in1, in1, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_in2, in2, size, cudaMemcpyHostToDevice);
		
	// copy device global --> constant
	cudaMemcpyToSymbol(c_in2, d_in2, size, 0, cudaMemcpyDeviceToDevice);

	// set up CUDA timing
	// https://developer.nvidia.com/blog/how-implement-performance-metrics-cuda-cc/
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	// execute kernels and time
	int blkSzBytes = *pBlockSize*sizeof(int);
	cudaEventRecord(start);	
	add<<<numBlocks, *pBlockSize, blkSzBytes>>>(*pDataSize, d_in1, d_out1);
	subtract<<<numBlocks, *pBlockSize, blkSzBytes>>>(*pDataSize, d_in1, d_out1);
	multiply<<<numBlocks, *pBlockSize, blkSzBytes>>>(*pDataSize, d_in1, d_out1);
	modulo<<<numBlocks, *pBlockSize, blkSzBytes>>>(*pDataSize, d_in1, d_out1);
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(pTimer, start, stop);	
	
	// // print statement to verify accuracy of results
	// cudaMemcpy(out1, d_out1, size, cudaMemcpyDeviceToHost);
	// for (int j = 0; j < *pDataSize; j+=128) {
	// 	printf("out[%d] = %d\n", j, out1[j]);
	// }
		
	// clean up
	cudaFree(d_in1); cudaFree(d_in2); cudaFree(d_out1);
	delete [] in1; 	delete [] in2; delete [] out1;
			
}

/* ========================================================================== */

/**
 * @brief 
 * 
 * @param in1 
 * @param in2 
 * @param out1 
 * @param pTotalThreads 
 * @param pBlockSize 
 * @param pDataSize 
 */
 void global_test(int * pTotalThreads, int * pBlockSize, int * pDataSize, 
	float * pTimer) {

	int size = *pDataSize * sizeof(int);
	int numBlocks = *pTotalThreads / *pBlockSize;

	// allocate host data arrays
	int *in1 = new int[*pDataSize] {0};
	int *in2 = new int[*pDataSize] {0};
	int *out1 = new int[*pDataSize] {0};
	
	// generate data and allocate device data (global)
	int *d_in1, *d_in2, *d_out1;
	in_generator(in1,*pDataSize,1);
	in_generator(in2,*pDataSize,2);
	cudaMalloc((void **) &d_in1, size);
	cudaMalloc((void **) &d_in2, size);
	cudaMalloc((void **) &d_out1, size);
	cudaMemcpy(d_in1, in1, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_in2, in2, size, cudaMemcpyHostToDevice);

	// set up CUDA timing
	// https://developer.nvidia.com/blog/how-implement-performance-metrics-cuda-cc/
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	// execute kernels and time
	cudaEventRecord(start);	
	add<<<numBlocks, *pBlockSize>>>(*pDataSize, d_in1, d_in2, d_out1);
	subtract<<<numBlocks, *pBlockSize>>>(*pDataSize, d_in1, d_in2, d_out1);
	multiply<<<numBlocks, *pBlockSize>>>(*pDataSize, d_in1, d_in2, d_out1);
	modulo<<<numBlocks, *pBlockSize>>>(*pDataSize, d_in1, d_in2, d_out1);
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(pTimer, start, stop);	
	
	// // print statement to verify accuracy of results
	// cudaMemcpy(out1, d_out1, size, cudaMemcpyDeviceToHost);
	// for (int j = 0; j < *pDataSize; j+=128) {
	// 	printf("out[%d] = %d\n", j, out1[j]);
	// }
		
	// clean up
	cudaFree(d_in1); cudaFree(d_in2); cudaFree(d_out1);
	delete [] in1; 	delete [] in2; delete [] out1;
			
}

/* ========================================================================== */

int main(int argc, char** argv)
{
	using namespace std;
	
	// read command line arguments
	int totalThreads = 8192;
	int blockSize = 256;
	int dataSize = 8192;
	float timer = 0.0;
	int *pTotalThreads = &totalThreads;
	int *pBlockSize = &blockSize;
	int *pDataSize = &dataSize;
	float *pTimer = &timer;
	parse_cmdline(argc, argv, pTotalThreads, pBlockSize, pDataSize);
	totalThreads = *pTotalThreads;
	blockSize = *pBlockSize;
	dataSize = *pDataSize;

	// test harness
	int iterations = 10;
	float** res = new float*[3];
	for(int i = 0; i < 3; ++i) {
		res[i] = new float[iterations];
	}

	for (int i = 0; i < iterations; i++) {
		
		// constant memory test
		constant_test(pTotalThreads, pBlockSize, pDataSize, pTimer);
		res[0][i] = *pTimer;
		
		// shared memory test
		shared_test(pTotalThreads, pBlockSize, pDataSize, pTimer);
		res[1][i] = *pTimer;

		// global memory test
		global_test(pTotalThreads, pBlockSize, pDataSize, pTimer);
		res[2][i] = *pTimer;

	}

	// write results array to file
	FILE * pFile;
	pFile = fopen("results.txt","w");

	float sum = 0.0;
	for(int i = 0; i < iterations; i++) {
        sum += res[0][i];
		fprintf(pFile, "Constant Memory[%d] = %f\n", i, res[0][i]);
    }
	printf("Constant Memory Average = %f\n", (sum/iterations));
	fprintf(pFile, "Constant Memory Average = %f\n", (sum/iterations));
	
	sum = 0.0;	
	for(int i = 0; i < iterations; i++) {
        sum += res[1][i];
		fprintf(pFile, "Shared Memory[%d] = %f\n", i, res[1][i]);
    }
	printf("Shared Memory Average = %f\n", (sum/iterations));
	fprintf(pFile, "Shared Memory Average = %f\n", (sum/iterations));

	sum = 0.0;	
	for(int i = 0; i < iterations; i++) {
        sum += res[2][i];
		fprintf(pFile, "Global Memory[%d] = %f\n", i, res[2][i]);
    }
	printf("Global Memory Average = %f\n", (sum/iterations));
	fprintf(pFile, "Global Memory Average = %f\n", (sum/iterations));

	fclose(pFile);
	
}

/* ========================================================================== */

/* APPENDIX */

// cudaError_t err = cudaGetLastError();
    // if ( err != cudaSuccess ) {
    // 	printf("CUDA Error: %s\n", cudaGetErrorString(err)); 
    // 	// Possibly: exit(-1) if program cannot continue....
    // }