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
			add<<<numBlocks, blockSize>>>(dataSize, d_in1, d_in2, d_out1);
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
void pinned(int * pTotalThreads, int * pBlockSize, int * pDataSize) {
	
	int size = *pDataSize * sizeof(int);
	int numBlocks = *pTotalThreads / *pBlockSize;

	// pinned memory -- host
	int *h_in1, *h_in2, *h_out1;
    cudaMallocHost((void **) &h_in1, size);
    cudaMallocHost((void **) &h_in2, size);
	cudaMallocHost((void **) &h_out1, size);
	in_generator(h_in1,*pDataSize,1);
	in_generator(h_in2,*pDataSize,2);
	memset(h_out1, 0, size);	

	// pinned memory -- device
	int *d_in1, *d_in2, *d_out1;
	cudaMalloc((void **) &d_in1, size);
	cudaMalloc((void **) &d_in2, size);
	cudaMalloc((void **) &d_out1, size);
	cudaMemcpy(d_in1, h_in1, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_in2, h_in2, size, cudaMemcpyHostToDevice);

	// kernel execution
	for (int i = 1; i < 5; i++) {
		do_action(numBlocks, *pBlockSize, *pDataSize, d_in1, d_in2, d_out1, i);
		cudaMemcpy(h_out1, d_out1, size, cudaMemcpyDeviceToHost);

		// // print statement to verify accuracy of results
		// for (int j = 0; j < 10; j++) {
		// 	printf("out[%d] = %d\n", j, h_out1[j]);
		// }		

	}
	
	// clean up
	cudaFreeHost(h_in1); cudaFreeHost(h_in2); cudaFreeHost(h_out1);
	cudaFree(d_in1); cudaFree(d_in2); cudaFree(d_out1);

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
void paged(int * pTotalThreads, int * pBlockSize, int * pDataSize) {

	int size = *pDataSize * sizeof(int);
	int numBlocks = *pTotalThreads / *pBlockSize;

	// paged memory -- host
	int *in1 = new int[*pDataSize] {0};
	int *in2 = new int[*pDataSize] {0};
	int *out1 = new int[*pDataSize] {0};
	in_generator(in1,*pDataSize,1);
	in_generator(in2,*pDataSize,2);


	// paged memory -- device
	int *d_in1, *d_in2, *d_out1;
	cudaMalloc((void **) &d_in1, size);
	cudaMalloc((void **) &d_in2, size);
	cudaMalloc((void **) &d_out1, size);
	cudaMemcpy(d_in1, in1, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_in2, in2, size, cudaMemcpyHostToDevice);

	// kernel execution
	for (int i = 1; i < 5; i++) {
		do_action(numBlocks, *pBlockSize, *pDataSize, d_in1, d_in2, d_out1, i);
		cudaMemcpy(out1, d_out1, size, cudaMemcpyDeviceToHost);

		// // print statement to verify accuracy of results
		// for (int j = 0; j < 10; j++) {
		// 	printf("out[%d] = %d\n", j, out1[j]);
		// }
		
	}

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
	int *pTotalThreads = &totalThreads;
	int *pBlockSize = &blockSize;
	int *pDataSize = &dataSize;
	parse_cmdline(argc, argv, pTotalThreads, pBlockSize, pDataSize);
	totalThreads = *pTotalThreads;
	blockSize = *pBlockSize;
	dataSize = *pDataSize;

	// set up CUDA timing
	// https://developer.nvidia.com/blog/how-implement-performance-metrics-cuda-cc/
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	// paged execution and performance
	cudaEventRecord(start);
	paged(pTotalThreads, pBlockSize, pDataSize);
	cudaEventRecord(stop);

	// pinned execution and performance
	cudaEventRecord(start);
	pinned(pTotalThreads, pBlockSize, pDataSize);
	cudaEventRecord(stop);
	
	// // OLD TIMERS
	// auto start = chrono::high_resolution_clock::now();
	// auto stop = chrono::high_resolution_clock::now();
	// auto duration = chrono::duration_cast<chrono::microseconds>(stop - start);
	
}

/* ========================================================================== */

