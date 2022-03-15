/*
Sean Connor - March 2022
605.617 Module 7 Assignment
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
 * @param pTotalThreads 
 * @param pBlockSize 
 * @param pDataSize 
 * @param pTimer 
 */
 void global_test(int * pTotalThreads, int * pBlockSize, int * pDataSize, 
	float * pTimer) {

	int size = *pDataSize * sizeof(int);
	int numBlocks = *pTotalThreads / *pBlockSize;

	// allocate host data arrays
	int *in1 = new int[*pDataSize] {0};
	int *in2 = new int[*pDataSize] {0};
	int *out1 = new int[*pDataSize] {0};
    int *out2 = new int[*pDataSize] {0};
    int *out3 = new int[*pDataSize] {0};
    int *out4 = new int[*pDataSize] {0};
	
	// generate data and allocate device data (global)
	int *d_in1, *d_in2, *d_out1, *d_out2, *d_out3, *d_out4;
	in_generator(in1,*pDataSize,1);
	in_generator(in2,*pDataSize,2);
	cudaMalloc((void **) &d_in1, size);
	cudaMalloc((void **) &d_in2, size);
	cudaMalloc((void **) &d_out1, size);
    cudaMalloc((void **) &d_out2, size);
    cudaMalloc((void **) &d_out3, size);
    cudaMalloc((void **) &d_out4, size);

	// set up CUDA timing
	// https://developer.nvidia.com/blog/how-implement-performance-metrics-cuda-cc/
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	// execute and time!
	cudaEventRecord(start);	

    // memcpy to device
    cudaMemcpy(d_in1, in1, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_in2, in2, size, cudaMemcpyHostToDevice);

    // execute kernels
	add<<<numBlocks, *pBlockSize>>>(*pDataSize, d_in1, d_in2, d_out1);
	subtract<<<numBlocks, *pBlockSize>>>(*pDataSize, d_in1, d_in2, d_out2);
	multiply<<<numBlocks, *pBlockSize>>>(*pDataSize, d_in1, d_in2, d_out3);
	modulo<<<numBlocks, *pBlockSize>>>(*pDataSize, d_in1, d_in2, d_out4);

    // memcpy again
    cudaMemcpy(out1, d_out1, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(out2, d_out2, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(out3, d_out3, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(out4, d_out4, size, cudaMemcpyDeviceToHost);

    // stop timer and record
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(pTimer, start, stop);	
	
	// print statement to verify accuracy of results
	// for (int j = 0; j < *pDataSize; j+=512) {
	// 	printf("add[%d] = %d\n", j, out1[j]);
    //     printf("sub[%d] = %d\n", j, out2[j]);
    //     printf("mul[%d] = %d\n", j, out3[j]);
    //     printf("mod[%d] = %d\n", j, out4[j]);
	// }
		
	// clean up
	cudaFree(d_in1); cudaFree(d_in2); cudaFree(d_out1);
    cudaFree(d_out2); cudaFree(d_out3); cudaFree(d_out4);
	delete [] in1; 	delete [] in2; delete [] out1;
    delete [] out2; delete [] out3; delete [] out4;
			
}

/* ========================================================================== */

void stream_test(int * pTotalThreads, int * pBlockSize, int * pDataSize, 
    float * pTimer) {
	
	int size = *pDataSize * sizeof(int);
	int numBlocks = *pTotalThreads / *pBlockSize;

	// pinned memory -- host
	int *h_in1, *h_in2, *h_out1, *h_out2, *h_out3, *h_out4;
    cudaMallocHost((void **) &h_in1, size);
    cudaMallocHost((void **) &h_in2, size);
	cudaMallocHost((void **) &h_out1, size);
    cudaMallocHost((void **) &h_out2, size);
    cudaMallocHost((void **) &h_out3, size);
    cudaMallocHost((void **) &h_out4, size);
	in_generator(h_in1,*pDataSize,1);
	in_generator(h_in2,*pDataSize,2);
	memset(h_out1, 0, size);
    memset(h_out2, 0, size);
    memset(h_out3, 0, size);
    memset(h_out4, 0, size);	

	// pinned memory -- device
	int *d_in1, *d_in2, *d_out1, *d_out2, *d_out3, *d_out4;
	cudaMalloc((void **) &d_in1, size);
	cudaMalloc((void **) &d_in2, size);
	cudaMalloc((void **) &d_out1, size);
    cudaMalloc((void **) &d_out2, size);
    cudaMalloc((void **) &d_out3, size);
    cudaMalloc((void **) &d_out4, size);

    // set up CUDA timing
	// https://developer.nvidia.com/blog/how-implement-performance-metrics-cuda-cc/
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
    
    // stream and event setup 
    cudaStream_t stream1, stream2, stream3, stream4;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);
    cudaStreamCreate(&stream3);
    cudaStreamCreate(&stream4);

    cudaEvent_t event1, event2;
    cudaEventCreate(&event1);
    cudaEventCreate(&event2);

    // execute and time!
	cudaEventRecord(start);

    // memcpy to device -- sync at end before starting kernel exec
	cudaMemcpyAsync(d_in1, h_in1, size, cudaMemcpyHostToDevice, stream1);
	cudaMemcpyAsync(d_in2, h_in2, size, cudaMemcpyHostToDevice, stream2);
    cudaEventRecord(event1);
    cudaEventRecord(event2);

    // execute kernels
	add<<<numBlocks,*pBlockSize,0,stream1>>>(*pDataSize, d_in1, d_in2, d_out1);
    subtract<<<numBlocks,*pBlockSize,0,stream2>>>(*pDataSize, d_in1, d_in2, d_out2);
    multiply<<<numBlocks,*pBlockSize,0,stream3>>>(*pDataSize, d_in1, d_in2, d_out3);
    modulo<<<numBlocks,*pBlockSize,0,stream4>>>(*pDataSize, d_in1, d_in2, d_out4);

    // memcpy again
    cudaMemcpyAsync(h_out1, d_out1, size, cudaMemcpyDeviceToHost, stream1);
    cudaMemcpyAsync(h_out2, d_out2, size, cudaMemcpyDeviceToHost, stream2);
    cudaMemcpyAsync(h_out3, d_out3, size, cudaMemcpyDeviceToHost, stream3);
    cudaMemcpyAsync(h_out4, d_out4, size, cudaMemcpyDeviceToHost, stream4);

    // stop timer and record
    cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(pTimer, start, stop);
    
    // // print statement to verify accuracy of results
	// for (int j = 0; j < *pDataSize; j+=512) {
	// 	printf("add[%d] = %d\n", j, h_out1[j]);
    //     printf("sub[%d] = %d\n", j, h_out2[j]);
    //     printf("mul[%d] = %d\n", j, h_out3[j]);
    //     printf("mod[%d] = %d\n", j, h_out4[j]);
	// }
	
	// clean up
	cudaFreeHost(h_in1); cudaFreeHost(h_in2); cudaFreeHost(h_out1); 
    cudaFreeHost(h_out2); cudaFreeHost(h_out3); cudaFreeHost(h_out4);
	cudaFree(d_in1); cudaFree(d_in2); cudaFree(d_out1);
    cudaFree(d_out2); cudaFree(d_out3); cudaFree(d_out4);
    cudaStreamDestroy(stream1); cudaStreamDestroy(stream2);
    cudaStreamDestroy(stream3); cudaStreamDestroy(stream4);

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

    // test harness
	int iterations = 10;
	float** res = new float*[2];
	for(int i = 0; i < 2; ++i) {
		res[i] = new float[iterations];
	}

	for (int i = 0; i < iterations; i++) {

		// global memory test
		global_test(pTotalThreads, pBlockSize, pDataSize, pTimer);
		res[0][i] = *pTimer;

        // stream test
        stream_test(pTotalThreads, pBlockSize, pDataSize, pTimer);
		res[1][i] = *pTimer;

	}

	// write results array to file
	FILE * pFile;
	pFile = fopen("results.txt","w");

	float sum = 0.0;
	for(int i = 0; i < iterations; i++) {
        sum += res[0][i];
		fprintf(pFile, "Global Memory[%d] = %f\n", i, res[0][i]);
    }
	printf("Global Memory Average = %f\n", (sum/iterations));
	fprintf(pFile, "Global Memory Average = %f\n", (sum/iterations));

    sum = 0.0;
	for(int i = 0; i < iterations; i++) {
        sum += res[1][i];
		fprintf(pFile, "Stream[%d] = %f\n", i, res[1][i]);
    }
	printf("Stream Average = %f\n", (sum/iterations));
	fprintf(pFile, "Stream Average = %f\n", (sum/iterations));

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

// cudaError_t err = cudaGetLastError();
    // if ( err != cudaSuccess ) {
    // 	printf("CUDA Error: %s\n", cudaGetErrorString(err)); 
    // 	// Possibly: exit(-1) if program cannot continue....
    // }