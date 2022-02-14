/*
Sean Connor - February 2022
605.617 Module 3 Assignment

REQUIREMENTS
- Create two input arrays (size >= 1024). **DONE**
	- in1[] = values between 0 and total number of threads.
	- in2[] = random values between 0 and 3.
- Compute out1[], out2[], out3[], and out4[]: **DONE**
	- out1[] = in1[i] + in2[i]
	- out2[] = in1[i] - in2[i]
	- out3[] = in1[i] * in2[i]
	- out4[] = in1[i] % in2[i]
- *Challenge* Compare pre-kernel data prep with conditional branching within 
  kernel. Make two random arrays (size >= 1024). Determine execution time for
  all runs.
	- Case 1 - Sort such that rand1[i] > rand2[i], then pass to subtraction 
	kernel.
	- Case 2 - Create alternate subtraction kernel that does conditional 
	branching.

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
__global__ void add(int n, int *a, int *b, int *c) {
	
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
__global__ void subtract(int n, int *a, int *b, int *c) {
	
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
__global__ void multiply(int n, int *a, int *b, int *c) {
	
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
__global__ void modulo(int n, int *a, int *b, int *c) {
	
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
 * @param n 
 * @param a 
 * @param b 
 * @param c 
 * @return __global__ 
 */
__global__ void abs(int n, int *a, int *b, int *c) {
	
	for (int i = (blockIdx.x * blockDim.x) + threadIdx.x;
		 i < n;
		 i += blockDim.x * gridDim.x)
	{
		if ( b[i] > a[i] ) {
			c[i] = b[i] - a[i];
		} else {
			c[i] = a[i] - b[i];
		}
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
}

/* ========================================================================== */

void do_action(int numBlocks, int blockSize, int dataSize, int * out1, 
	int * d_in1, int * d_in2, int * d_out1, int type) {
	
	using namespace std;

	auto start = chrono::high_resolution_clock::now();
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
		case 5:
			strcpy_s(name, 20, "p2sub.txt");
			subtract<<<numBlocks, blockSize>>>(dataSize, d_in1, d_in2, d_out1);
			break;
		case 6:
			strcpy_s(name, 20, "p2abs.txt");
			abs<<<numBlocks, blockSize>>>(dataSize, d_in1, d_in2, d_out1);
			break;
		default:
			std::cout << "INVALID" << std::endl;
			exit(0);
	}
	auto stop = chrono::high_resolution_clock::now();
	auto duration = chrono::duration_cast<chrono::microseconds>(stop - start);
	
	// copy the data back to host
	int size = dataSize * sizeof(int);
	cudaMemcpy(out1, d_out1, size, cudaMemcpyDeviceToHost);

	FILE * pFile;
	pFile = fopen(name,"w");
	fprintf(pFile, "Duration: %I64dus\n", duration.count());
	for(int i = 0; i < dataSize; i++) {
		fprintf(pFile, "out[%d] = %d\n", i, out1[i]);
	}	
	fclose(pFile);

}

/* ========================================================================== */

void do_action_p2(int numBlocks, int blockSize, int dataSize, int * out1, 
	int * d_in1, int * d_in2, int * d_out1, int type, int iter, int ** res) {
	
	using namespace std;

	auto start = chrono::high_resolution_clock::now();
	char name[20];
	switch(type) {
		case 1:
			strcpy_s(name, 20, "p2sub.txt");
			subtract<<<numBlocks, blockSize>>>(dataSize, d_in1, d_in2, d_out1);
			break;
		case 2:
			strcpy_s(name, 20, "p2abs.txt");
			abs<<<numBlocks, blockSize>>>(dataSize, d_in1, d_in2, d_out1);
			break;
		default:
			std::cout << "INVALID" << std::endl;
			exit(0);
	}
	auto stop = chrono::high_resolution_clock::now();
	auto duration = chrono::duration_cast<chrono::microseconds>(stop - start);

	if (type == 1) {
		res[0][iter] = (int) duration.count();
	} else if ( type == 2) {
		res[1][iter] = (int) duration.count();
	}
	
	// copy the data back to host
	int size = dataSize * sizeof(int);
	cudaMemcpy(out1, d_out1, size, cudaMemcpyDeviceToHost);

	FILE * pFile;
	pFile = fopen(name,"w");
	fprintf(pFile, "Duration: %I64dus\n", duration.count());
	for(int i = 0; i < dataSize; i++) {
		fprintf(pFile, "out[%d] = %d\n", i, out1[i]);
	}	
	fclose(pFile);

}

/* ========================================================================== */

void part2sub (int numBlocks, int blockSize, int dataSize, int iteration, 
	int ** results) {

	// generate data arrays
	int *in1 = new int[dataSize] {0};
	int *in2 = new int[dataSize] {0};
	int *out1 = new int[dataSize] {0};
	in_generator(in1,dataSize,3);
	in_generator(in2,dataSize,3);

	// sorting
	int *tmp = new int[dataSize] {0};
	for (int i = 0; i < dataSize; i++) {
		if (in1[i] < in2[i]) {
			tmp[i] = in1[i];
			in1[i] = in2[i];
			in2[i] = tmp[i];
		}
	}

	// handle device (GPU) memory
	int *d_in1, *d_in2, *d_out1;
	int size = dataSize * sizeof(int);
	cudaMalloc((void **) &d_in1, size);
	cudaMalloc((void **) &d_in2, size);
	cudaMalloc((void **) &d_out1, size);
	cudaMemcpy(d_in1, in1, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_in2, in2, size, cudaMemcpyHostToDevice);

	// CUDA time
	do_action_p2(numBlocks, blockSize, dataSize, out1, d_in1, d_in2, d_out1, 1, 
		iteration, results);

	// clean up
	delete [] in1; 	delete [] in2; delete [] out1;
	cudaFree(d_in1); cudaFree(d_in2); cudaFree(d_out1);
	
}

/* ========================================================================== */

void part2abs (int numBlocks, int blockSize, int dataSize, int iteration, 
	int ** results) {

	// generate data arrays
	int *in1 = new int[dataSize] {0};
	int *in2 = new int[dataSize] {0};
	int *out1 = new int[dataSize] {0};
	in_generator(in1,dataSize,3);
	in_generator(in2,dataSize,3);

	// handle device (GPU) memory
	int *d_in1, *d_in2, *d_out1;
	int size = dataSize * sizeof(int);
	cudaMalloc((void **) &d_in1, size);
	cudaMalloc((void **) &d_in2, size);
	cudaMalloc((void **) &d_out1, size);
	cudaMemcpy(d_in1, in1, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_in2, in2, size, cudaMemcpyHostToDevice);

	// no sorting

	// CUDA time
	do_action_p2(numBlocks, blockSize, dataSize, out1, d_in1, d_in2, d_out1, 2, 
		iteration, results);

	// clean up
	delete [] in1; 	delete [] in2; delete [] out1;
	cudaFree(d_in1); cudaFree(d_in2); cudaFree(d_out1);

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
	int numBlocks = totalThreads / blockSize;

	// generate data arrays
	int *in1 = new int[*pDataSize] {0};
	int *in2 = new int[*pDataSize] {0};
	int *out1 = new int[*pDataSize] {0};
	in_generator(in1,*pDataSize,1);
	in_generator(in2,*pDataSize,2);


	// handle device (GPU) memory
	int *d_in1, *d_in2, *d_out1;
	int size = *pDataSize * sizeof(int);
	cudaMalloc((void **) &d_in1, size);
	cudaMalloc((void **) &d_in2, size);
	cudaMalloc((void **) &d_out1, size);
	cudaMemcpy(d_in1, in1, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_in2, in2, size, cudaMemcpyHostToDevice);

	// do stuff!
	for (int i = 1; i < 5; i++) {
		do_action(numBlocks, blockSize, dataSize, out1, d_in1, d_in2, d_out1, i);
	}	

	// clean up part 1
	delete [] in1; 	delete [] in2; delete [] out1;
	cudaFree(d_in1); cudaFree(d_in2); cudaFree(d_out1);

	// part 2

	// results array
	int iterations = 100;
	int** res = new int*[2];
	for(int i = 0; i < 2; ++i) {
		res[i] = new int[iterations];
	}

	for (int i = 0; i < iterations; i++) {
		part2sub(numBlocks, blockSize, dataSize, i, res);
		part2abs(numBlocks, blockSize, dataSize, i, res);
	}   
	
	FILE * pFile;
	pFile = fopen("p2results.txt","w");

	float sum = 0.0;
	for(int i = 0; i < iterations; i++) {
        sum += res[0][i];
		fprintf(pFile, "Pre-Sort[%d] = %d\n", i, res[0][i]);
    }
	printf("Pre-Sort Average = %f\n", (sum/iterations));
	fprintf(pFile, "Pre-Sort Average = %f\n", (sum/iterations));
	
	sum = 0.0;	
	for(int i = 0; i < iterations; i++) {
        sum += res[1][i];
		fprintf(pFile, "Branching[%d] = %d\n", i, res[1][i]);
    }
	printf("Branching Average = %f\n", (sum/iterations));
	fprintf(pFile, "Branching Average = %f\n", (sum/iterations));

	fclose(pFile);
	
}

/* ========================================================================== */

