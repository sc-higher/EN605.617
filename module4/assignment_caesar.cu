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
 * @param offset 
 * @param a 
 * @param b 
 * @return __global__ 
 */
__global__ 
void caesar_encrypt(int n, int offset, char *a, char *b) {
	
	for (int i = (blockIdx.x * blockDim.x) + threadIdx.x;
		 i < n;
		 i += blockDim.x * gridDim.x)
	{
		if (a[i] >= 'a' && a[i] <= 'z') {
            b[i] = a[i] + offset;
            if (b[i] > 'z') {
                b[i] = b[i] - 'z' + 'a' - 1;
            } 
        } else if (a[i] >= 'A' && a[i] <= 'Z') {
            b[i] = a[i] + offset;
            if (b[i] > 'Z') {
                b[i] = b[i] - 'Z' + 'A' - 1;
            } 
        } else {
            b[i] = a[i];
        }
	}
	
}

/* ========================================================================== */

/**
 * @brief 
 * 
 * @param n 
 * @param offset 
 * @param a 
 * @param b 
 * @return __global__ 
 */
 __global__ 
 void caesar_decrypt(int n, int offset, char *a, char *b) {
	 
	 for (int i = (blockIdx.x * blockDim.x) + threadIdx.x;
		  i < n;
		  i += blockDim.x * gridDim.x)
	 {
		 if (a[i] >= 'a' && a[i] <= 'z') {
			 b[i] = a[i] - offset;
			 if (b[i] < 'a') {
				 b[i] = b[i] + 'z' - 'a' + 1;
			 } 
		 } else if (a[i] >= 'A' && a[i] <= 'Z') {
			 b[i] = a[i] - offset;
			 if (b[i] < 'A') {
				 b[i] = b[i] + 'Z' - 'A' + 1;
			 } 
		 } else {
            b[i] = a[i];
         }
	 }
	 
 }

 /* ========================================================================== */

 /**
  * @brief 
  * 
  * @param filename 
  * @return std::string 
  */
 std::string parse_file(char * filename) {
    
    if (FILE *f = fopen(filename, "rb")) {
        fseek(f, 0, SEEK_END);
        long fsize = ftell(f);
        fseek(f, 0, SEEK_SET);
        std::string file(fsize+1, '\0');
        fread(&file[0], sizeof(char), fsize, f);
        fclose(f);
        return file;

    } else {
        printf("ERROR READING FILE!\n");
        printf("Quitting...");
        exit(0);
    }

}

/* ========================================================================== */

/**
 * @brief 
 * 
 * @param argc 
 * @param argv 
 */
std::string parse_cmdline(int argc, char **argv, int * numTh, int * blSz)
{
	if (argc != 4) {
        printf("Incorrect number of arguments!\n");
        printf("Quitting...\n");
        exit(0);
    }
    
	*numTh = atoi(argv[1]);
    *blSz = atoi(argv[2]);
    std::string inFile = parse_file(argv[3]);

	int numBlocks = *numTh / *blSz;	

	// validate command line arguments
	if (*numTh % *blSz != 0) {
		++numBlocks;
		*numTh = numBlocks * (*blSz);
		printf("Warning: Total thread count is not evenly divisible by the block size\n");
		printf("The total number of threads will be rounded up to %d\n", *numTh);
	}
	printf("\n================\n");
	printf("Total Threads: %d\n", *numTh);
	printf("Block Size: %d\n", *blSz);
    printf("\n================\n");

    return inFile;
}

/* ========================================================================== */

int main(int argc, char** argv)
{
	using namespace std;
	
	// read command line arguments
	int totalThreads = 8192;
	int blockSize = 256;
	int *pTotalThreads = &totalThreads;
	int *pBlockSize = &blockSize;
	std::string inFile = parse_cmdline(argc, argv, pTotalThreads, pBlockSize);
    int len = inFile.length();
	totalThreads = *pTotalThreads;
	blockSize = *pBlockSize;
	int numBlocks = totalThreads / blockSize;

	// pinned memory -- host
	char *h_in, *h_out;
	int size = len * sizeof(char);
    cudaMallocHost((void **) &h_in, size);
    cudaMallocHost((void **) &h_out, size);
    std::memcpy(h_in, &inFile[0], size);
    memset(h_out, '\0', size);

    printf("ORIGINAL\n");
    for(int i = 0; i < size; i++) {
		printf("%c", h_in[i]);
	}
    printf("\n");

    // pinned memory -- device
    char *d_in, *d_out;
	cudaMalloc((void **) &d_in, size);
	cudaMalloc((void **) &d_out, size);
	cudaMemcpy(d_in, h_in, size, cudaMemcpyHostToDevice);

    // kernel execution -- encrypt
    caesar_encrypt<<<numBlocks, blockSize>>>(len, 3, d_in, d_out);
    cudaMemcpy(h_out, d_out, size, cudaMemcpyDeviceToHost);

    printf("ENCRYPTED\n");
    for(int i = 0; i < size; i++) {
		printf("%c", h_out[i]);
	}	
    printf("\n");

    // kernel execution -- decrypt
    caesar_decrypt<<<numBlocks, blockSize>>>(len, 3, d_out, d_in);
    cudaMemcpy(h_out, d_in, size, cudaMemcpyDeviceToHost);

    printf("DECRYPTED\n");
    for(int i = 0; i < size; i++) {
		printf("%c", h_out[i]);
	}	
    printf("\n");

	// clean up
    cudaFreeHost(h_in); cudaFreeHost(h_out);
	cudaFree(d_in); cudaFree(d_out);
	
}

/* ========================================================================== */

