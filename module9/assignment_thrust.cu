#include <random>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

static std::random_device rd;
static std::mt19937 rng{rd()};
static std::uniform_int_distribution<int> case2_val(0,3);
static std::uniform_int_distribution<int> case3_val(0,100);

/* ========================================================================== */

/**
 * @brief 
 * 
 * @param argc 
 * @param argv 
 */
void parseCmdline(int argc, char **argv, int * numTh, int * blSz, int * dataSz)
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

} /* END parseCmdline() */

/* ========================================================================== */

/**
 * @brief 
 * 
 * @param array Point to array to be filled.
 * @param len Length of array to be filled.
 * @param type Type 1 = Increment 0 - len; Type 2 = 0-3 random; Type 3 = random
 */
 void inGenerator(int * array, int len, int type) {

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
	
} /* END inGenerator() */

/* ========================================================================== */

/**
 * @brief 
 * 
 * @param size 
 * @param pTimer 
 * @param test 
 */
void performArithmetic( int size , float * pTimer , bool test = false ) {

    // generate random data on host
    int *data = new int[size] {0};
	inGenerator(data,size,3); // type 3 = random 0-100

    // set up CUDA timing
	// https://developer.nvidia.com/blog/how-implement-performance-metrics-cuda-cc/
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	// begin timer
	cudaEventRecord(start);
    
    // copy data to device and allocate output vectors
    thrust::device_vector<int> devIn(data, data + size);
    thrust::device_vector<int> devOutAdd(size);
    thrust::host_vector<int> hostOutAdd(size);
    thrust::device_vector<int> devOutSub(size);
    thrust::host_vector<int> hostOutSub(size);
    thrust::device_vector<int> devOutMult(size);
    thrust::host_vector<int> hostOutMult(size);
    thrust::device_vector<int> devOutMod(size);
    thrust::host_vector<int> hostOutMod(size);

    // define the four operations 
    thrust::plus<int> addOp;
    thrust::minus<int> subOp;
    thrust::multiplies<int> multOp;
    thrust::modulus<int> modOp;

    // perform the operations
    thrust::transform(devIn.begin(), devIn.end(), devIn.begin(), devOutAdd.begin(), addOp);
    thrust::transform(devIn.begin(), devIn.end(), devIn.begin(), devOutSub.begin(), subOp);
    thrust::transform(devIn.begin(), devIn.end(), devIn.begin(), devOutMult.begin(), multOp);
    thrust::transform(devIn.begin(), devIn.end(), devIn.begin(), devOutMod.begin(), modOp);

    thrust::copy(devOutAdd.begin(), devOutAdd.end(), hostOutAdd.begin());
    thrust::copy(devOutSub.begin(), devOutSub.end(), hostOutSub.begin());
    thrust::copy(devOutMult.begin(), devOutMult.end(), hostOutMult.begin());
    thrust::copy(devOutMod.begin(), devOutMod.end(), hostOutMod.begin());

    // stop timer
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(pTimer, start, stop);

    // print statement if 'test'
    if ( test ) {
        printf("\nResults Validation \n");
        for (int i=0; i<4; i++) {
            printf("orig[%d]: %d\n",i,data[i]);
            printf("add[%d]: %d\n",i,hostOutAdd[i]);
            printf("sub[%d]: %d\n",i,hostOutSub[i]);
            printf("mult[%d]: %d\n",i,hostOutMult[i]);
            printf("mod[%d]: %d\n",i,hostOutMod[i]);
            printf("---\n");
        }
    } 

    // clean up
	delete [] data;

} /* END performArithmetic() */

/* ========================================================================== */

/**
 * @brief 
 * 
 * @param pDataSize 
 * @param pTimer 
 */
 void testFramework(int * pDataSize, float * pTimer) {

    // test harness
	int iterations = 10;
	float** res = new float*[1];
	for(int i = 0; i < 1; ++i) {
		res[i] = new float[iterations];
	}

	for (int i = 0; i < iterations; i++) {

		// thrust arithmetic test
		performArithmetic(*pDataSize, pTimer);
		res[0][i] = *pTimer;

	}

	// write results array to file
	FILE * pFile;
	pFile = fopen("results.txt","w");

	float sum = 0.0;
	for(int i = 0; i < iterations; i++) {
        sum += res[0][i];
		fprintf(pFile, "Thrust[%d] = %f\n", i, res[0][i]);
    }
	printf("\nAverage Time (ms) = %f\n", (sum/iterations));
	fprintf(pFile, "Average Time (ms) = %f\n", (sum/iterations));

	fclose(pFile);
    
} /* END testFramework() */

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

	parseCmdline(argc, argv, pTotalThreads, pBlockSize, pDataSize);
    performArithmetic(*pDataSize, pTimer, true);
    testFramework(pDataSize, pTimer);
    
    return 0;
	
} /* END main() */

/* ========================================================================== */

/* APPENDIX */

// CUDA ERROR CHECKING 

	// cudaError_t err = cudaGetLastError();
    // if ( err != cudaSuccess ) {
    // 	printf("CUDA Error: %s\n", cudaGetErrorString(err)); 
    // 	// Possibly: exit(-1) if program cannot continue....
    // }