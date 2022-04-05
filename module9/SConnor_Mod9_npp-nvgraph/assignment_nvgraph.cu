#include <random>
#include "nvgraph.h"

static std::random_device rd;
static std::mt19937 rng{rd()};
static std::uniform_int_distribution<int> case2_val(0,3);
static std::uniform_int_distribution<int> case3_val(0,100);
static std::uniform_int_distribution<int> case4_val(0,999);
static std::uniform_real_distribution<float> case5_val(0.0, 1.0);

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
        case 4:
			for (int i = 0; i < len; i++) {
				array[i] = case4_val(rng);
			}
			break;
		default:
			break;
	}
	
} /* END inGenerator() */

/* ========================================================================== */

/**
 * @brief 
 * 
 * @param toNode 
 * @param fromNode 
 * @param weights 
 * @param len 
 */
void generateData(int * toNode , int * fromNode , int len) {
    inGenerator(toNode, len, 4); // 0-1000
    inGenerator(fromNode, len, 4); // 0-1000
} /* END generateData() */

/* ========================================================================== */

void executePageRank(int len) {

    // create data arrays
    int* toNode = new int[len];
    int* fromNode = new int[len];
    float* weights = new float[len];

    // fill with random data 
    generateData( toNode, fromNode , len );
    for (int i = 0; i < len; i++) {
        weights[i] = case5_val(rng);
    }

    //
    int numVertex = 1000;
    int numEdge = len;
    float *bookmark = new float[numVertex] {0};
    float *pr_1 = (float*)malloc(numVertex*sizeof(float));

    // nvgraph variables
    nvgraphGraphDescr_t graph;
    nvgraphHandle_t handle;
    nvgraphCOOTopology32I_t COO_input;
    nvgraphCSRTopology32I_t CSR_output;

    COO_input = (nvgraphCOOTopology32I_t) malloc(sizeof(struct
        nvgraphCOOTopology32I_st));
    
    COO_input->nvertices = numVertex; 
    COO_input->nedges = numEdge;
    COO_input->tag = NVGRAPH_UNSORTED;
    float *src_weights_d;

    cudaDataType_t* vertex_dimT = (cudaDataType_t*)malloc(2*sizeof(cudaDataType_t));
    vertex_dimT[0] = CUDA_R_32F; 
    vertex_dimT[1] = CUDA_R_32F; 
    vertex_dimT[2] = CUDA_R_32F;

    // Allocate source data
    cudaMalloc( (void**)&(COO_input->source_indices), numEdge*sizeof(int));
    cudaMalloc( (void**)&(COO_input->destination_indices), numEdge*sizeof(int));
    cudaMalloc( (void**)&src_weights_d, numEdge*sizeof(float));

    // Copy source data    
    cudaMemcpy(COO_input->destination_indices, fromNode, numEdge*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(COO_input->source_indices, toNode, numEdge*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(src_weights_d, weights, numEdge*sizeof(float), cudaMemcpyHostToDevice);

    // Allocate destination data
    float *dst_weights_d;
    CSR_output = (nvgraphCSRTopology32I_t) malloc(sizeof(struct
    nvgraphCSRTopology32I_st));
    cudaMalloc( (void**)&(CSR_output->source_offsets), (numVertex+1)*sizeof(int));
    cudaMalloc( (void**)&(CSR_output->destination_indices), numEdge*sizeof(int));
    cudaMalloc( (void**)&dst_weights_d, numEdge*sizeof(float));

    // Starting nvgraph to convert
    nvgraphCreate(&handle);
    cudaDataType_t edge_dimT = CUDA_R_32F;
    nvgraphConvertTopology(handle, NVGRAPH_COO_32, COO_input, src_weights_d, &edge_dimT, NVGRAPH_CSR_32, CSR_output, dst_weights_d);

    // Starting nvgraph to PageRank
    nvgraphCreateGraphDescr (handle, &graph);
    
    // Set graph connectivity and properties (tranfers)
    nvgraphSetGraphStructure(handle, graph, (void*)CSR_output, NVGRAPH_CSR_32);
    nvgraphAllocateVertexData(handle, graph, 2, vertex_dimT);
    nvgraphAllocateEdgeData (handle, graph, 1, &edge_dimT);

    nvgraphSetVertexData(handle, graph, (void*)bookmark , 0);
    nvgraphSetVertexData(handle, graph, (void*)pr_1 , 1);
    nvgraphSetEdgeData(handle, graph, dst_weights_d, 0);

    float alpha1 = 0.9f; void *alpha1_p = (void *) &alpha1;
    nvgraphPagerank(handle, graph, 0, alpha1_p, 0, 0, 1, 0.0f, 0);

    // Get result
    nvgraphGetVertexData(handle, graph, pr_1, 1);

    for(int i=0; i<5; i++) {
        printf("res[%d] = %f\n",i,pr_1[i]);
    }

    // Free memory
    nvgraphDestroy(handle);
    nvgraphDestroyGraphDescr(handle, graph);
    free(pr_1);
    cudaFree(COO_input->destination_indices);
    cudaFree(COO_input->source_indices);
    cudaFree(CSR_output->source_offsets);
    cudaFree(CSR_output->destination_indices);
    cudaFree(src_weights_d);
    cudaFree(dst_weights_d);
    free(COO_input);
    free(CSR_output);

    printf("SUCCESS!\n");

}

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
    executePageRank(*pDataSize);   
    // testFramework(pDataSize, pTimer);
    
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