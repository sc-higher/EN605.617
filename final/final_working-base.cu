/*
Sean Connor - May 2022
605.617 Final Project
*/

/* ========================================================================== */

#include <stdio.h>

#ifndef M_PI
    #define M_PI 3.14159265358979323846
#endif

#define NX 101
#define NT 10001

/* ========================================================================== */

/**
 * @brief 
 * 
 * @param data_size : length of data array
 * @param start : starting index for calculation
 * @param stop : ending index for calculation
 * @param r : ftcs equation constant
 * @param T : data array
 * @return __global__ 
 */
 __global__ 
 void ftcs(int data_size, int start, int stop, float r, float * T) {
	
	for (int i = (blockIdx.x * blockDim.x) + threadIdx.x;
		 i < data_size;
		 i += blockDim.x * gridDim.x)
	{
		if ( (i>=start) && (i<=stop) ) {
            
            int nx = stop-start+3; // this is the length of each 'row'
            T[i+nx] = r*T[i-1] + (1-2*r)*T[i] + r*T[i+1];

        }
	}
	
}

/* ========================================================================== */

/**
 * @brief 
 * 
 * @param argc 
 * @param argv 
 */
void parse_cmdline(
    int argc, 
    char **argv, 
    int * numTh, 
    int * blSz, 
    int * nx, 
    int * nt, 
    float * alpha) {

	if (argc >= 2) {
		*numTh = atoi(argv[1]);
	}
	if (argc >= 3) {
		*blSz = atoi(argv[2]);
	}
	if (argc >= 4) {
		*nx = atoi(argv[3]);
	}
    if (argc >= 5) {
		*nt = atoi(argv[4]);
	}
    if (argc >= 6) {
		*alpha = atoi(argv[5]);
	}

	int num_blocks = *numTh / *blSz;	

	// validate command line arguments
	if (*numTh % *blSz != 0) {
		++num_blocks;
		*numTh = num_blocks * (*blSz);
		printf("Warning: Total thread count is not evenly divisible by the block size\n");
		printf("The total number of threads will be rounded up to %d\n", *numTh);
	}
	printf("================\n");
	printf("Total Threads: %d\n", *numTh);
	printf("Block Size: %d\n", *blSz);
	printf("Data Size: %d\n", (*nx)*(*nt) );
	printf("================\n");
}

/* ========================================================================== */

/**
 * @brief 
 * 
 * @param in1 
 * @param in2 
 * @param out1 
 * @param p_total_threads 
 * @param p_block_size 
 * @param pDataSize 
 */
 void global_test(
    int * p_total_threads,
    int * p_block_size,
    int data_size, 
	float * p_timer,
    int * p_nx,
    int * p_nt,
    float dx,
    float dt,
    float * p_alpha) {

	int size = data_size * sizeof(float);
	int num_blocks = *p_total_threads / *p_block_size;

	// allocate host data arrays 
	float *T = new float[data_size] {0.0};

    // set initial condition
    for (int i=1; i<*p_nx-1; i++) {
        T[i] = 1.0;
    }

    // set boundary conditions
    float time = 0.0;
    for (int i=0; i<data_size; i += *p_nx) {
        T[i] = 1 + sin(M_PI*time); //1.0;
        T[i + (*p_nx) - 1] = 5.0;
        time += dt;
    }

    // 1 + sin(pi*t)
	
	// generate data and allocate device data (global)
	float *d_T;
	cudaMalloc((void **) &d_T, size);
	cudaMemcpy(d_T, T, size, cudaMemcpyHostToDevice);

	// set up CUDA timing
	// https://developer.nvidia.com/blog/how-implement-performance-metrics-cuda-cc/
	cudaEvent_t timer_start, timer_stop;
	cudaEventCreate(&timer_start);
	cudaEventCreate(&timer_stop);

	// execute kernels and time
	cudaEventRecord(timer_start);	
	
    int idx_start = 0;
    int idx_stop = 0;
    float r = *p_alpha*dt/(dx*dx);

    for (int i=0; i<(*p_nt); i++) {
        idx_start = i*(*p_nx) + 1;
        idx_stop = idx_start + (*p_nx) - 3;
        ftcs<<<num_blocks, *p_block_size>>>(data_size, idx_start, idx_stop, r, d_T);
    }

	cudaEventRecord(timer_stop);
	cudaEventSynchronize(timer_stop);
	cudaEventElapsedTime(p_timer, timer_start, timer_stop);	
	
	// print statement to verify accuracy of results
	cudaMemcpy(T, d_T, size, cudaMemcpyDeviceToHost);
	// for (int j = 0; j < data_size; j+=128) {
	// 	printf("T[%d] = %d\n", j, T[j]);
	// }

    // for (int i=0; i<data_size; i++) {
    //     if (i % *p_nx == 0) {
    //         printf("\n");
    //     }
    //     printf("%.2f ",T[i]);
    // }

    // make X and t linspaces
    float x_linspace[NX] = {0.0};
    float t_linspace[NT] = {0.0};

    for (int i=1; i<NX; i++) {
        x_linspace[i] = x_linspace[i-1] + dx;
    }

    for (int i=1; i<NT; i++) {
        t_linspace[i] = t_linspace[i-1] + dt;
    }


    // write results array to file
    FILE * p_file;
    p_file = fopen("results.dat","w");
    int indx = 0;
    for (int i=0; i<NT; i++) {
        for (int j=0; j<NX; j++) {
            indx = i*NX + j;
            fprintf(p_file,"%.4f,%.2f,%.2f\n",t_linspace[i],x_linspace[j],T[indx]);
        }
        fprintf(p_file,"\n");
    }
    
    // for (int i=0; i<data_size; i++) {
    //     if (i % *p_nx == 0) {
    //         fprintf(p_file, "\n");
    //     }
    //     fprintf(p_file,"%.2f ",T[i]);
    // }

    fclose(p_file);
		
	// clean up
	cudaFree(d_T);
	delete [] T;
			
}

/* ========================================================================== */

int main(int argc, char** argv)
{
	using namespace std;
	
	// read command line arguments
	int total_threads = 8192;
	int block_size = 256;
    int nx = NX;
    int nt = NT;
    float alpha = 0.1;
    int data_size = nx*nt;
    float timer = 0.0;

    // improve arg parsing later
    float t_max = 5.0;
    float length = 1.0;
    float dx = length / (nx-1);
    float dt = t_max / (nt-1);

	int *p_total_threads = &total_threads;
	int *p_block_size = &block_size;
	int *p_nx = &nx;
    int *p_nt = &nt;
    float *p_alpha = &alpha;
	float *p_timer = &timer;

	parse_cmdline(argc, argv, p_total_threads, p_block_size, p_nx, p_nt, p_alpha);
	total_threads = *p_total_threads;
	block_size = *p_block_size;
	nx = *p_nx;
    nt = *p_nt;
    alpha = *p_alpha;
    data_size = nx*nt;

	global_test(p_total_threads, p_block_size, data_size, p_timer, p_nx, p_nt, dx, dt, p_alpha);

    
    
}

/* ========================================================================== */

/* APPENDIX */

// cudaError_t err = cudaGetLastError();
    // if ( err != cudaSuccess ) {
    // 	printf("CUDA Error: %s\n", cudaGetErrorString(err)); 
    // 	// Possibly: exit(-1) if program cannot continue....
    // }


// // test harness
// int iterations = 10;
// float** res = new float*[3];
// for(int i = 0; i < 3; ++i) {
// 	res[i] = new float[iterations];
// }
// for (int i = 0; i < iterations; i++) {
// 	// global memory test
// 	global_test(p_total_threads, p_block_size, pDataSize, p_timer);
// 	res[2][i] = *p_timer;
// }
// // write results array to file
// FILE * p_file;
// p_file = fopen("results.txt","w");
// float sum = 0.0;
// for(int i = 0; i < iterations; i++) {
//     sum += res[2][i];
// 	fprintf(p_file, "Global Memory[%d] = %f\n", i, res[2][i]);
// }
// printf("Global Memory Average = %f\n", (sum/iterations));
// fprintf(p_file, "Global Memory Average = %f\n", (sum/iterations));
// fclose(p_file);