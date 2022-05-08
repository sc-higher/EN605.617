/*
Sean Connor - May 2022
605.617 Final Project
*/

/* ========================================================================== */

#include <stdio.h>
#include <cmath>

#ifndef M_PI
    #define M_PI 3.14159265358979323846
#endif

// CFD Default Parameters
#define NX 101
#define NT 10001
#define ALPHA 0.1
#define GAMMA 1.4
#define T_MAX 5.0
#define LENGTH 1.0

// CUDA Default Parameters
#define TOTAL_THREADS 8192
#define BLOCK_SIZE 256

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
 * @param total_threads 
 * @param block_size 
 * @param data_size 
 * @param nx 
 * @param nt 
 * @param dx 
 * @param dt 
 * @param alpha 
 */
 void execute_ftcs(
    int total_threads,
    int block_size,
    int data_size, 
    int nx,
    int nt,
    float dx,
    float dt,
    float alpha,
    float * p_timer) {

	int size = data_size * sizeof(float);
	int num_blocks = total_threads / block_size;

	// set constants
    float gamma = GAMMA;
    float r = dt/(4*dx);
    
    // allocate data arrays
    // Physical Results 
	float *rho = new float[data_size] {0.0};
    float *u = new float[data_size] {0.0};
    float *P = new float[data_size] {0.0};

	float *d_rho, *d_u, *d_P;
	cudaMalloc((void **) &d_rho, size);
    cudaMalloc((void **) &d_u, size);
    cudaMalloc((void **) &d_P, size);
	cudaMemcpy(d_rho, rho, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_u, u, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_P, P, size, cudaMemcpyHostToDevice);

    // Physical Intermediates
    float *rho1 = new float[data_size] {0.0};
    float *u1 = new float[data_size] {0.0};
    float *P1 = new float[data_size] {0.0};
    float *E1 = new float[data_size] {0.0};
    float *c1 = new float[data_size] {0.0};

    

    // F and F Intermediates
    float F1n = new float[data_size] {0.0};
    float F1p = new float[data_size] {0.0};
    float F2n = new float[data_size] {0.0};
    float F2p = new float[data_size] {0.0};
    float F3n = new float[data_size] {0.0};
    float F3p = new float[data_size] {0.0};
    float F1n1 = new float[data_size] {0.0};
    float F1p1 = new float[data_size] {0.0};
    float F2n1 = new float[data_size] {0.0};
    float F2p1 = new float[data_size] {0.0};
    float F3n1 = new float[data_size] {0.0};
    float F3p1 = new float[data_size] {0.0};

    // Q Intermediates
    float Q1 = new float[data_size] {0.0};
    float Q2 = new float[data_size] {0.0};
    float Q3 = new float[data_size] {0.0};
    float R = new float[data_size] {0.0};

    // Area
    float S = new float[nx] {0.0};
    float dS = new float[nx] {0.0};


    // set initial condition
    float P_init = 1.0/gamma;
    for (int i=0; i<nx-1; i++) {
        rho[i] = 1.0;
        u[i] = 0.8;
        P[i] = P_init;
    }

    // set boundary conditions
    for (int i=0; i<data_size; i += nx) {
        rho[i] = 1.0;
        rho[i+nx-1] = 1.0;
        u[i] =  0.8;
        u[i+nx-1] = 0.8;
        P[i] = P_init;
        P[i+nx-1] = P_init;
    }

    // initialize E, c, and Qn
    for (int i=0; i<data_size; i++) {
        E[i] = P[i] / (rho[i] * (gamma-1)) + pow(u[i],2) / 2; 
        c[i] = sqrt((gamma*P[i]) / rho[i]);
        Q1[i] = rho[i];
        Q2[i] = rho[i] * u[i];
        Q3[i] = rho[i] * E[i];
    }

    // initialize Fn and Fp

    // make X and t linspaces
    float x_linspace[nx] = {0.0};
    float t_linspace[nt] = {0.0};
    for (int i=1; i<nx; i++) {
        x_linspace[i] = x_linspace[i-1] + dx;
    }
    for (int i=1; i<nt; i++) {
        t_linspace[i] = t_linspace[i-1] + dt;
    }

    // set area
    for (int i=0; i<nx-1; i++) {
        S[i] = 1 + pow((2.2*(x_linspace[i]-1.5)),2);
        dS[i] = 4.4 * (x_linspace[i] - 1.5);
    }


    // set up CUDA timing
	// https://developer.nvidia.com/blog/how-implement-performance-metrics-cuda-cc/
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
    cudaEventRecord(start);
	
	// allocate device and copy from host
	float *d_T;
	cudaMalloc((void **) &d_T, size);
	cudaMemcpy(d_T, T, size, cudaMemcpyHostToDevice);	
	
    // all data kept in a single 1D array - need to index each time step
    int idx_start = 0;
    int idx_stop = 0;
    float r = alpha*dt/(dx*dx);

    // execute each time step in CUDA
    for (int i=0; i<nt; i++) {
        idx_start = i * nx + 1;
        idx_stop = idx_start + nx - 3;
        ftcs<<<num_blocks, block_size>>>(data_size, idx_start, idx_stop, r, d_T);
    }
	
	// copy data back to host
	cudaMemcpy(T, d_T, size, cudaMemcpyDeviceToHost);

    // stop CUDA timing
    cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(p_timer, start, stop);

    // write results array to file (t,x,T) 
    // note gnuplot format requires a newline between 'blocks' for 3D data
    FILE * p_file;
    p_file = fopen("results.dat","w");
    int indx = 0;
    for (int i=0; i<nt; i++) {
        for (int j=0; j<nx; j++) {
            indx = i*nx + j;
            fprintf(p_file,"%.4f,%.2f,%.2f\n",t_linspace[i],x_linspace[j],T[indx]);
        }
        fprintf(p_file,"\n");
    }

    printf("Elapsed Time (ms): %.2f\n", *p_timer);
    
    // Alternate format matrix-style
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
	
	// CUDA Parameters
	int total_threads = TOTAL_THREADS;
	int block_size = BLOCK_SIZE;
    float timer = 0.0;
	float *p_timer = &timer;

    // CFD Parameters
    int nx = NX;
    int nt = NT;
    float alpha = ALPHA;
    float t_max = T_MAX;
    float length = LENGTH;
    int data_size = nx*nt;   
    float dx = length / (nx-1);
    float dt = t_max / (nt-1);

	execute_ftcs(total_threads, block_size, data_size, nx, nt, dx, dt, alpha, p_timer);
    
}

/* ========================================================================== */

/* APPENDIX */

// cudaError_t err = cudaGetLastError();
    // if ( err != cudaSuccess ) {
    // 	printf("CUDA Error: %s\n", cudaGetErrorString(err)); 
    // 	// Possibly: exit(-1) if program cannot continue....
    // }