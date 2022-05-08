/*
Sean Connor - May 2022
605.617 Final Project
*/

/* ========================================================================== */

#include <stdio.h>
#include <chrono>

#ifndef M_PI
    #define M_PI 3.14159265358979323846
#endif

// CFD Default Parameters
#define NX 101
#define NT 10001
#define ALPHA 0.1
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

	// allocate host data arrays 
	float *T = new float[data_size] {0.0};

    // set initial condition
    for (int i=1; i<nx-1; i++) {
        T[i] = 1.0;
    }

    // set boundary conditions
    float time = 0.0;
    for (int i=0; i<data_size; i += nx) {
        T[i] = 1 + sin(M_PI*time); //1.0;
        T[i + nx - 1] = 5.0;
        time += dt;
    }

    // start timer
    auto start = std::chrono::high_resolution_clock::now();
	
    // all data kept in a single 1D array - need to index each time step
    int idx_start = 0;
    int idx_stop = 0;
    float r = alpha*dt/(dx*dx);

    // execute each time step in CPU
    for (int i=0; i<nt; i++) {
        idx_start = i * nx + 1;
        idx_stop = idx_start + nx - 3;

        for (int j=idx_start; j<=idx_stop; j++) {
            T[j+nx] = r*T[j-1] + (1-2*r)*T[j] + r*T[j+1];
        }
        
    }

    // stop timer
    auto stop = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

    printf("Duration: %.2fms\n", (float)duration.count()/1000);

    // make X and t linspaces for grid study
    float x_linspace[NX] = {0.0};
    float t_linspace[NT] = {0.0};
    for (int i=1; i<nx; i++) {
        x_linspace[i] = x_linspace[i-1] + dx;
    }
    for (int i=1; i<nt; i++) {
        t_linspace[i] = t_linspace[i-1] + dt;
    }

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
    
    // Alternate format matrix-style
    // for (int i=0; i<data_size; i++) {
    //     if (i % *p_nx == 0) {
    //         fprintf(p_file, "\n");
    //     }
    //     fprintf(p_file,"%.2f ",T[i]);
    // }

    fclose(p_file);
		
	// clean up
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