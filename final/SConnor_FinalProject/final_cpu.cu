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

/* ========================================================================== */

/**
 * @brief 
 * 
 * @param total_threads 
 * @param block_size 
 * @param data_size 
 * @param nx number of space steps
 * @param nt number of time steps
 * @param dx size of space step
 * @param dt size of time step
 * @param alpha thermal diffusivity constant
 */
 void execute_ftcs(
    int data_size, 
    int nx,
    int nt,
    float dx,
    float dt,
    float alpha) {

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

    fclose(p_file);
		
	// clean up
	delete [] T;
			
}

/* ========================================================================== */

int main(int argc, char** argv)
{
	using namespace std;

    // CFD Parameters
    int nx = NX;
    int nt = NT;
    float alpha = ALPHA;
    float t_max = T_MAX;
    float length = LENGTH;
    int data_size = nx*nt;   
    float dx = length / (nx-1);
    float dt = t_max / (nt-1);

    // execute the FTCS method on host
	execute_ftcs(data_size, nx, nt, dx, dt, alpha);
    
}

/* ========================================================================== */

/* APPENDIX */

// cudaError_t err = cudaGetLastError();
    // if ( err != cudaSuccess ) {
    // 	printf("CUDA Error: %s\n", cudaGetErrorString(err)); 
    // 	// Possibly: exit(-1) if program cannot continue....
    // }