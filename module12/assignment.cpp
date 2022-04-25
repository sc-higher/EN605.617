//
// Book:      OpenCL(R) Programming Guide
// Authors:   Aaftab Munshi, Benedict Gaster, Timothy Mattson, James Fung, Dan Ginsburg
// ISBN-10:   0-321-74964-2
// ISBN-13:   978-0-321-74964-2
// Publisher: Addison-Wesley Professional
// URLs:      http://safari.informit.com/9780132488006/
//            http://www.openclprogrammingguide.com
//

// raytracer.cpp
//
//    This is a (very) simple raytracer that is intended to demonstrate 
//    using OpenCL buffers.

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <algorithm>

#include "info.hpp"

#define DEFAULT_PLATFORM 0

#define NUM_BUFFER_ELEMENTS 16

// Function to check and handle OpenCL errors
inline void 
checkErr(cl_int err, const char * name)
{
    if (err != CL_SUCCESS) {
        std::cerr << "ERROR: " <<  name << " (" << err << ")" << std::endl;
        exit(EXIT_FAILURE);
    }
}

///
//	main() for simple buffer and sub-buffer example
//
int main(int argc, char** argv)
{
    cl_int errNum;
    cl_uint numPlatforms;
    cl_uint numDevices;
    cl_platform_id * platformIDs;
    cl_device_id * deviceIDs;
    cl_context context;
    cl_program program;
    std::vector<cl_kernel> kernels;
    std::vector<cl_command_queue> queues;
    std::vector<cl_mem> buffers;

    int platform = DEFAULT_PLATFORM; 

    std::cout << "Simple buffer and sub-buffer Example" << std::endl;

    for (int i = 1; i < argc; i++)
    {
        std::string input(argv[i]);

        if (!input.compare("--platform"))
        {
            input = std::string(argv[++i]);
            std::istringstream buffer(input);
            buffer >> platform;
        }
        else
        {
            std::cout << "usage: --platform n" << std::endl;
            return 0;
        }
    }


    // First, select an OpenCL platform to run on.  
    errNum = clGetPlatformIDs(0, NULL, &numPlatforms);
    checkErr( 
        (errNum != CL_SUCCESS) ? errNum : (numPlatforms <= 0 ? -1 : CL_SUCCESS), 
        "clGetPlatformIDs"); 
 
    platformIDs = (cl_platform_id *)alloca(
            sizeof(cl_platform_id) * numPlatforms);

    std::cout << "Number of platforms: \t" << numPlatforms << std::endl; 

    errNum = clGetPlatformIDs(numPlatforms, platformIDs, NULL);
    checkErr( 
       (errNum != CL_SUCCESS) ? errNum : (numPlatforms <= 0 ? -1 : CL_SUCCESS), 
       "clGetPlatformIDs");

    std::ifstream srcFile("assignment.cl");
    checkErr(srcFile.is_open() ? CL_SUCCESS : -1, "reading assignment.cl");

    std::string srcProg(
        std::istreambuf_iterator<char>(srcFile),
        (std::istreambuf_iterator<char>()));

    const char * src = srcProg.c_str();
    size_t length = srcProg.length();

    deviceIDs = NULL;
    DisplayPlatformInfo(
        platformIDs[platform], 
        CL_PLATFORM_VENDOR, 
        "CL_PLATFORM_VENDOR");

    errNum = clGetDeviceIDs(
        platformIDs[platform], 
        CL_DEVICE_TYPE_ALL, 
        0,
        NULL,
        &numDevices);
    if (errNum != CL_SUCCESS && errNum != CL_DEVICE_NOT_FOUND)
    {
        checkErr(errNum, "clGetDeviceIDs");
    }  

    deviceIDs = (cl_device_id *)alloca(sizeof(cl_device_id) * numDevices);
    errNum = clGetDeviceIDs(
        platformIDs[platform],
        CL_DEVICE_TYPE_ALL,
        numDevices, 
        &deviceIDs[0], 
        NULL);
    checkErr(errNum, "clGetDeviceIDs");

    cl_context_properties contextProperties[] =
    {
        CL_CONTEXT_PLATFORM,
        (cl_context_properties)platformIDs[platform],
        0
    };

    context = clCreateContext(
        contextProperties, 
        numDevices,
        deviceIDs, 
        NULL,
        NULL, 
        &errNum);
    checkErr(errNum, "clCreateContext");

    // Create program from source
    program = clCreateProgramWithSource(
        context, 
        1, 
        &src, 
        &length, 
        &errNum);
    checkErr(errNum, "clCreateProgramWithSource");

    // Build program
    errNum = clBuildProgram(
        program,
        numDevices,
        deviceIDs,
        "-I.",
        NULL,
        NULL);
    if (errNum != CL_SUCCESS) 
    {
        // Determine the reason for the error
        char buildLog[16384];
        clGetProgramBuildInfo(
            program, 
            deviceIDs[0], 
            CL_PROGRAM_BUILD_LOG,
            sizeof(buildLog), 
            buildLog, 
            NULL);

            std::cerr << "Error in OpenCL C source: " << std::endl;
            std::cerr << buildLog;
            checkErr(errNum, "clBuildProgram");
    }

    // create buffers and sub-buffers
    float inputOutput[4][2][2] =  
    {
        {
            {0.0,1.0},
            {2.0,3.0}
        },
        {
            {4.0,5.0},
            {6.0,7.0}
        },
        {
            {8.0,9.0},
            {10.0,11.0}
        },
        {
            {12.0,13.0},
            {14.0,15.0}
        }
    };

    printf("-----\n");
    printf("INPUT DATA\n");
    printf("-----\n");
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 2; j++) {
            for (int k = 0; k < 2; k++) {
                printf("%.2f ", inputOutput[i][j][k]);
            }
            printf("\n");
        }
        printf("-----\n");
    }

    // create a single buffer to cover all the input data
    cl_mem main_buffer = clCreateBuffer(
        context,
        CL_MEM_READ_WRITE,
        sizeof(float) * NUM_BUFFER_ELEMENTS * numDevices,
        NULL,
        &errNum);
    checkErr(errNum, "clCreateBuffer");

    // now for all devices other than the first create a sub-buffer
    // for (unsigned int i = 0; i < numDevices; i++)
    for (unsigned int i = 0; i < 4; i++)
    {
        cl_buffer_region region = 
            {
                NUM_BUFFER_ELEMENTS/4 * i * sizeof(float), 
                NUM_BUFFER_ELEMENTS/4 * sizeof(float)
            };

        cl_mem buffer = clCreateSubBuffer(
            main_buffer,
            CL_MEM_READ_WRITE,
            CL_BUFFER_CREATE_TYPE_REGION,
            &region,
            &errNum);
        checkErr(errNum, "clCreateSubBuffer");

        buffers.push_back(buffer);
    }

    // Create command queues
    for (unsigned int i = 0; i < numDevices; i++)
    {
        cl_command_queue queue = 
            clCreateCommandQueue(
                context,
                deviceIDs[0], // was i
                CL_QUEUE_PROFILING_ENABLE,
                &errNum);
        checkErr(errNum, "clCreateCommandQueue");

        queues.push_back(queue);
    }

    for (unsigned int i = 0; i < 4; i++) // one kernel for each sub-buffer (slice)
    {
        cl_kernel kernel = clCreateKernel(
            program,
            "square",
            &errNum);
        checkErr(errNum, "clCreateKernel(square)");

        errNum = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&buffers[i]);
        checkErr(errNum, "clSetKernelArg(square)");

        kernels.push_back(kernel);
    }

    // Write input data
    const size_t buffer_origin[3] = {0,0,0};
    const size_t host_origin[3] = {0,0,0};
    const size_t region[3] = {4,4,4};  // the documentation for this seems to be off... see below

    // When defining the region, the width and height are specified in bytes
    // whereas depth is specified in number of elements. This is hinted in the 
    // documentation which says to specify depth=1 for a 2D array. Took a bit
    // to figure this out.

    clEnqueueWriteBufferRect(
        queues[numDevices - 1], 
        main_buffer, 
        CL_TRUE, 
        buffer_origin,
        host_origin,
        region,
        0,  //buffer_row_pitch
        0,  //buffer_slice_pitch
        0,  //host_row_pitch
        0,  //host_slice_pitch
        (void*)inputOutput, 
        0, 
        NULL, 
        NULL
    );

    std::vector<cl_event> events;

    // call kernel for each device
    for (unsigned int i = 0; i < kernels.size(); i++)
    {
        cl_event event;

        size_t gWI = NUM_BUFFER_ELEMENTS / kernels.size();

        errNum = clEnqueueNDRangeKernel(
            queues[0], 
            kernels[i], 
            1,
            NULL,
            (const size_t*)&gWI, 
            (const size_t*)NULL, 
            0, 
            0, 
            &event);


        events.push_back(event);

    }

    // Technically don't need this as we are doing a blocking read
    // with in-order queue.
    clWaitForEvents(events.size(), &events[0]);
    std::vector<cl_ulong> start_times;
    std::vector<cl_ulong> end_times;
    cl_ulong time_start;
    cl_ulong time_end;

    for (int i=0; i<events.size(); i++) {
        clGetEventProfilingInfo(events[i], CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL);
        clGetEventProfilingInfo(events[i], CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL);
        start_times.push_back(time_start);
        end_times.push_back(time_end);
    }

    // Read back computed data
    clEnqueueReadBufferRect(
        queues[numDevices - 1], 
        main_buffer, 
        CL_TRUE, 
        buffer_origin,
        host_origin,
        region,
        0,  //buffer_row_pitch
        0,  //buffer_slice_pitch
        0,  //host_row_pitch
        0,  //host_slice_pitch
        (void*)inputOutput, 
        0, 
        NULL, 
        NULL
    );

    printf("-----\n");
    printf("OUTPUT DATA\n");
    printf("-----\n");
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 2; j++) {
            for (int k = 0; k < 2; k++) {
                printf("%.2f ", inputOutput[i][j][k]);
            }
            printf("\n");
        }
        printf("-----\n");
    }

    float sum = 0.0;
    for (int i=0; i<4; i++) {
        sum += inputOutput[i][0][0];
    }
    float average = sum / 4.0;

    printf("\navg: %.2f\tsum: %.2f\n",average,sum);

    // event times are per kernel -- need to find min and max of all to get total execution time
    double min = *max_element(start_times.begin(), start_times.end());
    double max = *max_element(end_times.begin(), end_times.end());

    double time_total = max-min;
    printf("\nTOTAL OpenCl kernel execution time is: %0.4f milliseconds \n", time_total / 1000000.0);

    std::cout << "\nProgram completed successfully" << std::endl;

    return 0;
}
