//
// Book:      OpenCL(R) Programming Guide
// Authors:   Aaftab Munshi, Benedict Gaster, Timothy Mattson, James Fung, Dan Ginsburg
// ISBN-10:   0-321-74964-2
// ISBN-13:   978-0-321-74964-2
// Publisher: Addison-Wesley Professional
// URLs:      http://safari.informit.com/9780132488006/
//            http://www.openclprogrammingguide.com
//

// HelloWorld.cpp
//
//    This is a simple example that demonstrates basic OpenCL setup and
//    use.

#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>

#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

///
//  Constants
//
const int ARRAY_SIZE = 1000;

void test_printResults( float * , float * , float * , float * , float * );

///
//  Create an OpenCL context on the first available platform using
//  either a GPU or CPU depending on what is available.
//
cl_context CreateContext()
{
    cl_int errNum;
    cl_uint numPlatforms;
    cl_platform_id firstPlatformId;
    cl_context context = NULL;

    // First, select an OpenCL platform to run on.  For this example, we
    // simply choose the first available platform.  Normally, you would
    // query for all available platforms and select the most appropriate one.
    errNum = clGetPlatformIDs(1, &firstPlatformId, &numPlatforms);
    if (errNum != CL_SUCCESS || numPlatforms <= 0)
    {
        std::cerr << "Failed to find any OpenCL platforms." << std::endl;
        return NULL;
    }

    // Next, create an OpenCL context on the platform.  Attempt to
    // create a GPU-based context, and if that fails, try to create
    // a CPU-based context.
    cl_context_properties contextProperties[] =
    {
        CL_CONTEXT_PLATFORM,
        (cl_context_properties)firstPlatformId,
        0
    };
    context = clCreateContextFromType(contextProperties, CL_DEVICE_TYPE_GPU,
                                      NULL, NULL, &errNum);
    if (errNum != CL_SUCCESS)
    {
        std::cout << "Could not create GPU context, trying CPU..." << std::endl;
        context = clCreateContextFromType(contextProperties, CL_DEVICE_TYPE_CPU,
                                          NULL, NULL, &errNum);
        if (errNum != CL_SUCCESS)
        {
            std::cerr << "Failed to create an OpenCL GPU or CPU context." << std::endl;
            return NULL;
        }
    }

    return context;
}

///
//  Create a command queue on the first device available on the
//  context
//
cl_command_queue CreateCommandQueue(cl_context context, cl_device_id *device)
{
    cl_int errNum;
    cl_device_id *devices;
    cl_command_queue commandQueue = NULL;
    size_t deviceBufferSize = -1;

    // First get the size of the devices buffer
    errNum = clGetContextInfo(context, CL_CONTEXT_DEVICES, 0, NULL, &deviceBufferSize);
    if (errNum != CL_SUCCESS)
    {
        std::cerr << "Failed call to clGetContextInfo(...,GL_CONTEXT_DEVICES,...)";
        return NULL;
    }

    if (deviceBufferSize <= 0)
    {
        std::cerr << "No devices available.";
        return NULL;
    }

    // Allocate memory for the devices buffer
    devices = new cl_device_id[deviceBufferSize / sizeof(cl_device_id)];
    errNum = clGetContextInfo(context, CL_CONTEXT_DEVICES, deviceBufferSize, devices, NULL);
    if (errNum != CL_SUCCESS)
    {
        delete [] devices;
        std::cerr << "Failed to get device IDs";
        return NULL;
    }

    // In this example, we just choose the first available device.  In a
    // real program, you would likely use all available devices or choose
    // the highest performance device based on OpenCL device queries
    commandQueue = clCreateCommandQueue(context, devices[0], CL_QUEUE_PROFILING_ENABLE, NULL);
    if (commandQueue == NULL)
    {
        delete [] devices;
        std::cerr << "Failed to create commandQueue for device 0";
        return NULL;
    }

    *device = devices[0];
    delete [] devices;
    return commandQueue;
}

///
//  Create an OpenCL program from the kernel source file
//
cl_program CreateProgram(cl_context context, cl_device_id device, const char* fileName)
{
    cl_int errNum;
    cl_program program;

    std::ifstream kernelFile(fileName, std::ios::in);
    if (!kernelFile.is_open())
    {
        std::cerr << "Failed to open file for reading: " << fileName << std::endl;
        return NULL;
    }

    std::ostringstream oss;
    oss << kernelFile.rdbuf();

    std::string srcStdStr = oss.str();
    const char *srcStr = srcStdStr.c_str();
    program = clCreateProgramWithSource(context, 1,
                                        (const char**)&srcStr,
                                        NULL, NULL);
    if (program == NULL)
    {
        std::cerr << "Failed to create CL program from source." << std::endl;
        return NULL;
    }

    errNum = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    if (errNum != CL_SUCCESS)
    {
        // Determine the reason for the error
        char buildLog[16384];
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG,
                              sizeof(buildLog), buildLog, NULL);

        std::cerr << "Error in kernel: " << std::endl;
        std::cerr << buildLog;
        clReleaseProgram(program);
        return NULL;
    }

    return program;
}

///
//  Create memory objects used as the arguments to the kernel
//  The kernel takes three arguments: result (output), a (input),
//  and b (input)
//
bool CreateMemObjects(cl_context context, cl_mem memObjects[7],
                      float *a, float *b)
{
    memObjects[0] = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                   sizeof(float) * ARRAY_SIZE, a, NULL);
    memObjects[1] = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                   sizeof(float) * ARRAY_SIZE, b, NULL);
    memObjects[2] = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                   sizeof(float) * ARRAY_SIZE, NULL, NULL);
    memObjects[3] = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                   sizeof(float) * ARRAY_SIZE, NULL, NULL);
    memObjects[4] = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                   sizeof(float) * ARRAY_SIZE, NULL, NULL);
    memObjects[5] = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                   sizeof(float) * ARRAY_SIZE, NULL, NULL);
    memObjects[6] = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                   sizeof(float) * ARRAY_SIZE, NULL, NULL);
                                   
                                   

    if (memObjects[0] == NULL || memObjects[1] == NULL || memObjects[2] == NULL)
    {
        std::cerr << "Error creating memory objects." << std::endl;
        return false;
    }

    return true;
}

///
//  Cleanup any created OpenCL resources
//
void Cleanup(cl_context context, cl_command_queue commandQueue,
             cl_program program, cl_kernel kernel, cl_mem memObjects[7])
{
    for (int i = 0; i < 7; i++)
    {
        if (memObjects[i] != 0)
            clReleaseMemObject(memObjects[i]);
    }
    if (commandQueue != 0)
        clReleaseCommandQueue(commandQueue);

    if (kernel != 0)
        clReleaseKernel(kernel);

    if (program != 0)
        clReleaseProgram(program);

    if (context != 0)
        clReleaseContext(context);

}


/**
 * @brief 
 * 
 * @param context 
 * @param commandQueue 
 * @param program 
 * @param device 
 * @param kernel 
 * @param memObjects 
 * @return int 
 */
int test_initFramework( 
    cl_context * context,
    cl_command_queue * commandQueue,
    cl_program * program,
    cl_device_id * device,
    cl_kernel * kernel,
    cl_mem * memObjects ) {

    // Create an OpenCL context on first available platform
    *context = CreateContext();
    if (*context == NULL)
    {
        std::cerr << "Failed to create OpenCL context." << std::endl;
        return 1;
    }

    // Create a command-queue on the first device available
    // on the created context
    *commandQueue = CreateCommandQueue(*context, device);
    if (*commandQueue == NULL)
    {
        Cleanup(*context, *commandQueue, *program, *kernel, memObjects);
        return 1;
    }

    // Create OpenCL program from HelloWorld.cl kernel source
    *program = CreateProgram(*context, *device, "HelloWorld.cl");
    if (*program == NULL)
    {
        Cleanup(*context, *commandQueue, *program, *kernel, memObjects);
        return 1;
    }

    return 0;

}


/**
 * @brief 
 * 
 * @param context 
 * @param commandQueue 
 * @param program 
 * @param add_kernel 
 * @param subtract_kernel 
 * @param multiply_kernel 
 * @param modulo_kernel 
 * @param power_kernel 
 * @param memObjects 
 * @param errNum 
 * @return int 
 */
int test_createKernels(
    cl_context * context,
    cl_command_queue * commandQueue,
    cl_program * program,
    cl_kernel * add_kernel,
    cl_kernel * subtract_kernel,
    cl_kernel * multiply_kernel,
    cl_kernel * modulo_kernel,
    cl_kernel * power_kernel,
    cl_mem * memObjects,
    cl_int * errNum) {

    // Create OpenCL kernels
    // create add kernel
    *add_kernel = clCreateKernel(*program, "add_kernel", NULL);
    if (add_kernel == NULL)
    {
        std::cerr << "Failed to create kernel" << std::endl;
        Cleanup(*context, *commandQueue, *program, *add_kernel, memObjects);
        return 1;
    }
    // create sub kernel
    *subtract_kernel = clCreateKernel(*program, "subtract_kernel", NULL);
    if (subtract_kernel == NULL)
    {
        std::cerr << "Failed to create kernel" << std::endl;
        Cleanup(*context, *commandQueue, *program, *subtract_kernel, memObjects);
        return 1;
    }
    // create mult kernel
    *multiply_kernel = clCreateKernel(*program, "multiply_kernel", NULL);
    if (add_kernel == NULL)
    {
        std::cerr << "Failed to create kernel" << std::endl;
        Cleanup(*context, *commandQueue, *program, *multiply_kernel, memObjects);
        return 1;
    }
    // create mod kernel
    *modulo_kernel = clCreateKernel(*program, "modulo_kernel", NULL);
    if (add_kernel == NULL)
    {
        std::cerr << "Failed to create kernel" << std::endl;
        Cleanup(*context, *commandQueue, *program, *modulo_kernel, memObjects);
        return 1;
    }
    // create power kernel
    *power_kernel = clCreateKernel(*program, "power_kernel", NULL);
    if (add_kernel == NULL)
    {
        std::cerr << "Failed to create kernel" << std::endl;
        Cleanup(*context, *commandQueue, *program, *power_kernel, memObjects);
        return 1;
    }

    // Create memory objects that will be used as arguments to kernel
    float a[ARRAY_SIZE];
    float b[ARRAY_SIZE];
    for (int i = 0; i < ARRAY_SIZE; i++)
    {
        a[i] = (float)(i / 10.0);
        b[i] = (float)(i / 20.0);
        // a[i] = (float)(i);
        // b[i] = (float)(i * 2);
    }

    if (!CreateMemObjects(*context, memObjects, a, b))
    {
        Cleanup(*context, *commandQueue, *program, *add_kernel, memObjects);
        Cleanup(*context, *commandQueue, *program, *subtract_kernel, memObjects);
        Cleanup(*context, *commandQueue, *program, *multiply_kernel, memObjects);
        Cleanup(*context, *commandQueue, *program, *modulo_kernel, memObjects);
        Cleanup(*context, *commandQueue, *program, *power_kernel, memObjects);
        return 1;
    }

    // Set the kernel arguments (result, a, b)
    // add kernel args
    *errNum = clSetKernelArg(*add_kernel, 0, sizeof(cl_mem), &memObjects[0]);
    *errNum |= clSetKernelArg(*add_kernel, 1, sizeof(cl_mem), &memObjects[1]);
    *errNum |= clSetKernelArg(*add_kernel, 2, sizeof(cl_mem), &memObjects[2]);
    // sub kernel args
    *errNum = clSetKernelArg(*subtract_kernel, 0, sizeof(cl_mem), &memObjects[0]);
    *errNum |= clSetKernelArg(*subtract_kernel, 1, sizeof(cl_mem), &memObjects[1]);
    *errNum |= clSetKernelArg(*subtract_kernel, 2, sizeof(cl_mem), &memObjects[3]);
    // mult kernel args
    *errNum = clSetKernelArg(*multiply_kernel, 0, sizeof(cl_mem), &memObjects[0]);
    *errNum |= clSetKernelArg(*multiply_kernel, 1, sizeof(cl_mem), &memObjects[1]);
    *errNum |= clSetKernelArg(*multiply_kernel, 2, sizeof(cl_mem), &memObjects[4]);
    // mod kernel args
    *errNum = clSetKernelArg(*modulo_kernel, 0, sizeof(cl_mem), &memObjects[0]);
    *errNum |= clSetKernelArg(*modulo_kernel, 1, sizeof(cl_mem), &memObjects[1]);
    *errNum |= clSetKernelArg(*modulo_kernel, 2, sizeof(cl_mem), &memObjects[5]);
    // power kernel args
    *errNum = clSetKernelArg(*power_kernel, 0, sizeof(cl_mem), &memObjects[0]);
    *errNum |= clSetKernelArg(*power_kernel, 1, sizeof(cl_mem), &memObjects[1]);
    *errNum |= clSetKernelArg(*power_kernel, 2, sizeof(cl_mem), &memObjects[6]);
    
    if (*errNum != CL_SUCCESS)
    {
        std::cerr << "Error setting kernel arguments." << std::endl;
        Cleanup(*context, *commandQueue, *program, *add_kernel, memObjects);
        Cleanup(*context, *commandQueue, *program, *subtract_kernel, memObjects);
        Cleanup(*context, *commandQueue, *program, *multiply_kernel, memObjects);
        Cleanup(*context, *commandQueue, *program, *modulo_kernel, memObjects);
        Cleanup(*context, *commandQueue, *program, *power_kernel, memObjects);
        return 1;
    }

    return 0;

}


/**
 * @brief 
 * 
 * @param context 
 * @param commandQueue 
 * @param program 
 * @param add_kernel 
 * @param subtract_kernel 
 * @param multiply_kernel 
 * @param modulo_kernel 
 * @param power_kernel 
 * @param memObjects 
 * @param errNum 
 * @return int 
 */
int test_executeKernels(
    cl_context * context,
    cl_command_queue * commandQueue,
    cl_program * program,
    cl_kernel * add_kernel,
    cl_kernel * subtract_kernel,
    cl_kernel * multiply_kernel,
    cl_kernel * modulo_kernel,
    cl_kernel * power_kernel,
    cl_mem * memObjects,
    cl_int * errNum) {

    size_t globalWorkSize[1] = { ARRAY_SIZE };
    size_t localWorkSize[1] = { 1 };

    // initialize timing events
    cl_event add_event = 0;
    cl_event subtract_event = 0;
    cl_event multiply_event = 0;
    cl_event modulo_event = 0;
    cl_event power_event = 0;
    cl_event events [5];
    events[0] = add_event;
    events[1] = subtract_event;
    events[2] = multiply_event;
    events[3] = modulo_event;
    events[4] = power_event;

    // Queue the kernel up for execution across the array
    // enqueue add kernel
    *errNum = clEnqueueNDRangeKernel(*commandQueue, *add_kernel, 1, NULL,
                                    globalWorkSize, localWorkSize,
                                    0, NULL, &add_event);
    if (*errNum != CL_SUCCESS)
    {
        std::cerr << "Error queuing kernel for execution." << std::endl;
        Cleanup(*context, *commandQueue, *program, *add_kernel, memObjects);
        return 1;
    }
    // enqueue sub kernel
    *errNum = clEnqueueNDRangeKernel(*commandQueue, *subtract_kernel, 1, NULL,
                                    globalWorkSize, localWorkSize,
                                    0, NULL, &subtract_event);
    if (*errNum != CL_SUCCESS)
    {
        std::cerr << "Error queuing kernel for execution." << std::endl;
        Cleanup(*context, *commandQueue, *program, *subtract_kernel, memObjects);
        return 1;
    }
    // enqueue mult kernel
    *errNum = clEnqueueNDRangeKernel(*commandQueue, *multiply_kernel, 1, NULL,
                                    globalWorkSize, localWorkSize,
                                    0, NULL, &multiply_event);
    if (*errNum != CL_SUCCESS)
    {
        std::cerr << "Error queuing kernel for execution." << std::endl;
        Cleanup(*context, *commandQueue, *program, *multiply_kernel, memObjects);
        return 1;
    }
    // enqueue mod kernel
    *errNum = clEnqueueNDRangeKernel(*commandQueue, *modulo_kernel, 1, NULL,
                                    globalWorkSize, localWorkSize,
                                    0, NULL, &modulo_event);
    if (*errNum != CL_SUCCESS)
    {
        std::cerr << "Error queuing kernel for execution." << std::endl;
        Cleanup(*context, *commandQueue, *program, *modulo_kernel, memObjects);
        return 1;
    }
    // enqueue power kernel
    *errNum = clEnqueueNDRangeKernel(*commandQueue, *power_kernel, 1, NULL,
                                    globalWorkSize, localWorkSize,
                                    0, NULL, &power_event);
    if (*errNum != CL_SUCCESS)
    {
        std::cerr << "Error queuing kernel for execution." << std::endl;
        Cleanup(*context, *commandQueue, *program, *power_kernel, memObjects);
        return 1;
    }

    // end timing events
    clWaitForEvents(5, events);
    clFinish(*commandQueue);
    cl_ulong time_start [5];
    cl_ulong time_end [5];

    // add time
    clGetEventProfilingInfo(add_event, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start[0], NULL);
    clGetEventProfilingInfo(add_event, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end[0], NULL);
    // sub time
    clGetEventProfilingInfo(subtract_event, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start[1], NULL);
    clGetEventProfilingInfo(subtract_event, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end[1], NULL);
    // mult time
    clGetEventProfilingInfo(multiply_event, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start[2], NULL);
    clGetEventProfilingInfo(multiply_event, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end[2], NULL);
    // mod time
    clGetEventProfilingInfo(modulo_event, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start[3], NULL);
    clGetEventProfilingInfo(modulo_event, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end[3], NULL);
    // power time
    clGetEventProfilingInfo(power_event, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start[4], NULL);
    clGetEventProfilingInfo(power_event, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end[4], NULL);

    // event times are per kernel -- need to find min and max of all to get total execution time
    int min = *std::min_element(time_start,time_start+5);
    int max = *std::max_element(time_end,time_end+5);

    double time_add = time_end[0] - time_start[0];
    double time_subtract = time_end[1] - time_start[1];
    double time_multiply = time_end[2] - time_start[2];
    double time_modulo = time_end[3] - time_start[3];
    double time_power = time_end[4] - time_start[4];
    double time_total = max-min;
    printf("ADD execution time is: %0.4f milliseconds \n", time_add / 1000000.0);
    printf("SUBTRACT execution time is: %0.4f milliseconds \n", time_subtract / 1000000.0);
    printf("MULTIPLY execution time is: %0.4f milliseconds \n", time_multiply / 1000000.0);
    printf("MODULO execution time is: %0.4f milliseconds \n", time_modulo / 1000000.0);
    printf("POWER execution time is: %0.4f milliseconds \n", time_power / 1000000.0);
    printf("TOTAL OpenCl kernel execution time is: %0.4f milliseconds \n", time_total / 1000000.0);

    return 0;

}


/**
 * @brief 
 * 
 * @param context 
 * @param commandQueue 
 * @param program 
 * @param add_kernel 
 * @param subtract_kernel 
 * @param multiply_kernel 
 * @param modulo_kernel 
 * @param power_kernel 
 * @param memObjects 
 * @param errNum 
 * @return int 
 */
int test_getData(
    cl_context * context,
    cl_command_queue * commandQueue,
    cl_program * program,
    cl_kernel * add_kernel,
    cl_kernel * subtract_kernel,
    cl_kernel * multiply_kernel,
    cl_kernel * modulo_kernel,
    cl_kernel * power_kernel,
    cl_mem * memObjects,
    cl_int * errNum) {

    // Create host arrays to store data
    float add_result[ARRAY_SIZE];
    float subtract_result[ARRAY_SIZE];
    float multiply_result[ARRAY_SIZE];
    float modulo_result[ARRAY_SIZE];
    float power_result[ARRAY_SIZE];

    // Read the output buffer back to the Host
    *errNum = clEnqueueReadBuffer(*commandQueue, memObjects[2], CL_TRUE,
                                 0, ARRAY_SIZE * sizeof(float), add_result,
                                 0, NULL, NULL);
    if (*errNum != CL_SUCCESS)
    {
        std::cerr << "Error reading result buffer." << std::endl;
        Cleanup(*context, *commandQueue, *program, *add_kernel, memObjects);
        return 1;
    }

    *errNum = clEnqueueReadBuffer(*commandQueue, memObjects[3], CL_TRUE,
                                 0, ARRAY_SIZE * sizeof(float), subtract_result,
                                 0, NULL, NULL);
    if (*errNum != CL_SUCCESS)
    {
        std::cerr << "Error reading result buffer." << std::endl;
        Cleanup(*context, *commandQueue, *program, *subtract_kernel, memObjects);
        return 1;
    }

    *errNum = clEnqueueReadBuffer(*commandQueue, memObjects[4], CL_TRUE,
                                 0, ARRAY_SIZE * sizeof(float), multiply_result,
                                 0, NULL, NULL);
    if (*errNum != CL_SUCCESS)
    {
        std::cerr << "Error reading result buffer." << std::endl;
        Cleanup(*context, *commandQueue, *program, *multiply_kernel, memObjects);
        return 1;
    }

    *errNum = clEnqueueReadBuffer(*commandQueue, memObjects[5], CL_TRUE,
                                 0, ARRAY_SIZE * sizeof(float), modulo_result,
                                 0, NULL, NULL);
    if (*errNum != CL_SUCCESS)
    {
        std::cerr << "Error reading result buffer." << std::endl;
        Cleanup(*context, *commandQueue, *program, *modulo_kernel, memObjects);
        return 1;
    }

    *errNum = clEnqueueReadBuffer(*commandQueue, memObjects[6], CL_TRUE,
                                 0, ARRAY_SIZE * sizeof(float), power_result,
                                 0, NULL, NULL);
    if (*errNum != CL_SUCCESS)
    {
        std::cerr << "Error reading result buffer." << std::endl;
        Cleanup(*context, *commandQueue, *program, *power_kernel, memObjects);
        return 1;
    }

    test_printResults(add_result, subtract_result, multiply_result, modulo_result, power_result);

    return 0;

}


/**
 * @brief 
 * 
 * @param add_result 
 * @param subtract_result 
 * @param multiply_result 
 * @param modulo_result 
 * @param power_result 
 */
void test_printResults(
    float * add_result,
    float * subtract_result,
    float * multiply_result,
    float * modulo_result,
    float * power_result) {

    int N = 100;
    
    // Output the result buffer
    std::cout << "-----ADD-----" << std::endl;
    for (int i = 0; i < N; i++)
    {
        printf("%0.2f ", add_result[i]);
    }
    std::cout << std::endl;

    std::cout << "-----SUBTRACT-----" << std::endl;
    for (int i = 0; i < N; i++)
    {
        printf("%0.2f ", subtract_result[i]);
    }
    std::cout << std::endl;

    std::cout << "-----MULTIPLY-----" << std::endl;
    for (int i = 0; i < N; i++)
    {
        printf("%0.2f ", multiply_result[i]);
    }
    std::cout << std::endl;
    
    std::cout << "-----MODULO-----" << std::endl;
    for (int i = 0; i < N; i++)
    {
        printf("%0.2f ", modulo_result[i]);
    }
    std::cout << std::endl;

    std::cout << "-----POWER-----" << std::endl;
    for (int i = 0; i < N; i++)
    {
        printf("%0.2f ", power_result[i]);
    }
    std::cout << std::endl;
    
    std::cout << "------------------------" << std::endl;
    std::cout << "Executed program succesfully" << std::endl;    

}


///
//	main() for HelloWorld example
//
int main(int argc, char** argv)
{
    cl_context context = 0;
    cl_command_queue commandQueue = 0;
    cl_program program = 0;
    cl_device_id device = 0;
    cl_kernel kernel = 0;
    cl_kernel add_kernel = 0;
    cl_kernel subtract_kernel = 0;
    cl_kernel multiply_kernel = 0;
    cl_kernel modulo_kernel = 0;
    cl_kernel power_kernel = 0;
    cl_mem memObjects[7] = { 0, 0, 0, 0, 0, 0, 0};
    cl_int errNum;

    test_initFramework(
        &context,
        &commandQueue,
        &program,
        &device,
        &kernel,
        memObjects
    );

    test_createKernels(
        &context,
        &commandQueue,
        &program,
        &add_kernel,
        &subtract_kernel,
        &multiply_kernel,
        &modulo_kernel,
        &power_kernel,
        memObjects,
        &errNum
    );

    test_executeKernels(
        &context,
        &commandQueue,
        &program,
        &add_kernel,
        &subtract_kernel,
        &multiply_kernel,
        &modulo_kernel,
        &power_kernel,
        memObjects,
        &errNum
    );

    test_getData(
        &context,
        &commandQueue,
        &program,
        &add_kernel,
        &subtract_kernel,
        &multiply_kernel,
        &modulo_kernel,
        &power_kernel,
        memObjects,
        &errNum
    );

    Cleanup(context, commandQueue, program, add_kernel, memObjects);     

    return 0;

}