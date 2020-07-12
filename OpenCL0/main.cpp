#include <stdio.h>
#include <stdlib.h>

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#define MAX_SOURCE_SIZE (0x100000)
#define TILE_W 1024

float *a, *b, *c;
size_t* l;

int main(void) {
    printf("started running\n");

    // Create the two input vectors
    const unsigned int N = 1048576;
    a = (float*)malloc(sizeof(float) * N);
    b = (float*)malloc(sizeof(float) * N);
    c = (float*)malloc(sizeof(float) * N);
    float* ab = (float*)malloc(sizeof(float) * N);
    float res = 0;
    for (int i = 0; i < N; i++) {
        a[i] = float(rand() % 10);
        b[i] = 0;
        res += a[i];
        ab[i] = res;
    }
    // Load the kernel source code into the array source_str
    FILE* fp;
    char* source_str;
    size_t source_size;

    fp = fopen("kernel.cl", "r");
    if (!fp) {
        fprintf(stderr, "Failed to load kernel.\n");
        exit(1);
    }
    source_str = (char*)malloc(MAX_SOURCE_SIZE);
    source_size = fread(source_str, 1, MAX_SOURCE_SIZE, fp);
    fclose(fp);
    printf("kernel loading done\n");
    // Get platform and device information
    cl_device_id device_id = NULL;
    cl_device_id gpu_device = NULL;
    size_t gpu_device_work_group_size = 0;
    cl_bool is_another_gpu = false;
    cl_device_id cpu_device = NULL;
    size_t cpu_device_work_group_size = 0;
    cl_bool is_another_cpu = false;
    cl_uint ret_num_devices;
    cl_uint ret_num_platforms;

    cl_int ret = clGetPlatformIDs(0, NULL, &ret_num_platforms);
    cl_platform_id* platforms = NULL;
    platforms = (cl_platform_id*)malloc(ret_num_platforms * sizeof(cl_platform_id));

    ret = clGetPlatformIDs(ret_num_platforms, platforms, NULL);
    printf("ret at %d is %d\n", __LINE__, ret);

    for (size_t i = 0; i < ret_num_platforms; i++) {
        char* line = (char*)malloc(1024 * sizeof(char));

        ret = clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME, 64, line, NULL);
        printf("ret at %d is %d\n", __LINE__, ret);

        printf("Check platform: %s\n", line);

        cl_device_id* cur_device = NULL;

        ret = clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, NULL,
            NULL, &ret_num_devices);

        cur_device = (cl_device_id*)malloc(ret_num_devices * sizeof(cl_device_id));

        ret = clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, 5,
            cur_device, &ret_num_devices);
        printf("ret at %d is %d\n", __LINE__, ret);

        for (size_t j = 0; j < ret_num_devices; j++) {
            ret = clGetDeviceInfo(cur_device[j], CL_DEVICE_NAME, 1024, line, NULL);
            printf("ret at %d is %d\n", __LINE__, ret);

            printf("Check device: %s\n", line);

            cl_device_type* device_type = (cl_device_type*)malloc(sizeof(cl_device_type));

            ret = clGetDeviceInfo(cur_device[j], CL_DEVICE_TYPE, 1024, device_type, NULL);
            printf("ret at %d is %d\n", __LINE__, ret);

            size_t current_work_group_size;
            ret = clGetDeviceInfo(cur_device[j], CL_DEVICE_MAX_WORK_GROUP_SIZE, 1024, &current_work_group_size, NULL);
            printf("ret at %d is %d\n", __LINE__, ret);
            printf("Current work group size: %Iu\n", current_work_group_size);

            if (*device_type == CL_DEVICE_TYPE_GPU) {
                printf("Device type: GPU\n");
                if (gpu_device == NULL) {
                    gpu_device_work_group_size = current_work_group_size;
                    gpu_device = cur_device[j];
                }
                else {
                    cl_bool is_uni;
                    ret = clGetDeviceInfo(cur_device[j], CL_DEVICE_HOST_UNIFIED_MEMORY, 1024, &is_uni, NULL);
                    printf("ret at %d is %d\n", __LINE__, ret);
                    if (!is_another_gpu && !is_uni) {
                        gpu_device_work_group_size = current_work_group_size;
                        gpu_device = cur_device[j];
                    }
                    else if (is_another_gpu == !is_uni && current_work_group_size > gpu_device_work_group_size) {
                        gpu_device_work_group_size = current_work_group_size;
                        gpu_device = cur_device[j];
                    }
                }
            }
            else {
                printf("Device type: Not GPU\n");
                if (cpu_device == NULL) {
                    cpu_device_work_group_size = current_work_group_size;
                    cpu_device = cur_device[j];
                }
                else {
                    cl_bool is_uni;
                    ret = clGetDeviceInfo(cur_device[j], CL_DEVICE_HOST_UNIFIED_MEMORY, 1024, &is_uni, NULL);
                    printf("ret at %d is %d\n", __LINE__, ret);
                    if (!is_another_cpu && !is_uni) {
                        cpu_device_work_group_size = current_work_group_size;
                        cpu_device = cur_device[j];
                    }
                    else if (is_another_cpu == !is_uni && current_work_group_size > cpu_device_work_group_size) {
                        cpu_device_work_group_size = current_work_group_size;
                        cpu_device = cur_device[j];
                    }
                }
            }
        }
    }

    if (gpu_device == NULL) {
        device_id = cpu_device;
    }
    else {
        device_id = gpu_device;
    }
    char* line = (char*)malloc(1024 * sizeof(char));
    ret = clGetDeviceInfo(device_id, CL_DEVICE_NAME, 1024, line, NULL);
    printf("ret at %d is %d\n", __LINE__, ret);

    printf("Chosen device: %s\n", line);
    // Create an OpenCL context
    cl_context context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &ret);
    printf("ret at %d is %d\n", __LINE__, ret);

    // Create a command queue
    cl_command_queue command_queue = clCreateCommandQueue(context, device_id, CL_QUEUE_PROFILING_ENABLE, &ret);
    printf("ret at %d is %d\n", __LINE__, ret);

    // Create memory buffers on the device for each vector 
    cl_mem a_mem_obj = clCreateBuffer(context, CL_MEM_READ_ONLY,
        sizeof(float) * N, NULL, &ret);
    cl_mem b_mem_obj = clCreateBuffer(context, CL_MEM_READ_WRITE,
        sizeof(float) * N, NULL, &ret);
    cl_mem c_mem_obj = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
        sizeof(float) * N, NULL, &ret);

    // Copy the lists A and B to their respective memory buffers
    ret = clEnqueueWriteBuffer(command_queue, a_mem_obj, CL_TRUE, 0,
        sizeof(float) * N, a, 0, NULL, NULL);
    printf("ret at %d is %d\n", __LINE__, ret);

    ret = clEnqueueWriteBuffer(command_queue, b_mem_obj, CL_TRUE, 0,
        sizeof(float) * N, b, 0, NULL, NULL);
    printf("ret at %d is %d\n", __LINE__, ret);

    ret = clEnqueueWriteBuffer(command_queue, c_mem_obj, CL_TRUE, 0,
        sizeof(float) * N, c, 0, NULL, NULL);
    printf("ret at %d is %d\n", __LINE__, ret);

    printf("before building\n");
    // Create a program from the kernel source
    cl_program program = clCreateProgramWithSource(context, 1,
        (const char**)&source_str, (const size_t*)&source_size, &ret);
    printf("ret at %d is %d\n", __LINE__, ret);

    // Build the program
    ret = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);
    printf("ret at %d is %d\n", __LINE__, ret);

    if (ret == CL_BUILD_PROGRAM_FAILURE) {
        // Determine the size of the log
        size_t log_size;
        clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);

        // Allocate memory for the log
        char* log = (char*)malloc(log_size);

        // Get the log
        clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);

        // Print the log
        printf("%s\n", log);
    }
    else {
        printf("no errors\n");
    }

    printf("after building\n");
    // Create the OpenCL kernel
    cl_kernel kernel = clCreateKernel(program, "prefix_sum_local", &ret);
    printf("ret at %d is %d\n", __LINE__, ret);
    // Count group sizes
    size_t global_item_size = N;
    size_t dimensions;
    ret = clGetDeviceInfo(device_id, CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, sizeof(size_t), &dimensions, NULL);
    size_t size = sizeof(size_t) * dimensions;
    l = (size_t*)malloc(size);
    ret = clGetDeviceInfo(device_id, CL_DEVICE_MAX_WORK_ITEM_SIZES, size, l, NULL);
    printf("Device max dimension: %Iu\n", l[0]);
    size_t local_item_size = l[0];
    if (TILE_W < local_item_size) {
        local_item_size = TILE_W;
    }
    printf("Chosen local group size: %Iu\n", local_item_size);

    // Set the arguments of the kernel
    ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&a_mem_obj);
    printf("ret at %d is %d\n", __LINE__, ret);

    ret = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&b_mem_obj);
    printf("ret at %d is %d\n", __LINE__, ret);

    ret = clSetKernelArg(kernel, 2, sizeof(unsigned int), (void*)&N);
    printf("ret at %d is %d\n", __LINE__, ret);

    unsigned int n = TILE_W;
    ret = clSetKernelArg(kernel, 3, sizeof(unsigned int), (void*)&n);
    printf("ret at %d is %d\n", __LINE__, ret);

    ret = clSetKernelArg(kernel, 4, sizeof(float) * local_item_size, NULL);
    printf("ret at %d is %d\n", __LINE__, ret);

    printf("before execution\n");
    // Execute the OpenCL kernel on the list
    cl_event event;
    ret = clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL,
        &global_item_size, &local_item_size, 0, NULL, &event);
    printf("ret at %d is %d\n", __LINE__, ret);
    printf("after execution\n");
    // Measure kernel time
    clWaitForEvents(1, &event);
    clFlush(command_queue);
    clFinish(command_queue);
    cl_ulong time_start;
    cl_ulong time_end;

    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL);
    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL);

    double time = double(time_end - time_start);
    printf("Kernel 1 execution time: %0.5f milliseconds \n", time / 1000000.0);

    // Create the OpenCL kernel
    cl_kernel kernel1 = clCreateKernel(program, "prefix_sum_global", &ret);
    printf("ret at %d is %d\n", __LINE__, ret);

    // Set the arguments of the kernel
    ret = clSetKernelArg(kernel1, 0, sizeof(cl_mem), (void*)&b_mem_obj);
    printf("ret at %d is %d\n", __LINE__, ret);

    ret = clSetKernelArg(kernel1, 1, sizeof(cl_mem), (void*)&c_mem_obj);
    printf("ret at %d is %d\n", __LINE__, ret);

    ret = clSetKernelArg(kernel1, 2, sizeof(unsigned int), (void*)&N);
    printf("ret at %d is %d\n", __LINE__, ret);

    ret = clSetKernelArg(kernel1, 3, sizeof(unsigned int), (void*)&n);
    printf("ret at %d is %d\n", __LINE__, ret);

    printf("before execution\n");
    // Execute the OpenCL kernel on the list
    ret = clEnqueueNDRangeKernel(command_queue, kernel1, 1, NULL,
        &global_item_size, &local_item_size, 0, NULL, &event);
    printf("after execution\n");
    // Measure kernel time
    clWaitForEvents(1, &event);
    clFinish(command_queue);

    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL);
    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL);

    double time2 = double(time_end - time_start);
    printf("Kernel 2 execution time: %0.5f milliseconds \n", time2 / 1000000.0);
    // Read the memory buffer c on the device to the local variable c
    ret = clEnqueueReadBuffer(command_queue, c_mem_obj, CL_TRUE, 0,
        sizeof(float) * N, c, 0, NULL, NULL);
    printf("after copying\n");
    // Display the result to the screen
    bool is_successful = true;
    for (int i = 0; i < N; i++) {
        //printf("c[%d] = %f || %f\n", i, c[i], ab[i]);
        if (c[i] != ab[i]) {
            is_successful = false;
        }
    }
    if (is_successful) {
        printf("DONE CORRECTLY\n");
    }
    else {
        printf("FAILED\n");
    }
    // Print total time
    printf("Total kernel execution time: %0.5fms", (time + time2) / 1000000);

    // Clean up
    ret = clFlush(command_queue);
    ret = clFinish(command_queue);
    ret = clReleaseKernel(kernel);
    ret = clReleaseProgram(program);
    ret = clReleaseMemObject(a_mem_obj);
    ret = clReleaseMemObject(b_mem_obj);
    ret = clReleaseMemObject(c_mem_obj);
    ret = clReleaseCommandQueue(command_queue);
    ret = clReleaseContext(context);
    free(a);
    free(b);
    free(c);
    return 0;
}