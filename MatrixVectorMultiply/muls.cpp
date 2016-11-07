#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include "opencl_interface.hpp"

#define WG_SIZE     256
#define BLOCK_SIZE  2
char _muls[] =
"#define WG_SIZE    256\n"
"#define BLOCK_SIZE 2\n"
OPENCL_CODE(
	kernel void _muls(
		global float *matrix,
		global float *vector,
		global float *result,
		int dim) {
		const int globalId = get_global_id(0);

		for (int i = 0; i < BLOCK_SIZE; i++) {
			int position = globalId + i * WG_SIZE;
			int row = position;
			int acc = 0;
			for (int c = 0; c < dim; c++) {
				acc += matrix[row * dim + c] * vector[c];
			}
			result[position] = acc;
		}
});

// Wrapper
static bool _muls_init = false;
static cl_program _muls_program;
void muls_cleanup() {
	if (_muls_init)
		clReleaseProgram(_muls_program);
}
void muls(float *matrix,
	float *vector,
	float *result,
	int dim) {
	cl_kernel _muls_kernel;

	if (!_muls_init) {
		_muls_program = opencl_compile_program(_muls);
		_muls_init = true;
	}
	cl_int cl_dim = dim;
	cl_mem cl_in_vector, cl_in_matrix, cl_out_result;
	cl_int err;
	cl_kernel kernel;

	cl_in_matrix = clCreateBuffer(opencl_get_context(),
		CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		sizeof(float) * dim * dim, (void *)matrix, &err);
	clCheckErr(err, "Failed to create device buffer");

	cl_in_vector = clCreateBuffer(opencl_get_context(),
		CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		sizeof(float) * dim, (void *)vector, &err);
	clCheckErr(err, "Failed to create device buffer");

	cl_out_result = clCreateBuffer(opencl_get_context(), CL_MEM_WRITE_ONLY,
		sizeof(float) * dim, NULL, &err);
	clCheckErr(err, "Failed to create device buffer");

	kernel = clCreateKernel(_muls_program, "_muls", &err);
	clCheckErr(err, "Failed to create kernel");

	clCheck(clSetKernelArg(kernel, 0, sizeof(cl_in_matrix), &cl_in_matrix));
	clCheck(clSetKernelArg(kernel, 1, sizeof(cl_in_vector), &cl_in_vector));
	clCheck(clSetKernelArg(kernel, 2, sizeof(cl_out_result), &cl_out_result));
	clCheck(clSetKernelArg(kernel, 3, sizeof(cl_float), &cl_dim));

	cl_event kernel_completion;
	size_t global_work_size[1] = { dim / BLOCK_SIZE };
	size_t local_work_size[1] = { WG_SIZE };

	clCheck(clEnqueueNDRangeKernel(opencl_get_queue(), kernel,
		1, NULL, global_work_size, local_work_size, 0, NULL, &kernel_completion));

	clCheck(clWaitForEvents(1, &kernel_completion));
	clCheck(clReleaseEvent(kernel_completion));

	clCheck(clEnqueueReadBuffer(opencl_get_queue(), cl_out_result, CL_TRUE,
		0, dim * sizeof(float), result, 0, NULL, NULL));

	clCheck(clReleaseMemObject(cl_in_matrix));
	clCheck(clReleaseMemObject(cl_in_vector));
	clCheck(clReleaseMemObject(cl_out_result));
	clCheck(clReleaseKernel(kernel));
}

void host_muls(float *matrix,
	float *vector,
	float *result,
	int dim) {
	for (int r = 0; r < dim; r++) {
		int acc = 0;
		for (int c = 0; c < dim; c++) {
			acc += matrix[r * dim + c] * vector[c];
		}
		result[r] = acc;
	}
}

void printVector(float *vector, int dim) {
	printf("[");
	for (int i = 0; i < dim; i++) {
		printf("%f\t", vector[i]);
	}
	printf("]\n");
}

int main(int argc, char *argv[]) {
	opencl_start();

	float   *matrix, *vector, *resultHost, *resultGpu;
	int     i;
	int     dim = atoi(argv[1]);
	//u_int64_t start, end;

	if (dim & (BLOCK_SIZE * WG_SIZE - 1)) {
		dim = (dim & (~(BLOCK_SIZE * WG_SIZE - 1))) + BLOCK_SIZE * WG_SIZE;
	}

	printf("Dimensions are %d\n", dim);

	matrix = (float*)malloc(sizeof(*matrix) * dim * dim);
	vector = (float*)malloc(sizeof(*vector) * dim);
	resultHost = (float*)malloc(sizeof(*resultHost) * dim);
	resultGpu = (float*)malloc(sizeof(*resultGpu) * dim);

	for (i = 0; i < dim * dim; i++) {
		matrix[i] = 1;
	}

	for (i = 0; i < dim; i++) {
		vector[i] = 2;
	}

	printf("Multiplying matrix and vector in host...\n");

	//start = gettimeofday_usec();
	host_muls(matrix, vector, resultHost, dim);
	//end = gettimeofday_usec();
	//printf("Host V*S: %.2f seconds\n", ((double)((end - start))) / ONE_MILLION);

	printf("Multiplying matrix and vector in GPU...\n");
	muls(matrix, vector, resultGpu, dim);
	//start = gettimeofday_usec();
	//end = gettimeofday_usec();
	//printf("GPU M*S: %.2f seconds\n", ((double)((end - start))) / ONE_MILLION);

	// Check the results
	for (i = 0; i < dim; i++) {
		float resultHostIndividual = resultHost[i];
		float resultGpuIndividual = resultGpu[i];
		if (fabs(resultHostIndividual - resultGpuIndividual) > fabs(0.1 * resultHostIndividual)) {
			printf("Result index: %d seems wrong: %f != %f\n",
				i, resultHostIndividual, resultGpuIndividual);
			exit(-__LINE__);
		}
	}

	free(vector);
	free(matrix);
	free(resultHost);
	free(resultGpu);

	muls_cleanup();
	opencl_end();
}

