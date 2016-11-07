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
	kernel void _muls(global const float *in_vector,
		global float *out_vector,
		float scalar) {
	const int pos = get_global_id(0) & (~(WG_SIZE - 1));
	const int local_id = get_local_id(0);
	int i;

	for (i = 0; i < BLOCK_SIZE; i++) {
		out_vector[pos * BLOCK_SIZE + local_id + i * WG_SIZE] =
			in_vector[pos * BLOCK_SIZE + local_id + i * WG_SIZE] * scalar;
	}
});

// Wrapper
static bool _muls_init = false;
static cl_program _muls_program;
void muls_cleanup() {
	if (_muls_init)
		clReleaseProgram(_muls_program);
}
void muls(const float *in_vector,
	float *out_vector,
	float scalar,
	int len) {
	cl_kernel _muls_kernel;

	if (!_muls_init) {
		_muls_program = opencl_compile_program(_muls);
		_muls_init = true;
	}
	cl_int cl_len = len;
	cl_mem cl_in_vector, cl_out_vector;
	cl_int err;
	cl_kernel kernel;
	cl_float cl_scalar = scalar;

	cl_in_vector = clCreateBuffer(opencl_get_context(),
		CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		sizeof(float) * len, (void *)in_vector, &err);
	clCheckErr(err, "Failed to create device buffer");

	cl_out_vector = clCreateBuffer(opencl_get_context(), CL_MEM_WRITE_ONLY,
		sizeof(float) * len, NULL, &err);
	clCheckErr(err, "Failed to create device buffer");

	kernel = clCreateKernel(_muls_program, "_muls", &err);
	clCheckErr(err, "Failed to create kernel");

	clCheck(clSetKernelArg(kernel, 0, sizeof(cl_in_vector), &cl_in_vector));
	clCheck(clSetKernelArg(kernel, 1, sizeof(cl_out_vector), &cl_out_vector));
	clCheck(clSetKernelArg(kernel, 2, sizeof(cl_float), &cl_scalar));

	cl_event kernel_completion;
	size_t global_work_size[1] = { len / BLOCK_SIZE };
	size_t local_work_size[1] = { WG_SIZE };

	clCheck(clEnqueueNDRangeKernel(opencl_get_queue(), kernel,
		1, NULL, global_work_size, local_work_size, 0, NULL, &kernel_completion));

	clCheck(clWaitForEvents(1, &kernel_completion));
	clCheck(clReleaseEvent(kernel_completion));

	clCheck(clEnqueueReadBuffer(opencl_get_queue(), cl_out_vector, CL_TRUE,
		0, len * sizeof(float), out_vector, 0, NULL, NULL));

	clCheck(clReleaseMemObject(cl_in_vector));
	clCheck(clReleaseMemObject(cl_out_vector));
	clCheck(clReleaseKernel(kernel));
}

void host_muls(const float *in_vector,
	float *out_vector,
	float scalar,
	int len) {
	int i;

	for (i = 0; i < len; i++)
		out_vector[i] = in_vector[i] * scalar;
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

	float   *vector, *result, temp;
	int     i;
	int     dim = atoi(argv[1]);
	//u_int64_t start, end;

	if (dim & (BLOCK_SIZE * WG_SIZE - 1)) {
		dim = (dim & (~(BLOCK_SIZE * WG_SIZE - 1))) + BLOCK_SIZE * WG_SIZE;
	}
	vector = (float*)malloc(sizeof(*vector) * dim);
	result = (float*)malloc(sizeof(*result) * dim);

	for (i = 0; i < dim; i++) {
		vector[i] = i;
	}

	printf("Input:\n");
	printVector(vector, dim);

	//start = gettimeofday_usec();
	host_muls(vector, result, 5.0, dim);
	printf("Times 5:\n");
	printVector(result, dim);
	//end = gettimeofday_usec();
	//printf("Host V*S: %.2f seconds\n", ((double)((end - start))) / ONE_MILLION);

	muls(vector, result, 6.0, dim);
	printf("Times 6:\n");
	printVector(result, dim);
	//start = gettimeofday_usec();
	muls(vector, result, 7.0, dim);
	printf("Times 7:\n");
	printVector(result, dim);
	//end = gettimeofday_usec();
	//printf("GPU M*S: %.2f seconds\n", ((double)((end - start))) / ONE_MILLION);

	// Check the results
	for (i = 0; i < dim; i++) {
		temp = vector[i] * 7.0;
		if ((temp != 0.0 || vector[i] != 0.0) && fabs(temp - result[i]) > 0.1 * temp) {
			printf("Result index: %d seems wrong: %f != %f\n",
				i, result[i], temp);
			exit(-__LINE__);
		}
	}

	muls_cleanup();
	opencl_end();
}

