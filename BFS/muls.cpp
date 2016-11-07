#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include "opencl_interface.hpp"
#include <stdio.h>
#include <string.h>
#include <stdbool.h>
#include <sys/types.h>
#include <time.h>
#include <map>
#include <vector>
#include <queue>
#include <stack>
#include <stdbool.h>
#include <time.h>

using namespace std;

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
			// TODO!
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
	// TODO
}

class Node {
public:
	unsigned long long id;
	bool visited;
	Node *previous;
	vector<Node*> *children;

	Node() {
		visited = false;
		previous = NULL;
		children = new vector<Node*>();
	}
};

struct edge {
	unsigned long long origin;
	unsigned long long destination;
};

Node* GetOrCreateNode(map<unsigned long long, Node*> *graphInputMap, unsigned long long id) {
	Node *node;
	std::map<unsigned long long, Node*>::iterator it;
	it = graphInputMap->find(id);
	if (it == graphInputMap->end()) {
		//printf("Does not contain\n");
		node = new Node();
		node->id = id;
		(*graphInputMap)[id] = node;
		return node;
	}
	else {
		//printf("Contains\n");
		// This would mean the dictionary already contained the node
		// simply return it
		return it->second;
	}
}

void PrintToFileAndConsole(int verticesTotal, int edgesTotal, unsigned long long rootValue, int vertexCount, int maxLevel) {
	printf("\n\nGraph vertices: %d with total edges %d. Reached vertices from %lld is %d and max level is %d.\n\n",
		verticesTotal, edgesTotal, rootValue, vertexCount, maxLevel);

	FILE *f = fopen("output.txt", "w+");
	if (f == NULL)
	{
		printf("Error opening file!\n");
		exit(1);
	}

	fprintf(f, "Graph vertices: %d with total edges %d. Reached vertices from %lld is %d and max level is %d.\n\n",
		verticesTotal, edgesTotal, rootValue, vertexCount, maxLevel);

	printf("I wrote this to output.txt\n");

	fclose(f);
}

void SingleThreadedBfs(int *maxLevel, int *vertexCount, int *edgeCount, Node *root, Node **deepestNode) {
	queue<Node*> *bfsQueue = new queue<Node*>();
	bfsQueue->push(root);
	root->visited = true;
	(*edgeCount)++;
	(*maxLevel) = -1; // Will be incremented to at least 0
	while (bfsQueue->size() > 0) {
		// printf("Level %d has %lu elements to explore. Exploring...\n", *maxLevel, bfsQueue->size());

		queue<Node*> *newQueue = new queue<Node*>();

		// Ennumerate through children of elements in the queue
		while (bfsQueue->size() > 0) {
			Node *nodeToExplore = bfsQueue->front();
			bfsQueue->pop();

			(*vertexCount)++;

			// Keep track of deepest node
			*deepestNode = nodeToExplore;

			// Push all children that are not visited
			for (vector<Node*>::iterator it = nodeToExplore->children->begin(); it != nodeToExplore->children->end(); ++it) {
				(*edgeCount)++; // Edges have to be counted, even if we already visited that node

				Node *child = *it;
				// Skip if already visited
				if (!child->visited) {
					child->previous = nodeToExplore; // For shortest path
					newQueue->push(child);
					// Mark as visited
					child->visited = true;
				}
			}
		}

		free(bfsQueue);
		bfsQueue = newQueue;
		(*maxLevel)++;
	}

	free(bfsQueue);
}

void printEdge(unsigned long long origin, unsigned long long destination) {
	printf("%llx -> %llx\n", origin, destination);
}

void printEdge(struct edge edge) {
	printf("%llx -> %llx\n", edge.origin, edge.destination);
}

#include <fstream>

int fileSize(const char *add) {
	ifstream mySource;
	mySource.open(add, ios_base::binary);
	mySource.seekg(0, ios_base::end);
	int size = mySource.tellg();
	mySource.close();
	return size;
}

int main(int argc, char *argv[]) {
	opencl_start();

	if (argc != 3) {
		printf("usage: %s root inputfile\n", argv[0]);
		printf("Sample root: 0");
		return -1;
	}

	char* inputFile = argv[1];
	printf("Will do BFS with GPU and will parse %s for input.\n", inputFile);

	printf("Doing BFS in host...\n");

	printf("Multiplying matrix and vector in GPU...\n");

	unsigned long long rootValue = (unsigned long long)atoi(argv[1]);
	char* fileName = argv[2];

	printf("Starting. root %llx, inputfile %s\n", rootValue, fileName);
	FILE* file = fopen(fileName, "rb");

	int buffSize = 1000;

	struct edge *buff = (struct edge *) malloc(sizeof(struct edge) * buffSize);

	int edgesTotal = 0;

	int size = fileSize(fileName);
	printf("The size is %d\n", size);

	map<unsigned long long, Node*> *graphInputMap = new map<unsigned long long, Node*>();
	//Node *nodeArray = new Node[64000];
	int nodeArrayOffset = 0;
	for (;;) {
		size_t elementsRead = fread(buff, sizeof(struct edge), buffSize, file);
		edgesTotal += elementsRead * 2;
		for (int i = 0; i < elementsRead; i++) {
			struct edge edge = buff[i];
			//printEdge(edge);
			Node *nodeOrigin = GetOrCreateNode(graphInputMap, edge.origin);
			Node *nodeDestination = GetOrCreateNode(graphInputMap, edge.destination);
			nodeOrigin->children->push_back(nodeDestination);
			nodeDestination->children->push_back(nodeOrigin);
		}
		if (elementsRead < buffSize) { break; }
	}

	fclose(file);

	// Get the pointer to the root
	Node *root = NULL;
	std::map<unsigned long long, Node*>::iterator it;
	it = graphInputMap->find(rootValue);
	if (it != graphInputMap->end()) {
		printf("The root you specified was in the input file. Now calculating stats...\n");
		root = it->second;
	}
	else {
		printf("Where did you get this value from? The root you specified wasn't in the input file\n");
		exit(1);
	}

	int verticesTotal = graphInputMap->size();

	printf("I read %d edges and have found %d vertices (in total). Now doing breadth first search to calculate the number of edges and vertices from root 0x%llx \n",
		edgesTotal, verticesTotal, rootValue);

	free(graphInputMap);
	free(buff);

	// Now that we've read and loaded all of the elements, we can do breadth first search

	int maxLevel = 0; // Assumming at least root exists, level is 1
	int vertexCount = 0;
	int edgeCount = 0;

	Node *deepestNode = NULL;

	printf("Running BFS...\n\n");

	SingleThreadedBfs(&maxLevel, &vertexCount, &edgeCount, root, &deepestNode);
	
	int     dim = 0;
	if (dim & (BLOCK_SIZE * WG_SIZE - 1)) {
		dim = (dim & (~(BLOCK_SIZE * WG_SIZE - 1))) + BLOCK_SIZE * WG_SIZE;
	}
	//GpuBfs(&maxLevel, &vertexCount, &edgeCount, root, &deepestNode, threadCount);

	printf("Done!===============\n");
	printf("\n\n");

	printf("Here is proof that the max level is %d.\n", maxLevel);
	Node *currentNode = deepestNode;
	stack<Node*> *s = new stack<Node*>();
	while (currentNode != NULL) {
		s->push(currentNode);
		currentNode = currentNode->previous;
	}
	printf("A sample path that has this depth is \n");
	printf("ROOT ->");
	while (s->size() > 0) {
		Node *next = s->top();
		s->pop();
		printf(" 0x%llx ->", next->id);
	}
	free(s);
	printf(" done!\n");

	PrintToFileAndConsole(verticesTotal, edgesTotal, rootValue, vertexCount, maxLevel);

	muls_cleanup();
	opencl_end();

	return 0;
}

