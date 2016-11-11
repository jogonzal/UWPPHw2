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
char _bfs[] =
"#define WG_SIZE    256\n"
"#define BLOCK_SIZE 2\n"
OPENCL_CODE(
	kernel void _bfs(
		global int *nodeList,
		global int *edgeList,
		global int *result,
		int dim,
		global int *foundNewLevel) {

		const int globalId = get_global_id(0);
		int currentLevel = 0;
		int totalEdges = 0;
		int totalVertex = 0;

		do {
			if (globalId == 0) {
				foundNewLevel[0] = 0;
			}
			printf("%d: Starting loop!", globalId);
			barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
			for (int i = 0; i < BLOCK_SIZE; i++) {
				int edgeListOffset = globalId * 2 + i;
				int origin = edgeList[edgeListOffset];
				int destination = edgeList[edgeListOffset + 1];

				// If the node is the current level, activate it's destination
				int currentLevelOrigin = nodeList[origin];
				int currentLevelDestination = nodeList[destination];

				if (currentLevelOrigin == currentLevel) {
					totalEdges++;
				}
				if (currentLevelDestination == currentLevel) {
					totalEdges++;
				}

				if (currentLevelOrigin == currentLevel && currentLevelDestination == -1) {
					nodeList[destination] = currentLevel + 1;
					foundNewLevel[0] = 1;
					totalVertex++;
				}
				else if (currentLevelDestination == currentLevel && currentLevelOrigin == -1) {
					nodeList[origin] = currentLevel + 1;
					foundNewLevel[0] = 1;
					totalVertex++;
				}
			}
			currentLevel++;
			barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
		} while (foundNewLevel[0] == 1);
		
		printf("%d: Done with loop!", globalId);

		int maxLevel = currentLevel--;

		result[globalId * 3] = maxLevel;
		result[globalId * 3 + 1] = totalVertex;
		result[globalId * 3 + 2] = totalEdges;
});

// Wrapper
static bool _bfs_init = false;
static cl_program _bfs_program;
void bfs_cleanup() {
	if (_bfs_init)
		clReleaseProgram(_bfs_program);
}
void gpuSinglealgorithm(int *maxLevelGpu,
	int *vertexCountGpu,
	int *edgeCountGpu,
	int dim,
	int nodeListSize,
	int *nodeList,
	int *edgeList) {

	// For every edge...
	int currentLevel = 0;
	int totalEdges = 0;
	int totalVertex = 0;
	bool foundNewLevel;
	do {
		foundNewLevel = false;
		for (int i = 0; i < dim; i++) {
			int origin = edgeList[i*2];
			int destination = edgeList[i*2 + 1];

			// If the node is the current level, activate it's destination
			int currentLevelOrigin = nodeList[origin];
			int currentLevelDestination = nodeList[destination];

			if (currentLevelOrigin == currentLevel) {
				totalEdges++;
			}
			if (currentLevelDestination == currentLevel) {
				totalEdges++;
			}

			if (currentLevelOrigin == currentLevel && currentLevelDestination == -1) {
				nodeList[destination] = currentLevel + 1;
				foundNewLevel = true;
				totalVertex++;
			}
			else if (currentLevelDestination == currentLevel && currentLevelOrigin == -1) {
				nodeList[origin] = currentLevel + 1;
				foundNewLevel = true;
				totalVertex++;
			}
		}
		currentLevel++;
	} while (foundNewLevel);

	*edgeCountGpu = totalEdges + 1;
	*vertexCountGpu = totalVertex + 1;
	*maxLevelGpu = currentLevel-1;
}

void bfs(int *maxLevelGpu,
	int *vertexCountGpu,
	int *edgeCountGpu,
	int dim,
	int nodeListSize,
	int *nodeList,
	int *edgeList) {
	cl_kernel _bfs_kernel;

	if (!_bfs_init) {
		_bfs_program = opencl_compile_program(_bfs);
		_bfs_init = true;
	}
	cl_int cl_dim = dim;
	cl_int cl_found_new = 0;
	cl_mem cl_in_node_list, cl_in_edge_list;
	cl_int err;
	cl_mem cl_out_result;
	cl_mem cl_currentstate;
	cl_kernel kernel;
	int resultSize = dim * 3;
	int *result = new int[resultSize];

	int foundNew = 0;

	cl_in_node_list = clCreateBuffer(opencl_get_context(),
		CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		sizeof(int) * nodeListSize, (void *)nodeList, &err);
	clCheckErr(err, "Failed to create device buffer");

	cl_in_edge_list = clCreateBuffer(opencl_get_context(),
		CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		sizeof(int) * dim * 2, (void *)edgeList, &err);
	clCheckErr(err, "Failed to create device buffer");

	cl_out_result = clCreateBuffer(opencl_get_context(), CL_MEM_WRITE_ONLY,
		sizeof(int) * resultSize, NULL, &err);
	clCheckErr(err, "Failed to create device buffer");

	cl_currentstate = clCreateBuffer(opencl_get_context(),
		CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
		sizeof(int) * 1, (void *)&foundNew, &err);

	kernel = clCreateKernel(_bfs_program, "_bfs", &err);
	clCheckErr(err, "Failed to create kernel");

	clCheck(clSetKernelArg(kernel, 0, sizeof(cl_in_node_list), &cl_in_node_list));
	clCheck(clSetKernelArg(kernel, 1, sizeof(cl_in_edge_list), &cl_in_edge_list));
	clCheck(clSetKernelArg(kernel, 2, sizeof(cl_out_result), &cl_out_result));
	clCheck(clSetKernelArg(kernel, 3, sizeof(cl_int), &cl_dim));
	clCheck(clSetKernelArg(kernel, 4, sizeof(cl_currentstate), &cl_currentstate));

	cl_event kernel_completion;
	size_t global_work_size[1] = { dim / BLOCK_SIZE };
	size_t local_work_size[1] = { WG_SIZE };

	clCheck(clEnqueueNDRangeKernel(opencl_get_queue(), kernel,
		1, NULL, global_work_size, local_work_size, 0, NULL, &kernel_completion));

	clCheck(clWaitForEvents(1, &kernel_completion));
	clCheck(clReleaseEvent(kernel_completion));

	clCheck(clEnqueueReadBuffer(opencl_get_queue(), cl_out_result, CL_TRUE,
		0, sizeof(int) * resultSize , result, 0, NULL, NULL));

	*vertexCountGpu = 1;
	*edgeCountGpu = 1;
	for (int i = 0; i < dim; i++) {
		int maxLevelLocal = result[i*3 + 0];
		int vertexCountLocal = result[i*3 + 1];
		int edgeCountLocal = result[i*3 + 2];

		*maxLevelGpu = maxLevelLocal;
		(*vertexCountGpu) += vertexCountLocal;
		(*edgeCountGpu) += edgeCountLocal;
	}

	clCheck(clReleaseMemObject(cl_in_node_list));
	clCheck(clReleaseMemObject(cl_in_edge_list));
	clCheck(clReleaseMemObject(cl_out_result));
	clCheck(clReleaseKernel(kernel));

	delete[] result;
}

void host_bfs(float *matrix,
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

	int index;

	Node(int indexParam) {
		visited = false;
		previous = NULL;
		children = new vector<Node*>();
		index = indexParam;
	}
};

struct edge {
	unsigned long long origin;
	unsigned long long destination;
};

Node* GetOrCreateNode(map<unsigned long long, Node*> *graphInputMap, unsigned long long id, int *nodeIndexOffset) {
	Node *node;
	std::map<unsigned long long, Node*>::iterator it;
	it = graphInputMap->find(id);
	if (it == graphInputMap->end()) {
		//printf("Does not contain\n");
		node = new Node(*nodeIndexOffset);
		(*nodeIndexOffset)++;
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

	unsigned long long edgeMaxCount = 3000000 * 2;
	unsigned long long nodeMaxCount = 64000 * 1;

	map<unsigned long long, Node*> *graphInputMap = new map<unsigned long long, Node*>();

	int     dim = edgeMaxCount / 2;
	if (dim & (BLOCK_SIZE * WG_SIZE - 1)) {
		dim = (dim & (~(BLOCK_SIZE * WG_SIZE - 1))) + BLOCK_SIZE * WG_SIZE;
	}

	int *edgeList = new int[dim * 2];
	for (int i = 0; i < dim; i++) {
		edgeList[i] = -1;
	}
	int *nodeList = new int[nodeMaxCount];
	for (int i = 0; i < nodeMaxCount; i++) {
		nodeList[i] = -1;
	}

	int nodeOffset = 0;
	long edgeListOffset = 0;
	for (;;) {
		size_t elementsRead = fread(buff, sizeof(struct edge), buffSize, file);
		edgesTotal += elementsRead * 2;
		for (int i = 0; i < elementsRead; i++) {
			struct edge edge = buff[i];
			//printEdge(edge);
			Node *nodeOrigin = GetOrCreateNode(graphInputMap, edge.origin, &nodeOffset);
			Node *nodeDestination = GetOrCreateNode(graphInputMap, edge.destination, &nodeOffset);
			nodeOrigin->children->push_back(nodeDestination);
			nodeDestination->children->push_back(nodeOrigin);
			edgeList[edgeListOffset] = nodeOrigin->index;
			edgeList[edgeListOffset + 1] = nodeDestination->index;
			edgeListOffset+=2;
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

	nodeList[root->index] = 0;

	int verticesTotal = graphInputMap->size();

	printf("I read %d edges and have found %d vertices (in total). Now doing breadth first search to calculate the number of edges and vertices from root 0x%llx \n",
		edgesTotal, verticesTotal, rootValue);

	free(graphInputMap);
	free(buff);

	// Now that we've read and loaded all of the elements, we can do breadth first search

	Node *deepestNode = NULL;

	printf("Running BFS...\n\n");

	int maxLevelSingleThreaded = 0; // Assumming at least root exists, level is 1
	int vertexCountSingleThreaded = 0;
	int edgeCountSingleThreaded = 0;
	SingleThreadedBfs(&maxLevelSingleThreaded, &vertexCountSingleThreaded, &edgeCountSingleThreaded, root, &deepestNode);

	int maxLevelGpu = 0; // Assumming at least root exists, level is 1
	int vertexCountGpu = 0;
	int edgeCountGpu = 0;
	bfs(&maxLevelGpu, &vertexCountGpu, &edgeCountGpu, dim, nodeMaxCount, nodeList, edgeList);
	//gpuSinglealgorithm(&maxLevelGpu, &vertexCountGpu, &edgeCountGpu, dim, nodeMaxCount, nodeList, edgeList);

	printf("Done!===============\n");
	printf("\n\n");

	PrintToFileAndConsole(verticesTotal, edgesTotal, rootValue, vertexCountSingleThreaded, maxLevelSingleThreaded);
	PrintToFileAndConsole(verticesTotal, edgesTotal, rootValue, vertexCountGpu, maxLevelGpu);

	bfs_cleanup();
	opencl_end();

	return 0;
}

