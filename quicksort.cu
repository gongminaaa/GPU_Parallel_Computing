#include <stdlib.h>
#include <stdio.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include "quicksort_kernel.cu"

#define CUDA_SAFE_CALL_NO_SYNC( call) do {                              \
  cudaError err = call;                                                 \
  if( cudaSuccess != err) {                                             \
    fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n",       \
                __FILE__, __LINE__, cudaGetErrorString( err) );         \
    exit(EXIT_FAILURE);                                                 \
    } } while (0)

#define CUDA_SAFE_CALL( call) do {                                      \
  CUDA_SAFE_CALL_NO_SYNC(call);                                         \
  cudaError err = cudaDeviceSynchronize();                              \
  if( cudaSuccess != err) {                                             \
     fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n",      \
                 __FILE__, __LINE__, cudaGetErrorString( err) );        \
     exit(EXIT_FAILURE);                                                \
     } } while (0)


void checkAnswer(int* answer)
{
	bool flag = true;
	int size = sizeof(answer)/sizeof(int);
	for(int i=0; i<size-1; i++)
	{
		if(answer[i] <= answer[i+1])
		{
		}
		else
		{
			flag = false;
			break;
		}
	}

        if(flag == true)
                printf("### ANSWER CORRECT ###\n");
        else
                printf("### ANSWER WRONG ###\n");
}

int main() {

    /******************************************
    *	quicksort_kernel(list, size)
    *   To test, change these two parameters
    *******************************************/
    int size = 1000000;
    int* list;
    int* answer = (int*)malloc(size*sizeof(int));

    cudaMallocManaged(&list, size*sizeof(int));

    for(int i=0; i<size; i++)
    {   
        list[i] = (int)rand();
    }

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    float elapsedTime;
    cudaDeviceSynchronize();
    cudaEventRecord(start, 0);

    answer = quicksort_kernel(list, size);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);

    // In the writeup, we excludes CPU prepartion time in quicksort_kernel
    // Also, applied same condition to the comparisons
    printf("Total time(Including CPU Prep Time): %f (ms)\n", elapsedTime);

    checkAnswer(answer);

    cudaFree(list);
}

