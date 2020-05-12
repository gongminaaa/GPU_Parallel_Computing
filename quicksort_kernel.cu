#include <stdio.h>
#include <math.h>
#include <time.h>
#include "cuda_runtime.h"

#define PICK_PIVOT 1

#define PIVOT_CANDIDATE 3

#define checkCudaError(e) { checkCudaErrorImpl(e, __FILE__, __LINE__); }

inline void checkCudaErrorImpl(cudaError_t e, const char* file, int line, bool abort = true) {
    if (e != cudaSuccess) {
        fprintf(stderr, "[CUDA Error] %s - %s:%d\n", cudaGetErrorString(e), file, line);
        if (abort) exit(e);
    }
}


/******************************************************************************
* __device__ bool IsExceptionError(int q_tid, int pid, int q_val, int n)
* 
* Description: 
* 	First three parameters are indexes used for the working queue.
* 	Prevent them from accessing to illegal memory.
*******************************************************************************/
__device__
bool IsExceptionError(int q_tid, int pid, int q_val, int n)
{
    int max = 2 * n;
    if(q_tid >= max  || q_tid < 0)
        {
            printf("q_tid: %d ILLEGAL MEMORY ACCESS\n", q_tid);
            return false;
        }
        
        if(q_val >= n || q_val < 0)
        {
            printf("q[q_tid]: %d ILLEGAL MEMORY ACCESS\n", q_val);
            return false;
        }

        if(pid >= max || pid < 0)
        {
            printf("pid: %d ILLEGAL MEMORY ACCESS\n", pid);
            return false;
        }

        return true;
}


/**************************************************
* __device__ bool IsOutOfBoundary(int val, int n)
* 
* Description: 
*	Parameter val gets head and tail index,
*	which should smaller than n(=MAXSIZE).
*
* 	Prevent them from accessing to illegal memory.
***************************************************/
__device__
bool IsOutOfBoundary(int val, int n)
{
    if(val >= n || val < 0)
    {
        return false;
    }
    return true;
}



/**********************************************************************************************************************************************
* __global__ void quickSort(int n, int *before_sort_array, int *after_sort_array, int *pivot_queue, int *working_queue, int *pivot_arr)
* 
* Description: 
*	This part is where each thread grabs a work and compares its value with the pivot.
*	The head and tail pointer's indexes are in the working queue.
*	The threads have different head and tail pointer depending on from where a thread grabbed a work.
*	If the value is bigger or smaller than the pivot,
*	use atomicAdd or atomicSub to move head and tail pointer.
*
*	If a thread id is responsible for pivot,
*	save the pivot in a pivot_arr.
*	It will be handled in sencond kernel.
*
**********************************************************************************************************************************************/
__global__
void quickSort(int n, int *before_sort_array, int *after_sort_array, int *pivot_queue, int *working_queue, int *pivot_arr)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    
    // 1. Compare own value with the pivot
    if(tid < n)
    {
        int q_tid = 2*tid;
        int pid = 2*working_queue[q_tid];
		int pivot = before_sort_array[working_queue[q_tid]];
        int value = before_sort_array[tid];
        int head = 0;
        int tail = 0;
        
        // Prevent Illegal Memory Access
        if(!IsExceptionError(q_tid, pid, pivot_queue[q_tid], n))
        {
            printf("q_tid: %d, pid: %d, pivot_queue[q_tid]: %d \n", q_tid, pid, pivot_queue[q_tid]);
        }

	if(pivot < value)
        {   
            tail = atomicSub(&pivot_queue[pid+1], 1);

			// Prevent Illegal Memory Access
            if(!IsOutOfBoundary(tail, n))
            {
                printf("ILLEGAL MEMORY ACCESS tail: %d, q_tid: %d, pivot_queue[q_tid]: %d, tid: %d, pivot_queue[q_tid]*2: %d, pid: %d\n", tail,q_tid, pivot_queue[q_tid],tid,pivot_queue[q_tid]*2, pid);
	    }
			
	    after_sort_array[tail] = value;
        }
	else if(pivot >= value)
        {
            // if condition for handling the same value as the pivot
            if(pid != q_tid)
            {
				head = atomicAdd(&pivot_queue[pid], 1);

				// Prevent Illegal Memory Access
				if(!IsOutOfBoundary(head, n))
				{
					printf("ILLEGAL MEMORY ACCESS tail: %d, q_tid: %d, pivot_queue[q_tid]: %d, tid: %d, pivot_queue[q_tid]*2: %d, pid: %d\n", head,q_tid, pivot_queue[q_tid],tid,pivot_queue[q_tid]*2, pid);
				}       

				after_sort_array[head] = value;
            }
            
        }
        
        // 2. Save the pivot into seperate pivot array -> Second kernel will locate the pivots into determined place
        if(pivot == value)
        {
            // if condition for handling the same value as the pivot(Only a thread that is responsible for the pivot can pass through)
            if(pid == q_tid)
            {
				// Block the pivot that has been saved into pivot array(reduce the redundant work)
                if(pivot_arr[tid] != pivot)
                {
                    pivot_arr[tid] = pivot;
                }
            }
        }
    }
}

/**************************************************
* __device__ void gpuSwap(int* a, int* b) 
* 
* Description: 
*	This function is used when we choose a better pivot.
*	Our implementation always picks first element, 
*	so put better pivot to the first location
* 	
***************************************************/
__device__
void gpuSwap(int* a, int* b) 
{ 
    int t = *a; 
    *a = *b; 
    *b = t; 
}


/****************************************************************************************************************************
* __global__ void arrangeQueueAndPivot(int *new_pivot_queue, int *old_pivot_queue, int n, int *sort_array, 
												int *pivot_arr, int *new_working_queue, int *old_working_queue, bool* done)
* 
* Description: 
*	First, pivots in the pivot_arr are stored back to the sort array where indexes in the old_pivot_queue designate.
*	Second, working queue is reinitialized with the information of pivot, head and tail in old_pivot_queue(pivot indexes) and old_working_queue(head and tail indexes).
*	Third, pick first, middle, and last elements from the sub-array and pick the median number.
*	Our implementation always sets the pivot at the first index of arrays.
*	Therefore, swap the median number with the value in the 0th index of the array.
*	Consequently, new_working_queue, new_pivot_queue, and array are ready to repeat the steps in the first and second kernel.
*
*****************************************************************************************************************************/
__global__
void arrangeQueueAndPivot(int *new_pivot_queue, int *old_pivot_queue, int n, int *sort_array, int *pivot_arr, int *new_working_queue, int *old_working_queue, bool* done)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    __shared__ int shared_old_working_queue[256];

    if(tid < n)
    {
        int q_tid = 2*tid;

        shared_old_working_queue[threadIdx.x] = old_working_queue[q_tid];
        //__syncthreads();
        
        int head = 2*shared_old_working_queue[threadIdx.x];
        int my_pivot_index = old_pivot_queue[head];

        // 1. Store the pivots
        if(pivot_arr[tid] != -1)
        {
                sort_array[my_pivot_index] = pivot_arr[tid];
        }

        // 2. Rearrange the queue
        if(tid == my_pivot_index) // pivot use q
        {
			new_pivot_queue[q_tid] = tid;
			new_pivot_queue[q_tid+1] = tid;
			new_working_queue[q_tid] = tid;
			new_working_queue[q_tid+1] = tid;
        }
        else if(tid == shared_old_working_queue[threadIdx.x])
        {
            // starting range changed
            new_pivot_queue[q_tid] = shared_old_working_queue[threadIdx.x];
            new_pivot_queue[q_tid+1] = my_pivot_index-1;
            new_working_queue[q_tid] = shared_old_working_queue[threadIdx.x];
            new_working_queue[q_tid+1] = my_pivot_index-1;
        }
        else if(tid == my_pivot_index + 1)
        {
            // next pivot range changed
            new_pivot_queue[q_tid] = my_pivot_index+1;
            new_pivot_queue[q_tid+1] = old_working_queue[head+1];
            new_working_queue[q_tid] = my_pivot_index+1;
            new_working_queue[q_tid+1] = old_working_queue[head+1];
        }
        else
        {
            if(tid > my_pivot_index) 
            {
                new_pivot_queue[q_tid] = my_pivot_index + 1;
                new_pivot_queue[q_tid+1] = -1;
                new_working_queue[q_tid] = my_pivot_index + 1;
                new_working_queue[q_tid+1] = -1;
            }
            else if(tid < my_pivot_index)
            {
                new_pivot_queue[q_tid] = shared_old_working_queue[threadIdx.x];
                new_pivot_queue[q_tid+1] = -1;
                new_working_queue[q_tid] = shared_old_working_queue[threadIdx.x];;
                new_working_queue[q_tid+1] = -1;
            }
	}
		
        #if PICK_PIVOT
        int pid = 2*new_working_queue[q_tid];

        // 3. Select a better pivot selection and put the selected pivot to the first index of sub-arrays
        if(pid == tid*2)
        {
	    int start = new_working_queue[pid];
	    int end = new_working_queue[pid+1];
            if(end - start >= PIVOT_CANDIDATE)
            {
                int pivot_cand1 = sort_array[start];
                int pivot_cand2 = sort_array[(end+start)/2];
                int pivot_cand3 = sort_array[end];
                int pivot = 0;
                for(int i=0; i<PIVOT_CANDIDATE; i++)
                {
                    pivot = sort_array[new_working_queue[pid]+i];
                    bool found_pivot = false;
                    switch(i) 
                    {
                        case 0:
                            if((pivot >= pivot_cand2 && pivot <= pivot_cand3) || (pivot >= pivot_cand3 && pivot <= pivot_cand2))
                            {
                                found_pivot = true;
                            }
                            break;
                        case 1:
                            if((pivot >= pivot_cand1 && pivot <= pivot_cand3) || (pivot >= pivot_cand3 && pivot <= pivot_cand1))
                            {
                                found_pivot = true;
                            }
                            break;
                        case 2:
                            if((pivot >= pivot_cand1 && pivot <= pivot_cand2) || (pivot >= pivot_cand2 && pivot <= pivot_cand1))
                            {
                                found_pivot = true;
                            }
                            break;
                    }
                    if(found_pivot)
                    {
                        gpuSwap(&sort_array[new_working_queue[pid]+i],&sort_array[new_working_queue[pid]]);
                        break;
                    }
                }
            }
        }
        #endif
        
        if(new_pivot_queue[q_tid] != new_pivot_queue[q_tid+1])
        {
            *done = false;
        }
    }

}


int* quicksort_kernel(int* list1, int size)
{
    struct timespec start, stop;
    
    int * list2;
    int *pivot_arr;
    bool *done;
    int *q;
    int *q_copy;
    int *q2;
    int *q_copy2;
    int head = 0;
    int tail = size - 1;
    
    int blockSize = 256;
    int numBlocks = (size+(blockSize-1)) / blockSize;

    cudaMallocManaged(&list2, size*sizeof(int));
    cudaMallocManaged(&pivot_arr, size*sizeof(int));
    cudaMallocManaged(&q, 2*size*sizeof(int));
    cudaMallocManaged(&q_copy, 2*size*sizeof(int));
    cudaMallocManaged(&q2, 2*size*sizeof(int));
    cudaMallocManaged(&q_copy2, 2*size*sizeof(int));
    cudaMallocManaged(&done, sizeof(bool));

    cudaMemcpy(list2, list1, size*sizeof(int), cudaMemcpyDeviceToDevice);

    for(int i=0; i<size; i++)
    {
        pivot_arr[i] = -1;
    }

    q[0] = head;
    q[1] = tail;

    for(int i=2; i<2*size; i++)
    {
        if(i % 2 == 0)
        {
            q[i] = head;
        }
        else
        {
            q[i] = -1;
        }
    }

    cudaMemcpy(q2, q, 2*size*sizeof(int), cudaMemcpyDeviceToDevice);
    cudaMemcpy(q_copy, q, 2*size*sizeof(int), cudaMemcpyDeviceToDevice);
    cudaMemcpy(q_copy2, q, 2*size*sizeof(int), cudaMemcpyDeviceToDevice);

    int count = 0;

    /*********************************
    * Quicksort Start
    **********************************/
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &start);
    while(!(*done))
    {
        *done = true;

        if(count%2 == 0) 
        {
            quickSort<<<numBlocks, blockSize>>>(size, list1, list2, q, q_copy, pivot_arr);
            checkCudaError(cudaDeviceSynchronize());
            
            arrangeQueueAndPivot<<<numBlocks, blockSize>>>(q2, q, size, list2, pivot_arr, q_copy2, q_copy, done);
            checkCudaError(cudaDeviceSynchronize());
        }
        else
        {
            quickSort<<<numBlocks, blockSize>>>(size, list2, list1, q2, q_copy2, pivot_arr);
            checkCudaError(cudaDeviceSynchronize());
            
            arrangeQueueAndPivot<<<numBlocks, blockSize>>>(q, q2, size, list1, pivot_arr, q_copy, q_copy2, done);
            checkCudaError(cudaDeviceSynchronize());
        }

        ++count;
    }

    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &stop);
    /*********************************
    * Quicksort End
    **********************************/
    // We used this time elapse checking method for every comparison case
    double result = (stop.tv_sec - start.tv_sec) * 1e3 + (stop.tv_nsec - start.tv_nsec) / 1e6;
    printf("TIME TAKEN for QuickSort(GPU): %fms\n", result);

    cudaFree(done);
    cudaFree(pivot_arr);
    cudaFree(q);
    cudaFree(q_copy);
    cudaFree(q_copy2);
    cudaFree(q2);

    if(count % 2 == 0)
	return list1;
    else
	return list2; 
}

