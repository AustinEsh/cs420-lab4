/******************************************************************************
 *cr
 *cr            (C) Copyright 2010 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ******************************************************************************/

#define BLOCK_SIZE 512

// Define your kernels in this file you may use more than one kernel if you
// need to

// INSERT KERNEL(S) HERE


/******************************************************************************
Setup and invoke your kernel(s) in this function. You may also allocate more
GPU memory if you need to
*******************************************************************************/
__global__ void preScan(float *out, float *in, unsigned n)
{
    extern __shared__ float temp[];  // allocated on invocation
    int thid = threadIdx.x;
    int offset = 1;
    temp[2*thid] = in[2*thid]; // load input into shared memory
    temp[2*thid+1] = in[2*thid+1];
    for (int d = n>>1; d > 0; d >>= 1){                    // build sum in place up the tree
        __syncthreads();
        if (thid < d){ 
            int ai = offset*(2*thid+1)-1;
            int bi = offset*(2*thid+2)-1;  
            temp[bi] += temp[ai];
        }
        offset *= 2;
    }
    if (thid == 0) {
        temp[n - 1] = 0;
    } // clear the last element
    for (int d = 1; d < n; d *= 2){ // traverse down tree & build scan
        offset >>= 1;
        __syncthreads();
        if (thid < d){ 
            int ai = offset*(2*thid+1)-1;
            int bi = offset*(2*thid+2)-1; 
 	        float t = temp[ai];
            temp[ai] = temp[bi];
            temp[bi] += t;
        }
    }
    __syncthreads(); 
    out[2*thid] = temp[2*thid]; // write results to device memory
    out[2*thid+1] = temp[2*thid+1]; 	
}