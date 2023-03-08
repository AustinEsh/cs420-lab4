/******************************************************************************
 *cr
 *cr            (C) Copyright 2010 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ******************************************************************************/

#define BLOCK_SIZE 512
#define SIMPLE

__global__ void reduction(float *out, float *in, unsigned size)
{
    /********************************************************************
    Load a segment of the input vector into shared memory
    Traverse the reduction tree
    Write the computed sum to the output vector at the correct index
    ********************************************************************/

    // INSERT KERNEL CODE HERE
    __shared__ float input_s[BLOCK_SIZE];
    unsigned int segment = 2 * blockDim.x * blockIdx.x;
    unsigned int t = segment + threadIdx.x;
    unsigned int i = threadIdx.x;
    input_s[i] = in[t] + in[t + BLOCK_SIZE];

    for (unsigned int stride = blockDim.x / 2; stride >= 1; stride /= 2){
        __syncthreads();
        if (i < stride){
            input_s[i] += input_s[i + stride];
        }
    }
    if (i == 0){
        atomicAdd(out, input_s[0]);
    }
}