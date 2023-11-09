#define STATUS_X 0
#define STATUS_A 1
#define STATUS_P 2

#include "decoupled_lookback.cuh"

struct BlockStatus {
    int status;
    int aggregate;
    int prefix_sum;
};

__global__ void decoupled_lookback_kernel(int* buffer, volatile BlockStatus* blockStatuses, int* global_counter, int size){


    extern __shared__ int s[];

    //first step init flags and synchronize
    unsigned int tid = threadIdx.x;
    unsigned int blockSize = blockDim.x;


    //get the id with atomicAdd
    if (tid == 0) {
        s[blockSize] = atomicAdd(global_counter, 1);
        s[0] = 0;
        s[blockSize+1] = 0;
    }

    __syncthreads();

    unsigned int blockId = s[blockSize];
    unsigned int id = threadIdx.x + blockId * blockDim.x;

    if (id >= size)
        return;

    __syncthreads();

    //step 2 compute the aggregate
    atomicAdd(&s[0], buffer[id]);
    __syncthreads();

    if (tid == 0) {

        blockStatuses[blockId].aggregate = s[0];

        if (blockId == 0) {
            atomicCAS((int*)&blockStatuses[blockId].status, STATUS_X, STATUS_P);
            blockStatuses[blockId].prefix_sum = 0;
        } else {
            atomicCAS((int*)&blockStatuses[blockId].status, STATUS_X, STATUS_A);
        }

        //step 3 look back
        int prevBlock = blockId - 1;
        unsigned int prefix_sum_accumulation = 0;

        while (prevBlock >= 0) {

            //wait for the prev block to leave STATUS_X
            while (atomicCAS((int*)&blockStatuses[prevBlock].status, STATUS_X, STATUS_X) == STATUS_X) {
                //printf("block %d blockId %d prevBlock %d\n", blockIdx.x, blockId,prevBlock);
            };

            if (blockStatuses[prevBlock].status == STATUS_A) {
                prefix_sum_accumulation += blockStatuses[prevBlock].aggregate;


            } else if (blockStatuses[prevBlock].status == STATUS_P) {
                blockStatuses[blockId].prefix_sum = blockStatuses[prevBlock].prefix_sum + blockStatuses[prevBlock].aggregate + prefix_sum_accumulation;
                blockStatuses[blockId].status = STATUS_P;

                //we store the prefix_sul in the shared memory to ease the access for the block-scan
                s[blockSize+1] = blockStatuses[blockId].prefix_sum;

                break;
            }
            prevBlock--;
        }
    }
    __syncthreads();

    //step 5 compute scan
    s[tid] = (tid > 0) ? buffer[id - 1] : 0;  // Initialize the first element to 0 for inclusive scan
    __syncthreads();

    for (int i = 1; i < blockSize; i *= 2) {
        int x;
        if (i <= tid) {
            x = s[tid - i];
        }
        __syncthreads();
        if (i <= tid) {
            s[tid] += x;
        }
        __syncthreads();

    }

    buffer[id] = s[tid] + s[blockSize+1];
}

void decoupled_lookback(int* buffer, int size){

    int blockSize = 1024;  // Set your desired block size
    int nbBlocks = (size + blockSize - 1) / blockSize;

    //global counter to replace blockIdx.x
    int* global_counter;
    cudaMalloc(&global_counter, sizeof(int));
    cudaMemset(global_counter, 0, sizeof(int));

    //create an array of blockStatus and init all attributes to 0
    BlockStatus* blockStatuses;
    cudaMalloc(&blockStatuses, nbBlocks*sizeof(BlockStatus));
    cudaMemset(blockStatuses, 0, nbBlocks*sizeof(BlockStatus));

    //launch kernel
    decoupled_lookback_kernel<<<nbBlocks, blockSize, (blockSize + 2)*sizeof(int)>>>(buffer, blockStatuses, global_counter, size);
    cudaCheckError();

    cudaFree(blockStatuses);
    cudaFree(global_counter);
}