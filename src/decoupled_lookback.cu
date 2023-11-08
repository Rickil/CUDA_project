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
        s[blockSize+1] = 0;
    }

    __syncthreads();

    unsigned int blockId = s[blockSize];
    unsigned int id = threadIdx.x + blockId * blockDim.x;

    if (id >= size)
        return;

    if (tid == 0)
        blockStatuses[blockId].status = STATUS_X;

    __syncthreads();

    //step 2 compute the aggregate
    //blockStatuses[blockId].aggregate +=
    //
    atomicAdd(&s[0], buffer[id]);
    __syncthreads();

    blockStatuses[blockId].aggregate = s[0];
    //atomicAdd(&blockStatuses[blockId].aggregate, buffer[id]);

    __syncthreads();
    //if (tid == 0)
    if (tid == 0) {
        if (blockId == 0) {
            blockStatuses[blockId].status = STATUS_P;
            blockStatuses[blockId].prefix_sum = 0;
        } else {
            blockStatuses[blockId].status = STATUS_A;
            //printf("block id: %d\n", blockId);
        }
    }

    __syncthreads();


    //step 3 look back
    int prevBlock = blockId - 1;

    if (tid == 0) {
        while (prevBlock >= 0) {

            //wait for the prev block to leave STATUS_X
            while (blockStatuses[prevBlock].status == STATUS_X) {
                //printf("block %d blockId %d prevBlock %d\n", blockIdx.x, blockId,prevBlock);
            };

            if (blockStatuses[prevBlock].status == STATUS_A) {
                blockStatuses[blockId].aggregate+= blockStatuses[prevBlock].aggregate;
                if (blockId==2)
                    printf("block %d found block %d in the A state", blockId, prevBlock);
                //atomicAdd(&blockStatuses[blockId].aggregate, blockStatuses[prevBlock].aggregate);

            } else if (blockStatuses[prevBlock].status == STATUS_P) {
                s[blockSize+1] = blockStatuses[prevBlock].prefix_sum;
                if (blockId==2)
                    printf("block %d found block %d in the P state", blockId, prevBlock);
                break;
            }

            /*s[blockSize+1]+=blockStatuses[prevBlock].aggregate;
            blockStatuses[blockId].aggregate = s[blockSize+1];*/
            prevBlock--;
        }

        blockStatuses[blockId].prefix_sum = blockStatuses[blockId].aggregate + s[blockSize+1];
        blockStatuses[blockId].status = STATUS_P;
    }
    __syncthreads();
    //if(blockId)
    //printf("s[blocksize+1]: %d\n", s[blockSize+1]);

    //step 5 compute scan
    s[tid] = (tid > 0) ? buffer[id - 1] : 0;  // Initialize the first element to 0 for inclusive scan
    __syncthreads();

    for (int i = 1; i < size; i *= 2) {
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
    cudaMemset(global_counter, -1, sizeof(int));

    //create an array of blockStatus and init all attributes to 0
    BlockStatus* blockStatuses;
    cudaMalloc(&blockStatuses, nbBlocks*sizeof(BlockStatus));
    cudaMemset(blockStatuses, 0, nbBlocks*sizeof(BlockStatus));

    //launch kernel
    decoupled_lookback_kernel<<<nbBlocks, blockSize, (blockSize + 2)*sizeof(int)>>>(buffer, blockStatuses, global_counter, size);
    cudaCheckError();
}