#include "gpu_scan.cuh"

void inclusive_scan_big(int *d_buffer, int size);

//#####EXCLUSIVE SCAN

// Step 1: Store Block Sum to Auxiliary Array
__global__ void storeBlockSum(int* buffer, int* blockSums, int size, int blockSize) {
    unsigned int id = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int tid = threadIdx.x;
    extern __shared__ int s[];

    if (id >= size)
        return;

    s[tid] = (tid > 0) ? buffer[id - 1] : 0;  // Initialize the first element to 0 for inclusive scan
    __syncthreads();

    for (int i = 1; i < blockSize; i *= 2) {
        int x;
        if (i <= tid) {
            x = s[tid - i];
        }
        __syncthreads();
        if (i<=tid) {
            s[tid] += x;
        }
        __syncthreads();
    }

    atomicAdd(&blockSums[blockIdx.x], buffer[id]);
    __syncthreads();

    buffer[id] = s[tid];
    __syncthreads();

    /*if (tid == blockSize - 1 || (id == size-1 && tid != blockSize - 1)) {
        printf("block %d, sum: %d\n", blockIdx.x, s[tid]);
        // Store the block sum in the blockSums array
        blockSums[blockIdx.x] = s[tid];
    }*/
}

__global__ void scan_kernel(int* buffer, int size) {
    unsigned int id = threadIdx.x;
    extern __shared__ int s[];
    s[id] = buffer[id];
    __syncthreads();

    if (id >= size)
        return;

    for (int i = 1; i < size; i *= 2) {
        int x;
        if (i <= id) {
            x = s[id - i];
        }
        __syncthreads();
        if (i <= id) {
            s[id] += x;
        }
        __syncthreads();

    }
    buffer[id] = s[id];
    //printf("index %d, value %d\n", id, buffer[id]);
}

// Step 2: Scan Block Sums (recursively if needed)
void scanBlockSums(int* blockSums, int numBlocks, int blockSize) {
    if (numBlocks <= 1) {
        // If there's only one block sum, no need to scan it further
        return;
    }

    if (numBlocks <= blockSize) {
        // Launch scan_kernel for blockSums
        scan_kernel<<<1, numBlocks, numBlocks * sizeof(int)>>>(blockSums, numBlocks);
    }else {
        // Recursively scan blockSums if necessary
        //scanBlockSums(blockSums, (numBlocks + blockSize - 1) / blockSize, blockSize);
        inclusive_scan_big(blockSums, numBlocks);
    }
}

// Step 3: Add Scanned Block Sum i to All Values of Scanned Block i + 1
__global__ void addScannedBlockSum(int* buffer, int* blockSums, int size) {
    unsigned int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id >= size)
        return ;
    if (blockIdx.x > 0) {
        // Add the scanned block sum of the previous block
        buffer[id] += blockSums[blockIdx.x - 1];
    }
}

void exclusive_scan(int *d_buffer, int size){
    int blockSize = 1024;  // Set your desired block size
    int numBlocks = (size + blockSize - 1) / blockSize;

    int* d_blockSums;
    cudaMalloc(&d_blockSums, numBlocks*sizeof(int));
    cudaMemset(d_blockSums, 0, numBlocks*sizeof(int));


    // Step 1: Store Block Sum to Auxiliary Array
    storeBlockSum<<<numBlocks, blockSize, blockSize * sizeof(int)>>>(d_buffer, d_blockSums, size, blockSize);

    // Step 2: Scan Block Sums (recursively if needed)
    scanBlockSums(d_blockSums, numBlocks, blockSize);

    // Step 3: Add Scanned Block Sum i to All Values of Scanned Block i + 1
    addScannedBlockSum<<<numBlocks, blockSize>>>(d_buffer, d_blockSums, size);

    cudaFree(d_blockSums);
}

//#####END EXCLUSIVE SCAN

//######INCLUSIVE SCAN

// Step 1: Store Block Sum to Auxiliary Array
__global__ void storeBlockSumInclusive(int* buffer, int* blockSums, int size, int blockSize) {
    unsigned int id = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int tid = threadIdx.x;
    extern __shared__ int s[];

    if (id >= size)
        return;

    s[tid] = buffer[id];
    __syncthreads();

    for (int i = 1; i < blockSize; i *= 2) {
        int x;
        if (i <= tid) {
            x = s[tid - i];
        }
        __syncthreads();
        if (i<=tid) {
            s[tid] += x;
        }
        __syncthreads();
    }

    atomicAdd(&blockSums[blockIdx.x], buffer[id]);
    __syncthreads();

    buffer[id] = s[tid];
    __syncthreads();

    /*if (tid == blockSize - 1 || (id == size-1 && tid != blockSize - 1)) {
        printf("block %d, sum: %d\n", blockIdx.x, s[tid]);
        // Store the block sum in the blockSums array
        blockSums[blockIdx.x] = s[tid];
    }*/
}

void inclusive_scan_big(int *d_buffer, int size){
    int blockSize = 1024;  // Set your desired block size
    int numBlocks = (size + blockSize - 1) / blockSize;

    int* d_blockSums;
    cudaMalloc(&d_blockSums, numBlocks*sizeof(int));
    cudaMemset(d_blockSums, 0, numBlocks*sizeof(int));


    // Step 1: Store Block Sum to Auxiliary Array
    storeBlockSumInclusive<<<numBlocks, blockSize, blockSize * sizeof(int)>>>(d_buffer, d_blockSums, size, blockSize);

    // Step 2: Scan Block Sums (recursively if needed)
    scanBlockSums(d_blockSums, numBlocks, blockSize);

    // Step 3: Add Scanned Block Sum i to All Values of Scanned Block i + 1
    addScannedBlockSum<<<numBlocks, blockSize>>>(d_buffer, d_blockSums, size);

    cudaFree(d_blockSums);
}
//#####END INCLUSIVE SCAN