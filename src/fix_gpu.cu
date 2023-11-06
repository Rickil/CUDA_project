#include "fix_gpu.cuh"

// decoupled look_back
#define STATUS_X 0
#define STATUS_AGGREGATE_AVAILABLE 1
#define STATUS_PREFIX_SUM_AVAILABLE 2

__global__ void predicate_kernel(int* buffer, int* out, int size) {
    /*int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size-1 && i > 0) {
        if (buffer[i] != garbage_val)
            predicate[i] = 1;
        else
            predicate[i] = 0;
    }*/

    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id >= size)
        return;

    constexpr int garbage = -27;
    int inter = buffer[id];
    __syncthreads();

    out[id] = inter != garbage ? 1 : 0;
}

void check_predicate(int* d_buffer, int* d_predicate, int size){
    std::vector<int> h_buffer(size, 0);
    cudaMemcpy(h_buffer.data(), d_buffer, size*sizeof(int), cudaMemcpyDeviceToHost);
    std::vector<int> gpu_predicate(size, 0);
    cudaMemcpy(gpu_predicate.data(), d_predicate, size*sizeof(int), cudaMemcpyDeviceToHost);

    std::vector<int> cpu_predicate(size, 0);
    constexpr int garbage_val = -27;
    int count_garbage = 0;
    for (int i = 0; i < size; ++i) {
        if (h_buffer[i] != garbage_val)
            cpu_predicate[i] = 1;
        else
            count_garbage++;
    }

    printf("cpu_size: %lu, gpu_size: %lu, garbage_count: %d\n", cpu_predicate.size(), gpu_predicate.size(), count_garbage);

    bool same = true;
    int count = 0;
    for (int i = 0; i < size; i++){
        if (cpu_predicate[i] != gpu_predicate[i]){
            same = false;
            count++;
            //printf("index: %d, cpu: %d, gpu: %d\n", i, cpu_predicate[i], gpu_predicate[i]);
        }
    }

    if (same)
        printf("predicate good !\n");
    else
        printf("predicate bad !, %d are bad, %i\n", count, (count/size)*100);
}

void check_scan(int* d_predicate, int* d_scan_result, int size){
    std::vector<int> h_scan_result(size, 0);
    cudaMemcpy(h_scan_result.data(), d_scan_result, size*sizeof(int), cudaMemcpyDeviceToHost);

    std::vector<int> h_predicate(size, 0);
    cudaMemcpy(h_predicate.data(), d_predicate, size * sizeof(int), cudaMemcpyDeviceToHost);
    std::exclusive_scan(h_predicate.begin(), h_predicate.end(), h_predicate.begin(), 0);


    bool same = true;
    int count = 0;
    for (int i = 0; i < size; i++){
        if (h_predicate[i] != h_scan_result[i]){
            same = false;
            count++;
            //printf("index: %d, cpu: %d, gpu: %d\n", i, h_predicate[i], h_scan_result[i]);
        }
    }

    if (same)
        printf("scan good !\n");
    else
        printf("scan bad !, %d are bad, %i\n", count, (count/size)*100);

}

void check_scatter(int *my_d_buffer, int *d_buffer, int *d_predicate, int size, int compact_size){
    std::vector<int> h_buffer(size, 0);
    cudaMemcpy(h_buffer.data(), d_buffer, size*sizeof(int), cudaMemcpyDeviceToHost);
    std::vector<int> my_h_buffer(size, 0);
    cudaMemcpy(my_h_buffer.data(), my_d_buffer, size*sizeof(int), cudaMemcpyDeviceToHost);
    std::vector<int> h_predicate(size, 0);
    cudaMemcpy(h_predicate.data(), d_predicate, size*sizeof(int), cudaMemcpyDeviceToHost);

    constexpr int garbage_val = -27;
    for (std::size_t i = 0; i < size; ++i) {
        if (h_buffer[i] != garbage_val) {
            h_buffer[h_predicate[i]] = h_buffer[i];
        }
    }

    bool same = true;
    int count = 0;
    for (int i = 0; i < size; i++){
        if (h_buffer[i] != my_h_buffer[i]){
            same = false;
            count++;
            //printf("index: %d, cpu: %d, gpu: %d\n", i, h_buffer[i], my_h_buffer[i]);
        }
    }

    if (same)
        printf("scatter good !\n");
    else
        printf("scatter bad !, %d are bad, %f\n", count, ((float)count/size)*100);
}

typedef struct TileStatus
{
    int flags;
    int aggregate;
    int inclusive_prefix;
} TileStatus;


template <int BLOCKSIZE>
__device__ static inline void warp_reduce_template(int* sdata, int tid)
{
    if constexpr (BLOCKSIZE >= 64) {sdata[tid] += sdata[tid + 32]; __syncwarp();}
    if constexpr (BLOCKSIZE >= 32) {sdata[tid] += sdata[tid + 16]; __syncwarp();}
    if constexpr (BLOCKSIZE >= 16) {sdata[tid] += sdata[tid + 8]; __syncwarp();}
    if constexpr (BLOCKSIZE >= 8)  {sdata[tid] += sdata[tid + 4]; __syncwarp();}
    if constexpr (BLOCKSIZE >= 4)  {sdata[tid] += sdata[tid + 2]; __syncwarp();}
    if constexpr (BLOCKSIZE >= 2)  {sdata[tid] += sdata[tid + 1]; __syncwarp();}
}

// TileStatus statuses[gridSize];
template <int blocksize>
__global__ void decoupled_look_back_scan_kernel(int *out, int* predicate, int size,
                                                  volatile TileStatus *statuses, int *id_)
{
    extern __shared__ int blockId[]; // + exlcusif prefix + block id
    int tid = threadIdx.x;
    if (tid == 0)
    {
        blockId[blocksize + 1] = 0;
        blockId[blocksize] = atomicAdd(id_, 1);
    }

    __syncthreads();
    //if (id >= size)
    //    return;
    int id = tid + blockId[blocksize] * blockDim.x;
    if (id >= size)
    {
        return;
    }
    __syncthreads();

    int inter = predicate[id];
    blockId[tid] = inter;
    __syncthreads();

    if constexpr (blocksize >= 1024)
    {
        if (tid < 512)
        {blockId[tid] += blockId[tid + 512]; }
        __syncthreads();
    }
    if constexpr (blocksize >= 512)
    {
        if (tid < 256)
        {blockId[tid] += blockId[tid + 256];}
        __syncthreads();
    }
    if constexpr (blocksize >= 256)
    {
        if (tid < 128)
        {
            blockId[tid] += blockId[tid + 128];
        }
        __syncthreads();
    }

    if constexpr (blocksize >= 128)
    {
        if (tid < 64){blockId[tid] += blockId[tid + 64]; }
        __syncthreads();
    }

    if (tid < 32)
    {
        warp_reduce_template<blocksize>(blockId, tid);
    }
    __syncthreads();

    if (tid == 0)
    {
        statuses[blockId[blocksize]].aggregate =  blockId[0];
        statuses[blockId[blocksize]].flags= STATUS_AGGREGATE_AVAILABLE;
    }

    __syncthreads();

    // Only need one thread by block here.
    // Can use multiple to lookback N block at a time
    if (blockId[blocksize] == 0)
    {

        if (tid == 0)
        {
            statuses[blockId[blocksize]].inclusive_prefix = statuses[blockId[blocksize]].aggregate;
            statuses[blockId[blocksize]].flags= STATUS_PREFIX_SUM_AVAILABLE;
            //printf("Block id =%i, block_agregate=%i\n", blockId[blocksize], statuses[blockId[blocksize]].aggregate);
            //printf("Block id =%i, block_sum=%i\n", blockId[blocksize], blockId[blocksize + 1]);
        }
    }
    else // look-back
    {
        if (tid == 0)
        {
            statuses[blockId[blocksize]].flags= STATUS_AGGREGATE_AVAILABLE;
            int prev_block = blockId[blocksize] - 1;
            //printf("Block id =%i, previous block=%i\n", blockId[blocksize], prev_block);
            while (prev_block >= 0) // go back
            {
                while (statuses[prev_block].flags == STATUS_X) // block
                {
                };
                if (statuses[prev_block].flags == STATUS_PREFIX_SUM_AVAILABLE)
                {
                    blockId[blocksize + 1] += statuses[prev_block].inclusive_prefix;
                    break;
                }
                // STATUS_AGGREGATE_AVAILABLE
                blockId[blocksize + 1] += statuses[prev_block--].aggregate;
            }
            // End == STATUS_PREFIX_SUM_AVAILABLE
            // Since block 0 is initialized, its ok
            //printf("Block id =%i, block_sum=%i\n", blockId[blocksize], blockId[blocksize + 1]);
            //printf("Block id =%i, block_agregate=%i\n", blockId[blocksize], statuses[blockId[blocksize]].aggregate);
        }
        __syncthreads();

        statuses[blockId[blocksize]].inclusive_prefix =
                statuses[blockId[blocksize]].aggregate + blockId[blocksize + 1];

        statuses[blockId[blocksize]].flags= STATUS_PREFIX_SUM_AVAILABLE;
    }
    __syncthreads();

    int prefix = predicate[id];
    blockId[tid] = prefix;

    __syncthreads();

    for (unsigned int stride = 1; stride < blockDim.x; stride *= 2)
    {
        int tmp_value = 0;
        int tmp_value2 = 0;
        if (tid >= stride)
        {
            tmp_value = blockId[tid - stride];
            tmp_value2 = blockId[tid];
        }
        __syncthreads();

        if (tid >= stride)
        {
            blockId[tid] = tmp_value + tmp_value2;
        }
        __syncthreads();

    }
    int prefix_sum = blockId[blocksize + 1];
    __syncthreads();
    //printf("prefix_sum=%i, block=%i\n", prefix_sum, blockId[blocksize]);
    if (id + 1 < size)
    {
        out[id + 1] = blockId[tid] + prefix_sum;
    }
}

//####NEW

// TileStatus statuses[gridSize];
template <int blocksize>
__global__ void decoupled_look_back_scan_exclusif(int *out, int* predicate, int size,
                                                  volatile TileStatus *statuses, int *id_)
{
    extern __shared__ int blockId[]; // + exlcusif prefix + block id
    int tid = threadIdx.x;
    if (tid == 0)
    {
        blockId[blocksize + 1] = 0;
        blockId[blocksize] = atomicAdd(id_, 1);
    }

    __syncthreads();
    //if (id >= size)
    //    return;
    int id = tid + blockId[blocksize] * blockDim.x;
    if (id >= size)
    {
        return;
    }
    __syncthreads();

    int inter = predicate[id];
    blockId[tid] = inter;
    __syncthreads();

    if constexpr (blocksize >= 1024)
    {
        if (tid < 512)
        {blockId[tid] += blockId[tid + 512]; }
        __syncthreads();
    }
    if constexpr (blocksize >= 512)
    {
        if (tid < 256)
        {blockId[tid] += blockId[tid + 256];}
        __syncthreads();
    }
    if constexpr (blocksize >= 256)
    {
        if (tid < 128)
        {
            blockId[tid] += blockId[tid + 128];
        }
        __syncthreads();
    }

    if constexpr (blocksize >= 128)
    {
        if (tid < 64){blockId[tid] += blockId[tid + 64]; }
        __syncthreads();
    }

    if (tid < 32)
    {
        warp_reduce_template<blocksize>(blockId, tid);
    }
    __syncthreads();

    if (tid == 0)
    {
        statuses[blockId[blocksize]].aggregate =  blockId[0];
        statuses[blockId[blocksize]].flags= STATUS_AGGREGATE_AVAILABLE;
    }

    __syncthreads();

    // Only need one thread by block here.
    // Can use multiple to lookback N block at a time
    if (blockId[blocksize] == 0)
    {

        if (tid == 0)
        {
            statuses[blockId[blocksize]].inclusive_prefix = statuses[blockId[blocksize]].aggregate;
            statuses[blockId[blocksize]].flags= STATUS_PREFIX_SUM_AVAILABLE;
            //printf("Block id =%i, block_agregate=%i\n", blockId[blocksize], statuses[blockId[blocksize]].aggregate);
            //printf("Block id =%i, block_sum=%i\n", blockId[blocksize], blockId[blocksize + 1]);
        }
    }
    else // look-back
    {
        if (tid == 0)
        {
            statuses[blockId[blocksize]].flags= STATUS_AGGREGATE_AVAILABLE;
            int prev_block = blockId[blocksize] - 1;
            //printf("Block id =%i, previous block=%i\n", blockId[blocksize], prev_block);
            while (prev_block >= 0) // go back
            {
                while (statuses[prev_block].flags == STATUS_X) // block
                {
                };
                if (statuses[prev_block].flags == STATUS_PREFIX_SUM_AVAILABLE)
                {
                    blockId[blocksize + 1] += statuses[prev_block].inclusive_prefix;
                    break;
                }
                // STATUS_AGGREGATE_AVAILABLE
                blockId[blocksize + 1] += statuses[prev_block--].aggregate;
            }
            // End == STATUS_PREFIX_SUM_AVAILABLE
            // Since block 0 is initialized, its ok
            //printf("Block id =%i, block_sum=%i\n", blockId[blocksize], blockId[blocksize + 1]);
            //printf("Block id =%i, block_agregate=%i\n", blockId[blocksize], statuses[blockId[blocksize]].aggregate);
        }
        __syncthreads();

        statuses[blockId[blocksize]].inclusive_prefix =
                statuses[blockId[blocksize]].aggregate + blockId[blocksize + 1];

        statuses[blockId[blocksize]].flags= STATUS_PREFIX_SUM_AVAILABLE;
    }
    __syncthreads();

    int prefix = predicate[id];
    blockId[tid] = prefix;

    __syncthreads();

    for (unsigned int stride = 1; stride < blockDim.x; stride *= 2)
    {
        int tmp_value = 0;
        int tmp_value2 = 0;
        if (tid >= stride)
        {
            tmp_value = blockId[tid - stride];
            tmp_value2 = blockId[tid];
        }
        __syncthreads();

        if (tid >= stride)
        {
            blockId[tid] = tmp_value + tmp_value2;
        }
        __syncthreads();

    }
    int prefix_sum = blockId[blocksize + 1];
    __syncthreads();
    //printf("prefix_sum=%i, block=%i\n", prefix_sum, blockId[blocksize]);
    if (id + 1 < size)
    {
        out[id + 1] = blockId[tid] + prefix_sum;
    }
}

// TileStatus statuses[gridSize];
template <int blocksize>
__global__ void decoupled_look_back_scan_inclusif(int *out, int* predicate, int size,
                                                  volatile TileStatus *statuses, int *id_)
{
    extern __shared__ int blockId[]; // + exlcusif prefix + block id
    int tid = threadIdx.x;
    if (tid == 0)
    {
        blockId[blocksize + 1] = 0;
        blockId[blocksize] = atomicAdd(id_, 1);
    }
    __syncthreads();
    int id = tid + blockId[blocksize] * blockDim.x;
    if (id >= size)
    {
        return;
    }
    __syncthreads();
    //if (id >= size)
    //    return;
    blockId[tid] = predicate[id];
    __syncthreads();

    if constexpr (blocksize >= 1024)
    {
        if (tid < 512)
        {blockId[tid] += blockId[tid + 512]; }
        __syncthreads();
    }
    if constexpr (blocksize >= 512)
    {
        if (tid < 256)
        {blockId[tid] += blockId[tid + 256];}
        __syncthreads();
    }
    if constexpr (blocksize >= 256)
    {
        if (tid < 128)
        {
            blockId[tid] += blockId[tid + 128];
        }
        __syncthreads();
    }

    if constexpr (blocksize >= 128)
    {
        if (tid < 64){blockId[tid] += blockId[tid + 64]; }
        __syncthreads();
    }

    if (tid < 32)
    {
        warp_reduce_template<blocksize>(blockId, tid);
    }
    __syncthreads();

    if (tid == 0)
    {
        statuses[blockId[blocksize]].aggregate =  blockId[0];
        statuses[blockId[blocksize]].flags= STATUS_AGGREGATE_AVAILABLE;
    }

    __syncthreads();

    // Only need one thread by block here.
    // Can use multiple to lookback N block at a time
    if (blockId[blocksize] == 0)
    {

        if (tid == 0)
        {
            statuses[blockId[blocksize]].inclusive_prefix = statuses[blockId[blocksize]].aggregate;
            statuses[blockId[blocksize]].flags= STATUS_PREFIX_SUM_AVAILABLE;
            //printf("Block id =%i, block_agregate=%i\n", blockId[blocksize], statuses[blockId[blocksize]].aggregate);
            //printf("Block id =%i, block_sum=%i\n", blockId[blocksize], blockId[blocksize + 1]);
        }
    }
    else // look-back
    {
        if (tid == 0)
        {
            statuses[blockId[blocksize]].flags= STATUS_AGGREGATE_AVAILABLE;
            int prev_block = blockId[blocksize] - 1;
            //printf("Block id =%i, previous block=%i\n", blockId[blocksize], prev_block);
            while (prev_block >= 0) // go back
            {
                while (statuses[prev_block].flags == STATUS_X) // block
                {
                };
                if (statuses[prev_block].flags == STATUS_PREFIX_SUM_AVAILABLE)
                {
                    blockId[blocksize + 1] += statuses[prev_block].inclusive_prefix;
                    break;
                }
                // STATUS_AGGREGATE_AVAILABLE
                blockId[blocksize + 1] += statuses[prev_block--].aggregate;
            }
            // End == STATUS_PREFIX_SUM_AVAILABLE
            // Since block 0 is initialized, its ok
            //printf("Block id =%i, block_sum=%i\n", blockId[blocksize], blockId[blocksize + 1]);
            //printf("Block id =%i, block_agregate=%i\n", blockId[blocksize], statuses[blockId[blocksize]].aggregate);
        }
        __syncthreads();

        statuses[blockId[blocksize]].inclusive_prefix =
                statuses[blockId[blocksize]].aggregate + blockId[blocksize + 1];

        statuses[blockId[blocksize]].flags= STATUS_PREFIX_SUM_AVAILABLE;
    }
    __syncthreads();

    int prefix = predicate[id];
    blockId[tid] = prefix;

    __syncthreads();

    for (unsigned int stride = 1; stride < blockDim.x; stride *= 2)
    {
        int tmp_value = 0;
        int tmp_value2 = 0;
        if (tid >= stride)
        {
            tmp_value = blockId[tid - stride];
            tmp_value2 = blockId[tid];
        }
        __syncthreads();

        if (tid >= stride)
        {
            blockId[tid] = tmp_value + tmp_value2;
        }
        __syncthreads();

    }
    int prefix_sum = blockId[blocksize + 1];
    __syncthreads();
    //printf("prefix_sum=%i, block=%i\n", prefix_sum, blockId[blocksize]);
    out[id] = blockId[tid] + prefix_sum;
}

//#####ENDNEW

void decoupled_look_back_scan(int* d_predicate, int *d_scan_result, int size){
    const int blockSize = 256;
    int gridSize = (size + blockSize - 1) / blockSize;

    int *lookBack_id;
    cudaMalloc(&lookBack_id, sizeof(int));
    cudaMemset(lookBack_id, 0, sizeof(int));

    TileStatus *statuses = nullptr;
    cudaMalloc(&statuses, gridSize * sizeof(TileStatus));
    cudaMemset(statuses, 0, gridSize * sizeof(TileStatus));

    decoupled_look_back_scan_kernel<blockSize><<<gridSize, blockSize, (blockSize + 2) * sizeof(int)>>>(
            d_scan_result, d_predicate, size, statuses, lookBack_id);
}


__global__ void scatter_kernel(int *buffer, int *exclusive_scan,
                               int *out_buffer, int size) {
    /*int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= size) {
        return;
    }

    int tmp = buffer[i];
    int pred_id = predicate[i];
    __syncthreads();
    if (tmp != garbage_val) {
        output[pred_id] = tmp;
    }*/

    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id >= size)
    {
        return;
    }

    int buffer_val = buffer[id];
    int exclu_idx = exclusive_scan[id];

    __syncthreads();

    if (buffer_val != -27)
    {
        out_buffer[exclu_idx] = buffer_val;
    }

}

// Fonction pour appeler les kernels
void compact_image_gpu(int* d_buffer, int* d_output, int image_size, int compact_size) {
    constexpr int garbage_val = -27;
    int *d_predicate, *d_scan_result;
    cudaMalloc(&d_predicate, image_size*sizeof(int));
    cudaMemset(d_predicate, 0, image_size*sizeof(int));
    cudaMalloc(&d_scan_result, image_size*sizeof(int));
    cudaMemset(d_scan_result, 0, image_size*sizeof(int));


    int blockSize = 256;
    int gridSize = (image_size + blockSize - 1) / blockSize;

    //copy the predicate for further tests
    int *d_buffer_copy;
    cudaMalloc(&d_buffer_copy, image_size*sizeof(int));
    cudaMemcpy(d_buffer_copy, d_buffer, image_size*sizeof(int), cudaMemcpyDeviceToDevice);

    predicate_kernel<<<gridSize, blockSize>>>(d_buffer, d_predicate, image_size);
    check_predicate(d_buffer_copy, d_predicate, image_size);

    int *d_predicate_copy;
    cudaMalloc(&d_predicate_copy, image_size*sizeof(int));
    cudaMemcpy(d_predicate_copy, d_predicate, image_size*sizeof(int), cudaMemcpyDeviceToDevice);

    decoupled_look_back_scan(d_predicate, d_scan_result, image_size);
    check_scan(d_predicate_copy, d_scan_result, image_size);

    cudaMemcpy(d_output, d_buffer, image_size*sizeof(int), cudaMemcpyDeviceToDevice);
    scatter_kernel<<<gridSize, blockSize>>>(d_buffer, d_scan_result, d_output, image_size);
    check_scatter(d_output, d_buffer_copy, d_scan_result, image_size, compact_size);

    cudaFree(d_predicate);
    cudaFree(d_scan_result);
}


__global__ void apply_map_kernel(int* buffer, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        if (i % 4 == 0) {
            buffer[i] += 1;
        } else if (i % 4 == 1) {
            buffer[i] -= 5;
        } else if (i % 4 == 2) {
            buffer[i] += 3;
        } else if (i % 4 == 3) {
            buffer[i] -= 8;
        }
    }
    /*if (buffer[i] < 0 || buffer[i] > 255) {
        printf("%d\n", buffer[i]);
    }*/
}

// Fonction pour appeler le kernel
void apply_map_to_pixels_gpu(int* d_buffer, int image_size) {
    int blockSize = 256;
    int gridSize = (image_size + blockSize - 1) / blockSize;
    apply_map_kernel<<<gridSize, blockSize>>>(d_buffer, image_size);
}

__global__ void calculate_histogram_kernel(int* buffer, int* histogram, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        /*if (buffer[i] < 0 || buffer[i] > 255) {
            printf("%d", buffer[i]);
        }*/
        int index = buffer[i];
        atomicAdd(&histogram[index], 1);
        //histogram[index] += 1;
    }
    __syncthreads();
}

__global__ void equalize_histogram_kernel(int* buffer, int* histogram, int size, int* cdf_min) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        int pixel_value = buffer[i];
        buffer[i] = roundf(((histogram[pixel_value] - *cdf_min) / (float)(size - *cdf_min)) * 255.0f);
    }
}

// Fonctions pour appeler les kernels
void calculate_histogram_gpu(int* d_buffer, int* d_histogram, int image_size) {
    int blockSize = 256;
    int gridSize = (image_size + blockSize - 1) / blockSize;
    calculate_histogram_kernel<<<gridSize, blockSize>>>(d_buffer, d_histogram, image_size);
}

void equalize_histogram_gpu(int* d_buffer, int* d_histogram, int image_size, int* cdf_min) {
    int blockSize = 256;
    int gridSize = (image_size + blockSize - 1) / blockSize;
    equalize_histogram_kernel<<<gridSize, blockSize>>>(d_buffer, d_histogram, image_size, cdf_min);
}

__global__ void findFirstNonZero_kernel(int *d_histogram, int *cdf_min, int size) {
    int idx = threadIdx.x + blockDim.x*blockIdx.x;
    if(idx < size && d_histogram[idx] != 0)
        atomicMin(cdf_min, idx);
}


void findFirstNonZero(int *d_histogram, int *cdf_min, int histogram_size){
    //int blockSize = 256;
    //int gridSize = (image_size + blockSize - 1) / blockSize;
    findFirstNonZero_kernel<<<1, histogram_size, histogram_size*sizeof(int)>>>(d_histogram, cdf_min, histogram_size);
}

void fix_image_gpu(Image& image){
    const int image_size = image.size();
    const int compact_size = image.width*image.height;

    int* d_buffer;
    int* d_output;
    int* d_histogram;
    int* cdf_min;
    int histogram_size = 256;
    cudaMalloc(&d_buffer, image_size*sizeof(int));
    cudaMalloc(&d_output, image_size*sizeof(int));
    cudaMemset(d_output, 0, image_size*sizeof(int));
    cudaMalloc(&d_histogram, histogram_size*sizeof(int));
    cudaMemset(d_histogram, 0, histogram_size*sizeof(int));
    cudaMalloc(&cdf_min, sizeof(int));
    cudaMemset(cdf_min, 255, sizeof(int));


    cudaMemcpy(d_buffer, image.buffer, image_size*sizeof(int), cudaMemcpyHostToDevice);

    compact_image_gpu(d_buffer, d_output, image_size, compact_size);
    apply_map_to_pixels_gpu(d_output, compact_size);

    /*calculate_histogram_gpu(d_output, d_histogram, compact_size);
    findFirstNonZero(d_histogram, cdf_min, histogram_size);
    equalize_histogram_gpu(d_output, d_histogram, compact_size, cdf_min);*/

    cudaMemcpy(image.buffer, d_output, image.width*image.height * sizeof(int), cudaMemcpyDeviceToHost);

    //check image_buffer
    int count = 0;
    for (int i=0; i<compact_size; i++){
        if (image.buffer[i] == -27){
            count++;
            //printf("value -27 at %d\n", i);
        }
    }
    printf("%d values at -27\n", count);

    cudaFree(d_buffer);
    cudaFree(d_output);
    cudaFree(d_histogram);
    cudaFree(cdf_min);

    //std::cout << "hey\n";

    //return image;
}