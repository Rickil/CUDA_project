#include "fix_gpu.cuh"

__global__ void predicate_kernel(int* buffer, int* predicate, int size, int garbage_val) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        predicate[i] = (buffer[i] != garbage_val) ? 1 : 0;
    }
}

__global__ void exclusive_scan_kernel(int* buffer, int size) {
    unsigned int id = threadIdx.x;
    extern __shared__ int s[];
    s[id] = buffer[id];

    // Perform scan within the block
    if (id < size) {
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
    }
}

void exclusive_scan_single_block(int* d_buffer, int size) {
    int blockSize = 256;  // Adjust this based on your GPU's capabilities

    // Allocate shared memory for the block
    int sharedMemSize = sizeof(int) * blockSize;

    // Call the exclusive_scan_kernel for a single block
    exclusive_scan_kernel<<<1, blockSize, sharedMemSize>>>(d_buffer, size);
}

// Example of how to perform exclusive scan on 1000 blocks
void exclusive_scan_1000_blocks(int* d_buffer, int size, int numBlocks) {
    int blockSize = 256;  // Adjust this based on your GPU's capabilities

    // Calculate the number of blocks required to scan the entire array
    int gridDim = (size + blockSize - 1) / blockSize;

    // Allocate shared memory for each block
    int sharedMemSize = sizeof(int) * blockSize;

    // Perform exclusive scan on each block
    for (int blockId = 0; blockId < numBlocks; blockId++) {
        exclusive_scan_kernel<<<1, blockSize, sharedMemSize>>>(d_buffer + blockId * blockSize, size);
    }

    // Wait for the GPU to finish
}

__global__ void scatter_kernel(int* buffer, int* scan_result, int* output, int size, int garbage_val) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        if (buffer[i] != garbage_val) {
            output[scan_result[i]] = buffer[i];
        }
    }
}

// Fonction pour appeler les kernels
void compact_image_gpu(int* d_buffer, int image_size) {
    constexpr int garbage_val = -27;
    int *d_predicate, *d_scan_result, *d_output;
    cudaMalloc(&d_predicate, image_size * sizeof(int));
    cudaMalloc(&d_scan_result, image_size * sizeof(int));
    cudaMalloc(&d_output, image_size * sizeof(int));

    std::vector<int> h_predicate;
    h_predicate.resize(image_size);

    int blockSize = 256;
    int gridSize = (image_size + blockSize - 1) / blockSize;

    predicate_kernel<<<gridSize, blockSize>>>(d_buffer, d_predicate, image_size, garbage_val);

    //GPU
    /*int *blockSums;
    cudaMalloc(&blockSums, blockSize * sizeof(int));*/

    //exclusive_scan_1000_blocks(d_predicate, /*d_scan_result,*/ image_size, gridSize);

    //CPU
    cudaMemcpy(h_predicate.data(), d_predicate, image_size * sizeof(int), cudaMemcpyDeviceToHost);
    std::exclusive_scan(h_predicate.begin(), h_predicate.end(), h_predicate.begin(), 0);
    cudaMemcpy(d_predicate, h_predicate.data(), image_size * sizeof(int), cudaMemcpyHostToDevice);

    scatter_kernel<<<gridSize, blockSize>>>(d_buffer, d_predicate, d_output, image_size, garbage_val);

    cudaMemcpy(d_buffer, d_output, image_size * sizeof(int), cudaMemcpyDeviceToDevice);

    cudaFree(d_predicate);
    cudaFree(d_scan_result);
    cudaFree(d_output);
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
        atomicAdd(&(histogram[buffer[i]]), 1);
    }
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
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Initialize the output to -1 (indicating not found)
    if (tid == 0)
        *cdf_min = -1;

    __syncthreads();

    // Search for the first non-zero element
    if (tid < size && *cdf_min == -1) {
        if (d_histogram[tid] != 0) {
            atomicCAS(cdf_min, -1, tid);
        }
    }
}

void findFirstNonZero(int *d_histogram, int *cdf_min, int image_size){
    int blockSize = 256;
    int gridSize = (image_size + blockSize - 1) / blockSize;
    findFirstNonZero_kernel<<<gridSize, blockSize>>>(d_histogram, cdf_min, image_size);
}

Image fix_image_gpu(Image image){
    int* d_buffer;
    int* d_histogram;
    int* cdf_min;
    int histogram_size = 256;
    cudaMalloc(&d_buffer, image.size() * sizeof(int));
    cudaMalloc(&d_histogram, histogram_size*sizeof(int));
    cudaMalloc(&cdf_min, sizeof(int));


    cudaMemcpy(d_buffer, image.buffer, image.size() * sizeof(int), cudaMemcpyHostToDevice);

    compact_image_gpu(d_buffer, image.size());
    apply_map_to_pixels_gpu(d_buffer, image.size());
    /*calculate_histogram_gpu(d_buffer, d_histogram, image.size());
    findFirstNonZero(d_histogram, cdf_min, image.size());
    equalize_histogram_gpu(d_buffer, d_histogram, image.size(), cdf_min);*/

    cudaMemcpy(image.buffer, d_buffer, image.size() * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_buffer);
    cudaFree(d_histogram);
    cudaFree(cdf_min);

    //std::cout << "hey\n";

    return image;
}