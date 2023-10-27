#include "fix_gpu.cuh"

__global__ void predicate_kernel(int* buffer, int* predicate, int size, int garbage_val) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        predicate[i] = (buffer[i] != garbage_val) ? 1 : 0;
    }
}

__global__ void exclusive_scan_kernel(int* predicate, int* scan_result, int size) {
    // Un simple scan exclusif, pour une efficacité accrue, considérez un scan hiérarchique ou l'utilisation de bibliothèques existantes.
    if (threadIdx.x > 0 && threadIdx.x < size) {
        scan_result[threadIdx.x] = predicate[threadIdx.x - 1] + scan_result[threadIdx.x - 1];
    } else {
        scan_result[0] = 0;
    }
}

__global__ void scatter_kernel(int* buffer, int* scan_result, int size, int garbage_val) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        if (buffer[i] != garbage_val) {
            buffer[scan_result[i]] = buffer[i];
        }
    }
}

// Fonction pour appeler les kernels
void compact_image_gpu(int* d_buffer, int image_size) {
    constexpr int garbage_val = -27;
    int *d_predicate, *d_scan_result;
    cudaMalloc(&d_predicate, image_size * sizeof(int));
    cudaMalloc(&d_scan_result, image_size * sizeof(int));

    int blockSize = 256;
    int gridSize = (image_size + blockSize - 1) / blockSize;

    predicate_kernel<<<gridSize, blockSize>>>(d_buffer, d_predicate, image_size, garbage_val);
    exclusive_scan_kernel<<<1, image_size>>>(d_predicate, d_scan_result, image_size);
    scatter_kernel<<<gridSize, blockSize>>>(d_buffer, d_scan_result, image_size, garbage_val);

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
    cudaMalloc(&d_buffer, image.size() * sizeof(int));
    cudaMalloc(&d_histogram, 255*sizeof(int));
    cudaMalloc(&cdf_min, sizeof(int));
    cudaMemcpy(d_buffer, image.buffer, image.size() * sizeof(int), cudaMemcpyHostToDevice);

    compact_image_gpu(d_buffer, image.size());
    apply_map_to_pixels_gpu(d_buffer, image.size());
    calculate_histogram_gpu(d_buffer, d_histogram, image.size());
    findFirstNonZero(d_histogram, cdf_min, image.size());
    equalize_histogram_gpu(d_buffer, d_histogram, image.size(), cdf_min);

    cudaMemcpy(image.buffer, d_buffer, image.size() * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_buffer);
    cudaFree(d_histogram);
    cudaFree(cdf_min);

    //std::cout << "hey\n";

    return image;
}