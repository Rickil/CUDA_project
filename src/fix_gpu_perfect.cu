#include "fix_gpu_perfect.cuh"

cub::CachingDeviceAllocator  g_allocator(true);

__global__ void predicate_kernel(int* buffer, int* predicate, int size, int garbage_val) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        predicate[i] = (buffer[i] != garbage_val) ? 1 : 0;
    }
}

__global__ void scatter_kernel(int* buffer, int* scan_result, int* output, int size, int garbage_val) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        if (buffer[i] != garbage_val) {
            output[scan_result[i]] = buffer[i];
        }
    }
}

void compact_image_gpu(int* d_buffer, int image_size) {
    constexpr int garbage_val = -27;
    int *d_predicate, *d_scan_result, *d_output;
    g_allocator.DeviceAllocate((void**)&d_predicate, sizeof(int) * image_size);
    g_allocator.DeviceAllocate((void**)&d_scan_result, sizeof(int) * image_size);
    //cudaMalloc(&d_predicate, image_size * sizeof(int));
    //cudaMalloc(&d_scan_result, image_size * sizeof(int));
    cudaMalloc(&d_output, image_size * sizeof(int));

    std::vector<int> h_predicate;
    h_predicate.resize(image_size);

    int blockSize = 256;
    int gridSize = (image_size + blockSize - 1) / blockSize;

    predicate_kernel<<<gridSize, blockSize>>>(d_buffer, d_predicate, image_size, garbage_val);

    //apply a scan on d_buffer with cub
    // Create a CUB device scan context
    void *d_temp_storage = nullptr;
    size_t temp_storage_bytes = image_size;
    g_allocator.DeviceAllocate((void**)&d_temp_storage, sizeof(int) * image_size);
    cub::DeviceScan::ExclusiveSum(
            d_temp_storage, // Temporary storage (nullptr to query required storage)
            image_size,
            d_predicate, // Output (scan result)
            d_scan_result, // Input (the predicate)
            image_size
    );

    scatter_kernel<<<gridSize, blockSize>>>(d_buffer, d_scan_result, d_output, image_size, garbage_val);

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

Image fix_image_gpu_perfect(Image image){
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
