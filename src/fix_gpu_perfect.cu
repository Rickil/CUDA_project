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

    // Allocate memory for predicate, scan result, and output
    g_allocator.DeviceAllocate((void**)&d_predicate, sizeof(int) * image_size);
    g_allocator.DeviceAllocate((void**)&d_scan_result, sizeof(int) * image_size);
    g_allocator.DeviceAllocate((void**)&d_output, sizeof(int) * image_size);

    int blockSize = 256;
    int gridSize = (image_size + blockSize - 1) / blockSize;

    // Step 1: Compute the predicate using a kernel
    predicate_kernel<<<gridSize, blockSize>>>(d_buffer, d_predicate, image_size, garbage_val);

    // Step 2: Perform exclusive scan on the predicate with CUB
    void *d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;  // Start with 0 bytes to query required storage
    cub::DeviceScan::ExclusiveSum(
            d_temp_storage,
            temp_storage_bytes,
            d_predicate,
            d_scan_result,
            image_size
    );

    // Allocate temporary storage for the scan
    g_allocator.DeviceAllocate((void**)&d_temp_storage, temp_storage_bytes);

    // Perform the actual scan operation
    cub::DeviceScan::ExclusiveSum(
            d_temp_storage,
            temp_storage_bytes,
            d_predicate,
            d_scan_result,
            image_size
    );

    // Step 3: Perform scatter operation to compact the input array
    scatter_kernel<<<gridSize, blockSize>>>(d_buffer, d_scan_result, d_output, image_size, garbage_val);

    // Copy the compacted data back to the original buffer
    cudaMemcpy(d_buffer, d_output, image_size * sizeof(int), cudaMemcpyDeviceToDevice);

    // Free allocated memory
    g_allocator.DeviceFree(d_predicate);
    g_allocator.DeviceFree(d_scan_result);
    g_allocator.DeviceFree(d_output);
    g_allocator.DeviceFree(d_temp_storage);
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

void calculate_histogram_gpu(int* d_input, int* d_histogram, int num_elements) {
    // Define the input and histogram types
    typedef int InputType;
    typedef int HistogramType;

    // Define CUB temporary storage and workspace variables
    void* d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;

    // Create a histogram configuration
    cub::Histogram::HistogramEvenConfig config(
            0, 255, histogram_size  // Min, Max, Number of bins
    );

    // Allocate temporary storage (query required size)
    cub::DeviceHistogram::Even(
            d_temp_storage,
            temp_storage_bytes,
            d_input,
            d_histogram,
            num_elements,
            config
    );

    // Allocate temporary storage based on the query size
    cudaMalloc(&d_temp_storage, temp_storage_bytes);

    // Calculate the histogram
    cub::DeviceHistogram::Even(
            d_temp_storage,
            temp_storage_bytes,
            d_input,
            d_histogram,
            num_elements,
            config
    );

    // Free temporary storage
    cudaFree(d_temp_storage);
}

void fix_image_gpu_perfect(Image& image){
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
    calculate_histogram_gpu(d_buffer, d_histogram, image.size());
    findFirstNonZero(d_histogram, cdf_min, image.size());
    equalize_histogram_gpu(d_buffer, d_histogram, image.size(), cdf_min);

    cudaMemcpy(image.buffer, d_buffer, image.size() * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_buffer);
    cudaFree(d_histogram);
    cudaFree(cdf_min);

    //std::cout << "hey\n";

    //return image;
}
