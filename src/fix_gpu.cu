#include "fix_gpu.cuh"
#include "gpu_scan.cuh"
#include "decoupled_lookback.cuh"

__global__ void predicate_kernel(int* buffer, int* out, int size) {
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id >= size)
        return;

    constexpr int garbage = -27;
    int inter = buffer[id];
    __syncthreads();

    out[id] = inter != garbage ? 1 : 0;
    //printf("%d\n", out[id]);
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
        /*int block = 1000;
        int blockSize = 1024;
        if ( i >= (block*blockSize)-3 && i<=(block*blockSize)+3) {
            if (i == block * blockSize)
                printf("### new block (%d) ###\n", block);
            printf("index: %d, cpu: %d, gpu: %d\n", i, h_predicate[i], h_scan_result[i]);
        }*/
        if (h_predicate[i] != h_scan_result[i]){
            same = false;
            count++;
            if (count<10)
                printf("index: %d, cpu: %d, gpu: %d\n", i, h_predicate[i], h_scan_result[i]);
        }
    }

    if (same)
        printf("scan good !\n");
    else
        printf("scan bad !, %d are bad, %i\n", count, (count/size)*100);

}

void check_scatter(int *my_d_buffer, int *d_buffer, int *d_predicate, int size){
    std::vector<int> h_buffer(size, 0);
    cudaMemcpy(h_buffer.data(), d_buffer, size*sizeof(int), cudaMemcpyDeviceToHost);
    std::vector<int> my_h_buffer(size, 0);
    cudaMemcpy(my_h_buffer.data(), my_d_buffer, size*sizeof(int), cudaMemcpyDeviceToHost);
    std::vector<int> h_predicate(size, 0);
    cudaMemcpy(h_predicate.data(), d_predicate, size*sizeof(int), cudaMemcpyDeviceToHost);

    constexpr int garbage_val = -27;
    for (int i = 0; i < size; ++i) {
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

__global__ void scatter_kernel(int *buffer, int *exclusive_scan,
                               int *out_buffer, int size) {

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
void compact_image_gpu(int* d_buffer, int* d_output, int image_size) {
    int *d_predicate;
    cudaMalloc(&d_predicate, image_size*sizeof(int));
    cudaMemset(d_predicate, 0, image_size*sizeof(int));


    int blockSize = 256;
    int gridSize = (image_size + blockSize - 1) / blockSize;

    //copy the predicate for further tests
    /*int *d_buffer_copy;
    cudaMalloc(&d_buffer_copy, image_size*sizeof(int));
    cudaMemcpy(d_buffer_copy, d_buffer, image_size*sizeof(int), cudaMemcpyDeviceToDevice);*/

    predicate_kernel<<<gridSize, blockSize>>>(d_buffer, d_predicate, image_size);
    //check_predicate(d_buffer_copy, d_predicate, image_size);

    /*int *d_predicate_copy;
    cudaMalloc(&d_predicate_copy, image_size*sizeof(int));
    cudaMemcpy(d_predicate_copy, d_predicate, image_size*sizeof(int), cudaMemcpyDeviceToDevice);*/

    //exclusive_scan(d_predicate, image_size);
    decoupled_lookback(d_predicate, image_size);
    //check_scan(d_predicate_copy, d_predicate, image_size);

    cudaMemcpy(d_output, d_buffer, image_size*sizeof(int), cudaMemcpyDeviceToDevice);
    scatter_kernel<<<gridSize, blockSize>>>(d_buffer, d_predicate, d_output, image_size);
    //check_scatter(d_output, d_buffer_copy, d_predicate, image_size);

    cudaFree(d_predicate);
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
    if (i >= size) {
        return;
    }

    int index = buffer[i];
    __syncthreads();

    atomicAdd(&histogram[index], 1);

    __syncthreads();
}

void check_histogram(int* d_histogram, int* d_buffer, int histogram_size, int image_size){
    std::vector<int> h_histogram(histogram_size, 0);
    cudaMemcpy(h_histogram.data(), d_histogram, histogram_size*sizeof(int), cudaMemcpyDeviceToHost);
    std::vector<int> h_buffer(image_size, 0);
    cudaMemcpy(h_buffer.data(), d_buffer, image_size*sizeof(int), cudaMemcpyDeviceToHost);

    std::array<int, 256> histo;
    histo.fill(0);
    for (int i = 0; i < image_size; ++i)
        ++histo[h_buffer[i]];

    bool same = true;
    int count = 0;
    for (int i = 0; i < histogram_size; i++){
        if (histo[i] != h_histogram[i]){
            same = false;
            count++;
            //printf("index: %d, cpu: %d, gpu: %d\n", i, h_buffer[i], my_h_buffer[i]);
        }
    }

    if (same)
        printf("calculate histogram good !\n");
    else
        printf("calculate histogram !, %d are bad, %f\n", count, ((float)count/histogram_size)*100);
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

__global__ void inclusive_scan_kernel(int* buffer, int size) {
    unsigned int id = threadIdx.x;
    extern __shared__ int s[];
    s[id] = buffer[id];
    __syncthreads();

    if (id < size) {
        for (int i = 1; i < size; i *= 2) {
            int x;
            if (i <= id) {
                x = s[id - i];
            }
            __syncthreads();
            if(i<=id){
                s[id] += x;
            }
            __syncthreads();
        }
    }
    buffer[id] = s[id];
}

void inclusive_scan(int* d_buffer, int size){
    inclusive_scan_kernel<<<1, size, size*sizeof(int)>>>(d_buffer, size);
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

    compact_image_gpu(d_buffer, d_output, image_size);
    apply_map_to_pixels_gpu(d_output, compact_size);

    calculate_histogram_gpu(d_output, d_histogram, compact_size);
    //check_histogram(d_histogram, d_output, histogram_size, compact_size);

    inclusive_scan(d_histogram, histogram_size);

    findFirstNonZero(d_histogram, cdf_min, histogram_size);

    equalize_histogram_gpu(d_output, d_histogram, compact_size, cdf_min);

    cudaMemcpy(image.buffer, d_output, image.width*image.height * sizeof(int), cudaMemcpyDeviceToHost);

    //check image_buffer
    /*int count = 0;
    for (int i=0; i<compact_size; i++){
        if (image.buffer[i] == -27){
            count++;
            //printf("value -27 at %d\n", i);
        }
    }
    printf("%d values at -27\n", count);*/

    cudaFree(d_buffer);
    cudaFree(d_output);
    cudaFree(d_histogram);
    cudaFree(cdf_min);

    //std::cout << "hey\n";

    //return image;
}