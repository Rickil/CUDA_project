#include "fix_gpu.cuh"

__global__ void fix_image_gpu(int* buffer, int image_size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < image_size) {
        if (idx % 4 == 0)
            buffer[idx] += 1;
        else if (idx % 4 == 1)
            buffer[idx] -= 5;
        else if (idx % 4 == 2)
            buffer[idx] += 3;
        else if (idx % 4 == 3)
            buffer[idx] -= 8;
    }
}