#ifndef PROJECT_STUDENT_FIX_GPU_CUH
#define PROJECT_STUDENT_FIX_GPU_CUH

#include "image.hh"

__global__ void fix_image_gpu(int* buffer, int image_size);

#endif //PROJECT_STUDENT_FIX_GPU_CUH
