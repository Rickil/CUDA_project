#ifndef IRGPUA_PROJECT_GPU_SCAN_CUH
#define IRGPUA_PROJECT_GPU_SCAN_CUH

#include <iostream>

void exclusive_scan(int *d_buffer, int size);

#endif //IRGPUA_PROJECT_GPU_SCAN_CUH
