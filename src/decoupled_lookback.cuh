#ifndef IRGPUA_PROJECT_DECOUPLED_LOOKBACK_CUH
#define IRGPUA_PROJECT_DECOUPLED_LOOKBACK_CUH

#define cudaCheckError()                                                       \
    {                                                                          \
        cudaError_t e = cudaGetLastError();                                    \
        if (e != cudaSuccess)                                                  \
        {                                                                      \
            printf("Cuda fail: %s:%d: '%s'\n", __FILE__, __LINE__,             \
                   cudaGetErrorString(e));                                     \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    }

#include <iostream>

void decoupled_lookback(int* buffer, int size);

#endif //IRGPUA_PROJECT_DECOUPLED_LOOKBACK_CUH
