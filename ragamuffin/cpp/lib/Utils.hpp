#pragma once

#include <iostream>

#include <cuda_runtime.h>

namespace ragamuffin {

inline void CheckCudaError(cudaError_t ret, const std::string &msg) {
    if (ret != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorName(ret) << "; " << cudaGetErrorString(ret) << std::endl;
        throw std::runtime_error(msg);
    }
}

} // namespace ragamuffin