#pragma once

#include <iostream>
#include <stdexcept>

#include <cuda_runtime.h>

namespace ragamuffin {

struct CudaStream {
    CudaStream(bool blocking = false, int priority = 0) {
        auto ret = cudaStreamCreateWithPriority(&this->stream_, blocking ? cudaStreamDefault : cudaStreamNonBlocking, priority);

        if (ret != cudaSuccess)
            throw std::runtime_error("Failed to create CUDA stream");
    }

    ~CudaStream() {
        auto ret = cudaStreamDestroy(this->stream_);

        if (ret != cudaSuccess)
            throw std::runtime_error("Failed to destroy CUDA stream");
    }

    inline cudaStream_t GetHandle() const noexcept { return this->stream_; }

private:
    cudaStream_t stream_;
};

} // namespace ragamuffin