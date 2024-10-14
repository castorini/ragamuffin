#pragma once

#include <iostream>
#include <stdexcept>

#include <cuda_runtime.h>

#include "Utils.hpp"

namespace ragamuffin {

struct CudaStream {
    CudaStream(bool blocking = false, int priority = 0) {
        auto ret = cudaStreamCreateWithPriority(&this->stream_, blocking ? cudaStreamDefault : cudaStreamNonBlocking, priority);

        if (ret != cudaSuccess)
            throw std::runtime_error("Failed to create CUDA stream");
    }

    CudaStream(CudaStream &&other) noexcept : stream_(other.stream_) {
        other.stream_ = nullptr;
    }

    CudaStream(CudaStream &other) = delete;

    ~CudaStream() {
        if (this->stream_) {
            auto ret = cudaStreamDestroy(this->stream_);
            CheckCudaError(ret, "Failed to destroy CUDA stream");
        }
    }

    inline cudaStream_t GetHandle() const noexcept { return this->stream_; }

private:
    cudaStream_t stream_;
};

} // namespace ragamuffin