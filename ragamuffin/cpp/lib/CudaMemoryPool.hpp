#pragma once

#include "CudaDevice.hpp"
#include "CudaStream.hpp"

#include <cuda_runtime.h>

#include <iostream>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace ragamuffin {

struct CudaMemoryPoolBlock;

struct CudaMemoryPool {
    CudaMemoryPool(std::size_t max_size, const CudaDevice &device);
    ~CudaMemoryPool();

    // Allocate/deallocate memory from the pool
    std::shared_ptr<CudaMemoryPoolBlock> Allocate(std::size_t size);  // default stream
    std::shared_ptr<CudaMemoryPoolBlock> Allocate(std::size_t size, std::shared_ptr<CudaStream> stream);
    void Free(void *handle, const CudaStream &stream, bool from_destructor = false);

    // Resize the pool; this is very slow and should be used sparingly. It copies all the memory from the old pool to the new pool,
    // updating the pointers in the process. This is not thread-safe.
    void Resize(std::size_t new_size, std::shared_ptr<CudaStream> stream);

    inline std::size_t GetMaxSize() const noexcept { return this->max_size_; }

private:
    std::size_t max_size_;
    cudaMemPool_t pool_;
    std::unordered_map<void *, std::shared_ptr<CudaMemoryPoolBlock>> blocks_;
    CudaDevice device_;
};

struct CudaMemoryPoolBlock {
    CudaMemoryPoolBlock(CudaMemoryPool &pool, void *ptr, std::size_t size, std::shared_ptr<CudaStream> stream)
        : pool_(pool), ptr_(ptr), size_(size), stream_(stream) {}

    // Destructor
    ~CudaMemoryPoolBlock() {
        if (this->ptr_) {
            this->pool_.Free(this->ptr_, *this->stream_, true);
        }
    }

    // Move constructor
    CudaMemoryPoolBlock(CudaMemoryPoolBlock &&other) noexcept : pool_(other.pool_), stream_(other.stream_) {
        this->ptr_ = other.ptr_;
        this->size_ = other.size_;
        other.ptr_ = nullptr;
        other.size_ = 0;
    }

    // Delete copy constructor and assignment operator
    CudaMemoryPoolBlock(const CudaMemoryPoolBlock &) = delete;
    CudaMemoryPoolBlock &operator=(const CudaMemoryPoolBlock &) = delete;

    inline void *GetHandle() const noexcept { return this->ptr_; }
    inline std::size_t GetSize() const noexcept { return this->size_; }
    inline const CudaStream &GetStream() const noexcept { return *this->stream_; }

    inline void Free() {
        this->pool_.Free(this->ptr_, *this->stream_);
        this->ptr_ = nullptr;
    }

private:
    CudaMemoryPool &pool_;
    void *ptr_;
    std::size_t size_;
    std::shared_ptr<CudaStream> stream_;
    bool freed_ = false;
};

} // namespace ragamuffin