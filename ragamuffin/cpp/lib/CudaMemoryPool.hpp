#include "CudaDevice.hpp"
#include "CudaStream.hpp"

#include <cuda_runtime.h>

#include <vector>
#include <mutex>
#include <iostream>

namespace ragamuffin {

struct CudaMemoryPoolBlock;

struct CudaMemoryPool {
    CudaMemoryPool(std::size_t max_size, const CudaDevice &device);
    ~CudaMemoryPool();

    CudaMemoryPoolBlock Allocate(std::size_t size, const CudaStream &stream) const;
    void Free(void *handle, const CudaStream &stream) const;

    inline std::size_t GetMaxSize() const noexcept { return this->max_size_; }

private:
    std::size_t max_size_;
    cudaMemPool_t pool_;
};

struct CudaMemoryPoolBlock {
    friend struct CudaMemoryPool;

    // Destructor
    ~CudaMemoryPoolBlock() {
        if (this->ptr_)
            this->pool_.Free(this->ptr_, this->stream_);
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
    inline const CudaStream &GetStream() const noexcept { return this->stream_; }

private:
    CudaMemoryPoolBlock(const CudaMemoryPool &pool, void *ptr, std::size_t size, const CudaStream &stream)
        : pool_(pool), ptr_(ptr), size_(size), stream_(stream) {}

    const CudaMemoryPool &pool_;
    void *ptr_;
    std::size_t size_;
    const CudaStream &stream_;
};

} // namespace ragamuffin