#pragma once

#include "CudaMemoryPool.hpp"
#include "Utils.hpp"

#include <algorithm>
#include <cstddef>
#include <iterator>
#include <memory>
#include <numeric>
#include <optional>

#include <cuda_runtime.h>

namespace ragamuffin {

template <typename T>
struct CudaVector {
    using cuda_data_type = T;

    // Allocate from a memory pool
    CudaVector(
        std::size_t size,
        CudaMemoryPool &pool,
        bool autoresize = true,
        std::shared_ptr<CudaStream> stream = std::make_shared<CudaStream>(true)
    ) : size_(size) {
        if (autoresize && pool.GetSize() + size * sizeof(T) > pool.GetMaxSize())
            pool.Resize(pool.GetMaxSize() + size * sizeof(T), stream);

        this->block_ = pool.Allocate(size * sizeof(T), stream);
    }

    // Copy constructor (deleted)
    CudaVector(const CudaVector &other) = delete;
    CudaVector &operator=(const CudaVector &other) = delete;
    
    // Move constructor
    CudaVector(CudaVector &&other) noexcept : size_(other.size_), block_(std::move(other.block_)) {
        other.block_ = nullptr;
        other.size_ = 0;
    }

    // Destructor
    ~CudaVector() {
        this->block_ = nullptr;
    }

    void CopyFromHost(T *host_data, std::size_t size) {
        if (this->block_) {
            const auto &stream = this->block_->GetStream();
            auto ret = cudaMemcpyAsync(this->GetRawPtr(), host_data, size * sizeof(T), cudaMemcpyHostToDevice, stream.GetHandle());
            CheckCudaError(ret, "Failed to copy memory");
        } else {
            auto ret = cudaMemcpy(this->GetRawPtr(), host_data, size * sizeof(T), cudaMemcpyHostToDevice);
            CheckCudaError(ret, "Failed to copy memory");
        }
    }

    void CopyFromDevice(T *device_data, std::size_t size) {
        if (this->block_) {
            const auto &stream = this->block_->GetStream();
            auto ret = cudaMemcpyAsync(this->GetRawPtr(), device_data, size * sizeof(T), cudaMemcpyDeviceToDevice, stream.GetHandle());
            CheckCudaError(ret, "Failed to copy memory");
        } else {
            auto ret = cudaMemcpy(this->GetRawPtr(), device_data, size * sizeof(T), cudaMemcpyDeviceToDevice);
            CheckCudaError(ret, "Failed to copy memory");
        }
    }

    T *GetRawPtr() const noexcept { return static_cast<T*>(this->block_->GetHandle()); }
    std::size_t GetSize() const noexcept { return this->size_; }

private:
    std::size_t size_;
    std::shared_ptr<CudaMemoryPoolBlock> block_;
};

template <typename FirstIt, typename LastIt>
inline std::pair<
    std::vector<
        std::unique_ptr<CudaVector<typename std::iterator_traits<FirstIt>::value_type::cuda_data_type>>
    >,
    std::shared_ptr<CudaMemoryPoolBlock>
> Contigify(
    FirstIt first,
    LastIt last,
    CudaMemoryPool &pool,
    std::shared_ptr<CudaStream> stream
) {
    using T = typename std::iterator_traits<FirstIt>::value_type::cuda_data_type;
    int total_size = std::accumulate(first, last, 0, [](int sum, const auto &vec) {
        return sum + vec.GetSize();
    });

    auto block = pool.Allocate(total_size * sizeof(T), stream);
    T *curr_ptr = static_cast<T*>(block->GetHandle());
    std::vector<std::unique_ptr<CudaVector<T>>> result;

    for (auto it = first; it != last; ++it) {
        const auto &vec = *it;
        auto ret = cudaMemcpyAsync(curr_ptr, vec.GetRawPtr(), vec.GetSize() * sizeof(T), cudaMemcpyDeviceToDevice, stream->GetHandle());
        CheckCudaError(ret, "Failed to copy memory");

        result.emplace_back(std::make_unique<CudaVector<T>>(vec.GetSize(), pool, true, stream));
        curr_ptr += vec.GetSize();
    }

    return std::make_pair(std::move(result), std::move(block));
}

} // namespace ragamuffin