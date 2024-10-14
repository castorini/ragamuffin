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
    CudaVector(std::size_t size, CudaMemoryPool &pool) : data_(nullptr), size_(size) {
        this->block_ = pool.Allocate(size * sizeof(T));
        this->data_ = static_cast<T*>((*this->block_)->GetHandle());
    }

    CudaVector(std::size_t size, CudaMemoryPool &pool, std::shared_ptr<CudaStream> stream) : data_(nullptr), size_(size) {
        this->block_ = pool.Allocate(size * sizeof(T), stream);
        this->data_ = static_cast<T*>((*this->block_)->GetHandle());
    }

    // From an existing pointer
    CudaVector(T *data, std::size_t size) : data_(data), size_(size) {}

    // Copy constructor (deleted)
    CudaVector(const CudaVector &other) = delete;
    CudaVector &operator=(const CudaVector &other) = delete;
    
    // Move constructor
    CudaVector(CudaVector &&other) noexcept : data_(other.data_), size_(other.size_), block_(std::move(other.block_)) {
        other.block_ = nullptr;
        other.data_ = nullptr;
        other.size_ = 0;
    }

    // Destructor
    ~CudaVector() {
        this->block_ = std::nullopt;
    }

    void CopyFromHost(T *host_data, std::size_t size) {
        if (this->block_) {
            const auto &stream = (*this->block_)->GetStream();
            auto ret = cudaMemcpyAsync(this->data_, host_data, size * sizeof(T), cudaMemcpyHostToDevice, stream.GetHandle());

            if (ret != cudaSuccess)
                throw std::runtime_error("Failed to copy memory");
        } else {
            auto ret = cudaMemcpy(this->data_, host_data, size * sizeof(T), cudaMemcpyHostToDevice);

            if (ret != cudaSuccess)
                throw std::runtime_error("Failed to copy memory");
        }
    }

    T *GetRawPtr() const noexcept { return this->data_; }
    std::size_t GetSize() const noexcept { return this->size_; }

private:
    T *data_;
    std::size_t size_;
    std::optional<std::shared_ptr<CudaMemoryPoolBlock>> block_;
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

        result.emplace_back(std::make_unique<CudaVector<T>>(curr_ptr, vec.GetSize()));
        curr_ptr += vec.GetSize();
    }

    return std::make_pair(std::move(result), std::move(block));
}

} // namespace ragamuffin