#include "CudaMemoryPool.hpp"
#include "CudaDevice.hpp"
#include "Utils.hpp"

#include <stdexcept>

#include <cuda_runtime.h>

using namespace ragamuffin;

CudaMemoryPool::CudaMemoryPool(std::size_t max_size, const CudaDevice &device) : max_size_(max_size), device_(device) {
    cudaMemPoolProps props = {};
    props.allocType = cudaMemAllocationTypePinned;
    props.handleTypes = cudaMemHandleTypeNone;
    props.location = device.GetLocation();
    props.maxSize = max_size;
    auto ret = cudaMemPoolCreate(&this->pool_, &props);
    CheckCudaError(ret, "Failed to create CUDA memory pool");
}

CudaMemoryPool::~CudaMemoryPool() {
    auto ret = cudaMemPoolDestroy(this->pool_);
    CheckCudaError(ret, "Failed to destroy CUDA memory pool");
}

std::shared_ptr<CudaMemoryPoolBlock> CudaMemoryPool::Allocate(std::size_t size, std::shared_ptr<CudaStream> stream) {
    void *ptr;
    auto ret = cudaMallocFromPoolAsync(&ptr, size, this->pool_, stream->GetHandle());
    CheckCudaError(ret, "Failed to allocate memory from pool");
    
    auto block = std::make_shared<CudaMemoryPoolBlock>(std::ref(*this), ptr, size, stream);
    this->blocks_[ptr] = block;

    return block;
}

void CudaMemoryPool::Free(void *handle, const CudaStream &stream, bool from_destructor) {
    auto ret = cudaFreeAsync(handle, stream.GetHandle());
    CheckCudaError(ret, "Failed to free memory");

    if (!from_destructor) {
        this->blocks_.erase(handle);
    }
}

void CudaMemoryPool::Resize(std::size_t new_size, std::shared_ptr<CudaStream> stream) {
    auto old_pool = this->pool_;

    cudaMemPoolProps props = {};
    props.allocType = cudaMemAllocationTypePinned;
    props.handleTypes = cudaMemHandleTypeNone;
    props.location = this->device_.GetLocation();
    props.maxSize = new_size;
    auto ret = cudaMemPoolCreate(&this->pool_, &props);
    CheckCudaError(ret, "Failed to create CUDA memory pool");

    for (auto &block : this->blocks_) {
        auto new_block = this->Allocate(block.second->GetSize(), stream);
        auto ret = cudaMemcpyAsync(
            new_block->GetHandle(),
            block.second->GetHandle(),
            block.second->GetSize(),
            cudaMemcpyDeviceToDevice,
            stream->GetHandle()
        );

        CheckCudaError(ret, "Failed to resize pool");
        block.second->Replace(std::move(*new_block));
    }

    ret = cudaMemPoolDestroy(old_pool);
    CheckCudaError(ret, "Failed to destroy CUDA memory pool");
}

std::size_t CudaMemoryPool::GetSize() const noexcept {
    return std::accumulate(this->blocks_.begin(), this->blocks_.end(), std::size_t(0), [](std::size_t sum, const auto &block) {
        return sum + block.second->GetSize();
    });
}