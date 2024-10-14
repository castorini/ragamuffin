#include "CudaMemoryPool.hpp"
#include "CudaDevice.hpp"

#include <stdexcept>

#include <cuda_runtime.h>

using namespace ragamuffin;

CudaMemoryPool::CudaMemoryPool(std::size_t max_size, const CudaDevice &device) : max_size_(max_size) {
    cudaMemPoolProps props = {};
    props.allocType = cudaMemAllocationTypePinned;
    props.handleTypes = cudaMemHandleTypeNone;
    props.location = device.GetLocation();
    props.maxSize = max_size;
    auto ret = cudaMemPoolCreate(&this->pool_, &props);

    if (ret != cudaSuccess)
        throw std::runtime_error("Failed to create CUDA memory pool");
}

CudaMemoryPool::~CudaMemoryPool() {
    auto ret = cudaMemPoolDestroy(this->pool_);

    if (ret != cudaSuccess)
        throw std::runtime_error("Failed to destroy CUDA memory pool");
}

std::unique_ptr<CudaMemoryPoolBlock> CudaMemoryPool::Allocate(std::size_t size, std::shared_ptr<CudaStream> stream) const {
    void *ptr;
    auto ret = cudaMallocFromPoolAsync(&ptr, size, this->pool_, stream->GetHandle());

    if (ret != cudaSuccess)
        throw std::runtime_error("Failed to allocate memory from pool");

    return std::make_unique<CudaMemoryPoolBlock>(*this, ptr, size, stream);
}

std::unique_ptr<CudaMemoryPoolBlock> CudaMemoryPool::Allocate(std::size_t size) const {
    return this->Allocate(size, std::make_shared<CudaStream>(true));
}

void CudaMemoryPool::Free(void *handle, const CudaStream &stream) const {
    auto ret = cudaFreeAsync(handle, stream.GetHandle());

    if (ret != cudaSuccess)
        throw std::runtime_error("Failed to free memory");
}
