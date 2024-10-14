#pragma once

#include "CudaMemoryPool.hpp"
#include "CudaVector.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

namespace ragamuffin {

template <typename T>
struct CudaVectorStore {
    using vector_type = T;

    CudaVectorStore(int dim, int init_size = 1000, CudaDevice device = CudaDevice(0)) : dim_(dim), device_(device) {}
    virtual ~CudaVectorStore() = default;

    // Return the ID of the newly added embedding
    virtual int Add(py::array_t<T> embedding) = 0;
    virtual int Add(CudaVector<T> embedding) = 0;

    /**
     * Freeze the store, making it read-only. This conveys a number of optimizations to the store, such
     * as contiguous memory allocation. Defaults to a no-op.
     */
    virtual void Freeze() {}

    virtual CudaVector<T> &Get(int id) = 0;

    inline int GetDim() const noexcept { return this->dim_; }
    inline int GetVectorSize() const noexcept { return this->dim_ * sizeof(T); }

private:
    int dim_;
    CudaDevice device_;
};

template <typename T>
struct FlatCudaVectorStore : CudaVectorStore<T> {
    FlatCudaVectorStore(int dim, int init_size = 1000, CudaDevice device = CudaDevice(0)) : CudaVectorStore<T>(dim, init_size, device) {
        this->pool_ = std::make_unique<CudaMemoryPool>(dim * sizeof(T) * init_size, device);
    }

    virtual ~FlatCudaVectorStore() = default;

    virtual int Add(py::array_t<T> embedding) final {
        while (this->pool_->GetSize() + embedding.request().shape[0] * sizeof(T) > this->pool_->GetMaxSize()) {
            this->pool_->Resize(this->pool_->GetMaxSize() * 2);
        }

        CudaVector<T> vec(embedding.request().shape[0], *this->pool_);
        vec.CopyFromHost(embedding.mutable_data(), embedding.request().shape[0]);

        return this->Add(std::move(vec));
    }

    virtual int Add(CudaVector<T> embedding) final {
        // embedding is assumed to be from the same pool
        this->vectors_.push_back(std::move(embedding));
        return this->current_id_++;
    }

    virtual CudaVector<T> &Get(int id) final {
        return this->vectors_[id];
    }

private:
    std::unique_ptr<CudaMemoryPool> pool_;
    std::vector<CudaVector<T>> vectors_;
    int current_id_;
};

}