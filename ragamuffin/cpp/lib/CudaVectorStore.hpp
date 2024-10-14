#pragma once

#include "CudaVector.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

namespace ragamuffin {

template <typename T>
struct CudaVectorStore {
    virtual ~CudaVectorStore() = default;

    // Return the ID of the newly added embedding
    virtual int Add(py::array_t<float> embedding) {
        auto vec = CudaVector<T>(embedding.request().shape[0], pool);
        return this->Add(vec);
    }
    virtual int Add(CudaVector<T> embedding) = 0;

    /**
     * Freeze the store, making it read-only. This conveys a number of optimizations to the store, such
     * as contiguous memory allocation. Defaults to a no-op.
     */
    virtual void Freeze() {}

    virtual CudaVector<T> &Get(int id) = 0;
};

template <typename T>
struct FlatCudaVectorStore : CudaVectorStore<T> {
    FlatCudaVectorStore(int dim);
    ~FlatCudaVectorStore();

    int Add(py::array_t<float> embedding) final;
    int Add(CudaVector<T> embedding) final;

    CudaVector<T> &Get(int id);

private:
    
};

}