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
    virtual int Add(py::array_t<float> embedding) = 0;
    virtual int Add(CudaVector<T> embedding) = 0;

    virtual CudaVector<T> &Get(int id) = 0;
};

template <typename T>
struct FlatCudaVectorStore : CudaVectorStore<T> {
    FlatCudaVectorStore(int dim);
    ~FlatCudaVectorStore();

    int Add(py::array_t<float> embedding);
    int Add(CudaVector<T> embedding);

    CudaVector<T> &Get(int id);
};

}