#pragma once

#include <cuda_runtime.h>

namespace ragamuffin {

struct CudaDevice {
    CudaDevice(int id) {
        this->location_.type = cudaMemLocationTypeDevice;
        this->location_.id = id;
    }

    inline const cudaMemLocation &GetLocation() const noexcept { return this->location_; }

private:
    cudaMemLocation location_;
};

} // namespace ragamuffin