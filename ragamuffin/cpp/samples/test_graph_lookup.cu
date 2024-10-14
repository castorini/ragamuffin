#include "../lib/CudaMemoryPool.hpp"
#include "../lib/CudaDevice.hpp"
#include "../lib/CudaVector.hpp"

#include <chrono>
#include <iostream>
#include <thread>


int main() {
    ragamuffin::CudaMemoryPool pool(1024 * 1000 * 1000, ragamuffin::CudaDevice(0));
    std::cout << "Allocated memory pool" << std::endl;
    auto stream = std::make_shared<ragamuffin::CudaStream>();
    std::vector<ragamuffin::CudaMemoryPoolBlock> blocks;
    
    for (int i = 0; i < 10; ++i) {
        auto start = std::chrono::steady_clock::now();
        auto block = pool.Allocate(1000000, stream);
        std::cout << "Speed: " << 1000000.0 / (std::chrono::steady_clock::now() - start).count() << " MB/s" << std::endl;
    }

    std::vector<ragamuffin::CudaVector<int>> tensors;
    tensors.emplace_back(1000, pool);
    tensors.emplace_back(1000, pool);

    auto [vecs, block] = ragamuffin::Contigify(tensors.begin(), tensors.end(), pool, stream);

    for (const auto &vec : vecs) {
        std::cout << vec->GetRawPtr() << std::endl;
    }
}
