#include "../lib/CudaMemoryPool.hpp"
#include "../lib/CudaDevice.hpp"

#include <chrono>
#include <iostream>
#include <thread>


int main() {
    ragamuffin::CudaMemoryPool pool(1024 * 1000 * 1000, ragamuffin::CudaDevice(0));
    std::cout << "Allocated memory pool" << std::endl;
    ragamuffin::CudaStream stream;
    std::vector<ragamuffin::CudaMemoryPoolBlock> blocks;
    
    while (true) {
        auto start = std::chrono::steady_clock::now();
        auto block = pool.Allocate(1000000, stream);
        std::cout << "Speed: " << 1000000.0 / (std::chrono::steady_clock::now() - start).count() << " MB/s" << std::endl;
    }
}