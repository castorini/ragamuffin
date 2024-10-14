#include "../lib/CudaMemoryPool.hpp"
#include "../lib/CudaDevice.hpp"
#include "../lib/CudaVector.hpp"
#include "../lib/CudaVectorStore.hpp"

#include <chrono>
#include <iostream>
#include <thread>


int main() {
    ragamuffin::CudaMemoryPool pool(1024 * 1000 * 1000, ragamuffin::CudaDevice(0));
    std::cout << "Allocated memory pool" << std::endl;
    auto stream = std::make_shared<ragamuffin::CudaStream>();
    std::vector<std::shared_ptr<ragamuffin::CudaMemoryPoolBlock>> blocks;
    
    for (int i = 0; i < 50000; ++i) {
        blocks.push_back(pool.Allocate(100, stream));
    }

    std::cerr << "First address: " << blocks[0]->GetHandle() << std::endl;

    auto start = std::chrono::steady_clock::now();
    pool.Resize(1024 * 1000 * 100, stream);
    auto duration = std::chrono::steady_clock::now() - start;
    std::cout << "Resize time: " << std::chrono::duration_cast<std::chrono::milliseconds>(duration).count() << " ms" << std::endl;

    std::cerr << "First address after resize: " << blocks[0]->GetHandle() << std::endl;

    std::vector<ragamuffin::CudaVector<int>> tensors;
    tensors.emplace_back(1000, pool);
    tensors.emplace_back(1000, pool);

    auto [vecs, block] = ragamuffin::Contigify(tensors.begin(), tensors.end(), pool, stream);

    for (const auto &vec : vecs) {
        std::cout << vec->GetRawPtr() << std::endl;
    }

    ragamuffin::FlatCudaVectorStore<int> store(1000, 1000, ragamuffin::CudaDevice(0));
    store.Add(ragamuffin::CudaVector<int>(1000, pool));
}