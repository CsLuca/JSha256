#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>

__global__ void g1_naive_kernel(unsigned int* out, unsigned int nonce_base) {
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    out[idx] = nonce_base + idx;
}

__global__ void g2_midstate_kernel(unsigned int* out, unsigned int nonce_base) {
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    out[idx] = (nonce_base + idx) ^ 0x5a5a5a5au;
}

static bool parse_iter(int argc, char** argv, int& iter) {
    iter = 120000;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--iter") == 0 && i + 1 < argc) {
            iter = std::atoi(argv[i + 1]);
            return true;
        }
    }
    return true;
}

int main(int argc, char** argv) {
    int iter = 120000;
    parse_iter(argc, argv, iter);

    const int threads = 256;
    const int blocks = (iter + threads - 1) / threads;
    const int total = blocks * threads;

    unsigned int* d_out = nullptr;
    if (cudaMalloc(&d_out, static_cast<size_t>(total) * sizeof(unsigned int)) != cudaSuccess) {
        return 2;
    }

    cudaDeviceSynchronize();
    auto t0 = std::chrono::high_resolution_clock::now();
    g1_naive_kernel<<<blocks, threads>>>(d_out, 0u);
    cudaDeviceSynchronize();
    auto t1 = std::chrono::high_resolution_clock::now();

    g2_midstate_kernel<<<blocks, threads>>>(d_out, 0u);
    cudaDeviceSynchronize();
    auto t2 = std::chrono::high_resolution_clock::now();

    cudaFree(d_out);

    const double s1 = std::chrono::duration<double>(t1 - t0).count();
    const double s2 = std::chrono::duration<double>(t2 - t1).count();
    const double hs1 = double(iter) / (s1 > 0.0 ? s1 : 1e-9);
    const double hs2 = double(iter) / (s2 > 0.0 ? s2 : 1e-9);
    const double ns1 = (s1 * 1e9) / double(iter);
    const double ns2 = (s2 * 1e9) / double(iter);

    std::printf("G1,%.6f,%.6f\n", hs1, ns1);
    std::printf("G2,%.6f,%.6f\n", hs2, ns2);
    return 0;
}
