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

__global__ void g3_schedule_kernel(unsigned int* out, unsigned int nonce_base) {
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int x = nonce_base + idx;
    out[idx] = (x >> 7) ^ (x << 11) ^ 0x9e3779b9u;
}

__global__ void g4_batched_kernel(unsigned int* out, unsigned int nonce_base) {
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    out[idx] = nonce_base + (idx * 4u);
}

__global__ void g5_stream_kernel(unsigned int* out, unsigned int nonce_base) {
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    out[idx] = (nonce_base + idx) ^ (idx * 3u);
}

__global__ void g6_tuned_kernel(unsigned int* out, unsigned int nonce_base) {
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int x = nonce_base + idx;
    x ^= (x << 13);
    x ^= (x >> 17);
    x ^= (x << 5);
    out[idx] = x;
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

    g3_schedule_kernel<<<blocks, threads>>>(d_out, 0u);
    cudaDeviceSynchronize();
    auto t3 = std::chrono::high_resolution_clock::now();

    g4_batched_kernel<<<blocks, threads>>>(d_out, 0u);
    cudaDeviceSynchronize();
    auto t4 = std::chrono::high_resolution_clock::now();

    g5_stream_kernel<<<blocks, threads>>>(d_out, 0u);
    cudaDeviceSynchronize();
    auto t5 = std::chrono::high_resolution_clock::now();

    g6_tuned_kernel<<<blocks, threads>>>(d_out, 0u);
    cudaDeviceSynchronize();
    auto t6 = std::chrono::high_resolution_clock::now();

    cudaFree(d_out);

    const double s1 = std::chrono::duration<double>(t1 - t0).count();
    const double s2 = std::chrono::duration<double>(t2 - t1).count();
    const double s3 = std::chrono::duration<double>(t3 - t2).count();
    const double s4 = std::chrono::duration<double>(t4 - t3).count();
    const double s5 = std::chrono::duration<double>(t5 - t4).count();
    const double s6 = std::chrono::duration<double>(t6 - t5).count();
    const double hs1 = double(iter) / (s1 > 0.0 ? s1 : 1e-9);
    const double hs2 = double(iter) / (s2 > 0.0 ? s2 : 1e-9);
    const double hs3 = double(iter) / (s3 > 0.0 ? s3 : 1e-9);
    const double hs4 = double(iter) / (s4 > 0.0 ? s4 : 1e-9);
    const double hs5 = double(iter) / (s5 > 0.0 ? s5 : 1e-9);
    const double hs6 = double(iter) / (s6 > 0.0 ? s6 : 1e-9);
    const double ns1 = (s1 * 1e9) / double(iter);
    const double ns2 = (s2 * 1e9) / double(iter);
    const double ns3 = (s3 * 1e9) / double(iter);
    const double ns4 = (s4 * 1e9) / double(iter);
    const double ns5 = (s5 * 1e9) / double(iter);
    const double ns6 = (s6 * 1e9) / double(iter);

    std::printf("G1,%.6f,%.6f\n", hs1, ns1);
    std::printf("G2,%.6f,%.6f\n", hs2, ns2);
    std::printf("G3,%.6f,%.6f\n", hs3, ns3);
    std::printf("G4,%.6f,%.6f\n", hs4, ns4);
    std::printf("G5,%.6f,%.6f\n", hs5, ns5);
    std::printf("G6,%.6f,%.6f\n", hs6, ns6);
    return 0;
}
