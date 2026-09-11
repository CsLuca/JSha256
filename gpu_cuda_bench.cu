#include <cuda_runtime.h>

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t _err = (call);                                             \
        if (_err != cudaSuccess) {                                             \
            std::fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__,      \
                         __LINE__, cudaGetErrorString(_err));                  \
            return false;                                                      \
        }                                                                      \
    } while (0)

namespace {

constexpr uint32_t kShaInit[8] = {
    0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u, 0xa54ff53au,
    0x510e527fu, 0x9b05688cu, 0x1f83d9abu, 0x5be0cd19u};

constexpr uint32_t kShaKHost[64] = {
    0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u, 0x3956c25bu,
    0x59f111f1u, 0x923f82a4u, 0xab1c5ed5u, 0xd807aa98u, 0x12835b01u,
    0x243185beu, 0x550c7dc3u, 0x72be5d74u, 0x80deb1feu, 0x9bdc06a7u,
    0xc19bf174u, 0xe49b69c1u, 0xefbe4786u, 0x0fc19dc6u, 0x240ca1ccu,
    0x2de92c6fu, 0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau, 0x983e5152u,
    0xa831c66du, 0xb00327c8u, 0xbf597fc7u, 0xc6e00bf3u, 0xd5a79147u,
    0x06ca6351u, 0x14292967u, 0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu,
    0x53380d13u, 0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u,
    0xa2bfe8a1u, 0xa81a664bu, 0xc24b8b70u, 0xc76c51a3u, 0xd192e819u,
    0xd6990624u, 0xf40e3585u, 0x106aa070u, 0x19a4c116u, 0x1e376c08u,
    0x2748774cu, 0x34b0bcb5u, 0x391c0cb3u, 0x4ed8aa4au, 0x5b9cca4fu,
    0x682e6ff3u, 0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u,
    0x90befffau, 0xa4506cebu, 0xbef9a3f7u, 0xc67178f2u};

__constant__ uint8_t c_header[80];
__constant__ uint32_t c_midstate[8];
__constant__ uint32_t c_sha_k[64] = {
    0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u, 0x3956c25bu,
    0x59f111f1u, 0x923f82a4u, 0xab1c5ed5u, 0xd807aa98u, 0x12835b01u,
    0x243185beu, 0x550c7dc3u, 0x72be5d74u, 0x80deb1feu, 0x9bdc06a7u,
    0xc19bf174u, 0xe49b69c1u, 0xefbe4786u, 0x0fc19dc6u, 0x240ca1ccu,
    0x2de92c6fu, 0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau, 0x983e5152u,
    0xa831c66du, 0xb00327c8u, 0xbf597fc7u, 0xc6e00bf3u, 0xd5a79147u,
    0x06ca6351u, 0x14292967u, 0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu,
    0x53380d13u, 0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u,
    0xa2bfe8a1u, 0xa81a664bu, 0xc24b8b70u, 0xc76c51a3u, 0xd192e819u,
    0xd6990624u, 0xf40e3585u, 0x106aa070u, 0x19a4c116u, 0x1e376c08u,
    0x2748774cu, 0x34b0bcb5u, 0x391c0cb3u, 0x4ed8aa4au, 0x5b9cca4fu,
    0x682e6ff3u, 0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u,
    0x90befffau, 0xa4506cebu, 0xbef9a3f7u, 0xc67178f2u};

struct KnownHash {
    uint32_t nonce;
    const char* hex;
};

constexpr KnownHash kKnown[] = {
    {0u, "0214cabb113df87d438977fd1f4ce32d17ca9d228076596adb2e655abe361e61"},
    {1u, "65decd3eeac87b5d993f5e821a0512937ce3ec14a92e5f75a6898ba1af885dfa"},
    {42u, "24fae796bb9468816192bbfc96a6f03ecd5ee41cd3b911f5cba90c94d2069514"},
    {123456u, "7d4c06e486282268f74ef86360057532417bcb0b90c21814adee7bfb01bae8eb"}};

constexpr size_t kKnownCount = sizeof(kKnown) / sizeof(kKnown[0]);

struct Options {
    int iter = 120000;
    bool run_check = true;
};

struct Timing {
    double g1_seconds = 0.0;
    double g2_seconds = 0.0;
    double g3_seconds = 0.0;
    double g4_seconds = 0.0;
    double g5_seconds = 0.0;
    double g6_seconds = 0.0;
};

__host__ __device__ inline uint32_t rotr(uint32_t x, uint32_t n) {
    return (x >> n) | (x << (32u - n));
}

__host__ __device__ inline uint32_t load_be(const uint8_t* p) {
    return (static_cast<uint32_t>(p[0]) << 24u) |
           (static_cast<uint32_t>(p[1]) << 16u) |
           (static_cast<uint32_t>(p[2]) << 8u) |
           static_cast<uint32_t>(p[3]);
}

__host__ __device__ inline void store_be(uint8_t* p, uint32_t x) {
    p[0] = static_cast<uint8_t>((x >> 24u) & 0xffu);
    p[1] = static_cast<uint8_t>((x >> 16u) & 0xffu);
    p[2] = static_cast<uint8_t>((x >> 8u) & 0xffu);
    p[3] = static_cast<uint8_t>(x & 0xffu);
}

__host__ __device__ inline void sha256_compress(uint32_t state[8],
                                                const uint8_t block[64],
                                                const uint32_t k[64]) {
    uint32_t w[64];
    for (int i = 0; i < 16; ++i) {
        w[i] = load_be(block + i * 4);
    }
    for (int i = 16; i < 64; ++i) {
        const uint32_t s0 = rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >> 3u);
        const uint32_t s1 = rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >> 10u);
        w[i] = w[i - 16] + s0 + w[i - 7] + s1;
    }

    uint32_t a = state[0];
    uint32_t b = state[1];
    uint32_t c = state[2];
    uint32_t d = state[3];
    uint32_t e = state[4];
    uint32_t f = state[5];
    uint32_t g = state[6];
    uint32_t h = state[7];

    for (int i = 0; i < 64; ++i) {
        const uint32_t S1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
        const uint32_t ch = (e & f) ^ ((~e) & g);
        const uint32_t temp1 = h + S1 + ch + k[i] + w[i];
        const uint32_t S0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
        const uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
        const uint32_t temp2 = S0 + maj;
        h = g;
        g = f;
        f = e;
        e = d + temp1;
        d = c;
        c = b;
        b = a;
        a = temp1 + temp2;
    }

    state[0] += a;
    state[1] += b;
    state[2] += c;
    state[3] += d;
    state[4] += e;
    state[5] += f;
    state[6] += g;
    state[7] += h;
}

__host__ __device__ inline void sha256_compress_words(uint32_t state[8],
                                                      const uint32_t w16[16],
                                                      const uint32_t k[64]) {
    uint32_t w[64];
    for (int i = 0; i < 16; ++i) {
        w[i] = w16[i];
    }
    for (int i = 16; i < 64; ++i) {
        const uint32_t s0 = rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >> 3u);
        const uint32_t s1 = rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >> 10u);
        w[i] = w[i - 16] + s0 + w[i - 7] + s1;
    }

    uint32_t a = state[0];
    uint32_t b = state[1];
    uint32_t c = state[2];
    uint32_t d = state[3];
    uint32_t e = state[4];
    uint32_t f = state[5];
    uint32_t g = state[6];
    uint32_t h = state[7];

    for (int i = 0; i < 64; ++i) {
        const uint32_t S1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
        const uint32_t ch = (e & f) ^ ((~e) & g);
        const uint32_t temp1 = h + S1 + ch + k[i] + w[i];
        const uint32_t S0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
        const uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
        const uint32_t temp2 = S0 + maj;
        h = g;
        g = f;
        f = e;
        e = d + temp1;
        d = c;
        c = b;
        b = a;
        a = temp1 + temp2;
    }

    state[0] += a;
    state[1] += b;
    state[2] += c;
    state[3] += d;
    state[4] += e;
    state[5] += f;
    state[6] += g;
    state[7] += h;
}

__host__ __device__ inline void build_second_block_80(const uint8_t tail16[16],
                                                       uint32_t nonce,
                                                       uint8_t out64[64]) {
    for (int i = 0; i < 64; ++i) {
        out64[i] = 0;
    }
    for (int i = 0; i < 16; ++i) {
        out64[i] = tail16[i];
    }
    store_be(out64 + 12, nonce);
    out64[16] = 0x80;
    out64[62] = 0x02;
    out64[63] = 0x80;
}

__host__ __device__ inline void build_second_hash_block(const uint32_t first_digest[8],
                                                         uint8_t out64[64]) {
    for (int i = 0; i < 64; ++i) {
        out64[i] = 0;
    }
    for (int i = 0; i < 8; ++i) {
        store_be(out64 + i * 4, first_digest[i]);
    }
    out64[32] = 0x80;
    out64[62] = 0x01;
    out64[63] = 0x00;
}

__device__ inline void double_sha256_header_g1(uint32_t nonce, uint32_t out_digest[8]) {
    uint32_t state1[8];
    for (int i = 0; i < 8; ++i) {
        state1[i] = kShaInit[i];
    }

    uint8_t block1[64];
    sha256_compress(state1, c_header, c_sha_k);
    build_second_block_80(c_header + 64, nonce, block1);
    sha256_compress(state1, block1, c_sha_k);

    uint32_t state2[8];
    uint8_t block2[64];
    for (int i = 0; i < 8; ++i) {
        state2[i] = kShaInit[i];
    }
    build_second_hash_block(state1, block2);
    sha256_compress(state2, block2, c_sha_k);

    for (int i = 0; i < 8; ++i) {
        out_digest[i] = state2[i];
    }
}

__device__ inline void double_sha256_header_g2(uint32_t nonce, uint32_t out_digest[8]) {
    uint32_t state1[8];
    for (int i = 0; i < 8; ++i) {
        state1[i] = c_midstate[i];
    }

    uint8_t block1[64];
    build_second_block_80(c_header + 64, nonce, block1);
    sha256_compress(state1, block1, c_sha_k);

    uint32_t state2[8];
    uint8_t block2[64];
    for (int i = 0; i < 8; ++i) {
        state2[i] = kShaInit[i];
    }
    build_second_hash_block(state1, block2);
    sha256_compress(state2, block2, c_sha_k);

    for (int i = 0; i < 8; ++i) {
        out_digest[i] = state2[i];
    }
}

__device__ inline void double_sha256_header_g3_sched(uint32_t nonce, uint32_t out_digest[8]) {
    uint32_t state1[8];
    for (int i = 0; i < 8; ++i) {
        state1[i] = c_midstate[i];
    }

    uint32_t w1[16];
    w1[0] = load_be(c_header + 64);
    w1[1] = load_be(c_header + 68);
    w1[2] = load_be(c_header + 72);
    w1[3] = nonce;
    w1[4] = 0x80000000u;
    for (int i = 5; i < 15; ++i) {
        w1[i] = 0u;
    }
    w1[15] = 640u;
    sha256_compress_words(state1, w1, c_sha_k);

    uint32_t state2[8];
    for (int i = 0; i < 8; ++i) {
        state2[i] = kShaInit[i];
    }

    uint32_t w2[16];
    for (int i = 0; i < 8; ++i) {
        w2[i] = state1[i];
    }
    w2[8] = 0x80000000u;
    for (int i = 9; i < 15; ++i) {
        w2[i] = 0u;
    }
    w2[15] = 256u;
    sha256_compress_words(state2, w2, c_sha_k);

    for (int i = 0; i < 8; ++i) {
        out_digest[i] = state2[i];
    }
}

__global__ void g1_naive_kernel(uint32_t* out, uint32_t iter, uint32_t nonce_base) {
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= iter) {
        return;
    }
    uint32_t digest[8];
    double_sha256_header_g1(nonce_base + idx, digest);
    out[idx] = digest[0];
}

__global__ void g2_midstate_kernel(uint32_t* out, uint32_t iter, uint32_t nonce_base) {
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= iter) {
        return;
    }
    uint32_t digest[8];
    double_sha256_header_g2(nonce_base + idx, digest);
    out[idx] = digest[0];
}

__global__ void g3_schedule_kernel(uint32_t* out, uint32_t iter, uint32_t nonce_base) {
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= iter) {
        return;
    }
    uint32_t digest[8];
    double_sha256_header_g3_sched(nonce_base + idx, digest);
    out[idx] = digest[0];
}

__global__ void g4_batched_kernel(uint32_t* out, uint32_t iter, uint32_t nonce_base) {
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t base = nonce_base + idx * 4u;
    const uint32_t end_nonce = nonce_base + iter;
    if (base >= end_nonce) {
        return;
    }
    uint32_t acc = 0u;
    for (uint32_t lane = 0; lane < 4u; ++lane) {
        const uint32_t n = base + lane;
        if (n >= end_nonce) {
            break;
        }
        uint32_t digest[8];
        double_sha256_header_g3_sched(n, digest);
        acc ^= digest[0] ^ (digest[1] + lane);
    }
    out[idx] = acc;
}

__global__ void g5_stream_kernel(uint32_t* out, uint32_t iter, uint32_t nonce_base) {
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= iter) {
        return;
    }
    uint32_t digest[8];
    double_sha256_header_g3_sched(nonce_base + idx, digest);
    out[idx] = digest[2];
}

__global__ __launch_bounds__(256, 2) void g6_tuned_kernel(uint32_t* out,
                                                           uint32_t iter,
                                                           uint32_t nonce_base) {
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t stride = blockDim.x * gridDim.x;
    for (uint32_t i = idx; i < iter; i += stride) {
        uint32_t digest[8];
        double_sha256_header_g3_sched(nonce_base + i, digest);
        out[i] = digest[3];
    }
}

__global__ void g1_check_kernel(const uint32_t* nonces, uint8_t* out_hashes, uint32_t count) {
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) {
        return;
    }
    uint32_t digest[8];
    double_sha256_header_g1(nonces[idx], digest);
    uint8_t* out = out_hashes + idx * 32;
    for (int i = 0; i < 8; ++i) {
        store_be(out + i * 4, digest[i]);
    }
}

__global__ void g2_check_kernel(const uint32_t* nonces, uint8_t* out_hashes, uint32_t count) {
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) {
        return;
    }
    uint32_t digest[8];
    double_sha256_header_g2(nonces[idx], digest);
    uint8_t* out = out_hashes + idx * 32;
    for (int i = 0; i < 8; ++i) {
        store_be(out + i * 4, digest[i]);
    }
}

void init_example_header(uint8_t header[80]) {
    std::memset(header, 0, 80);
    header[0] = 0x01;
    for (int i = 0; i < 32; ++i) {
        header[4 + i] = static_cast<uint8_t>(i);
        header[36 + i] = static_cast<uint8_t>(0x20 + i);
    }
    store_be(header + 68, 0x65F00000u);
    store_be(header + 72, 0x1d00ffffu);
    store_be(header + 76, 0u);
}

void compute_midstate_host(const uint8_t header[80], uint32_t mid[8]) {
    for (int i = 0; i < 8; ++i) {
        mid[i] = kShaInit[i];
    }
    sha256_compress(mid, header, kShaKHost);
}

int hex_nibble(char c) {
    if (c >= '0' && c <= '9') {
        return c - '0';
    }
    if (c >= 'a' && c <= 'f') {
        return 10 + (c - 'a');
    }
    if (c >= 'A' && c <= 'F') {
        return 10 + (c - 'A');
    }
    return -1;
}

bool parse_hex_32(const char* hex, uint8_t out[32]) {
    const size_t n = std::strlen(hex);
    if (n != 64) {
        return false;
    }
    for (size_t i = 0; i < 32; ++i) {
        const int hi = hex_nibble(hex[i * 2]);
        const int lo = hex_nibble(hex[i * 2 + 1]);
        if (hi < 0 || lo < 0) {
            return false;
        }
        out[i] = static_cast<uint8_t>((hi << 4) | lo);
    }
    return true;
}

bool run_correctness_check() {
    const uint32_t count = static_cast<uint32_t>(kKnownCount);
    uint32_t nonces[kKnownCount];
    for (uint32_t i = 0; i < count; ++i) {
        nonces[i] = kKnown[i].nonce;
    }

    uint32_t* d_nonces = nullptr;
    uint8_t* d_g1 = nullptr;
    uint8_t* d_g2 = nullptr;
    CUDA_CHECK(cudaMalloc(&d_nonces, count * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_g1, count * 32));
    CUDA_CHECK(cudaMalloc(&d_g2, count * 32));
    CUDA_CHECK(cudaMemcpy(d_nonces, nonces, count * sizeof(uint32_t), cudaMemcpyHostToDevice));

    const int threads = 128;
    const int blocks = static_cast<int>((count + threads - 1) / threads);
    g1_check_kernel<<<blocks, threads>>>(d_nonces, d_g1, count);
    CUDA_CHECK(cudaGetLastError());
    g2_check_kernel<<<blocks, threads>>>(d_nonces, d_g2, count);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    uint8_t got_g1[kKnownCount * 32];
    uint8_t got_g2[kKnownCount * 32];
    CUDA_CHECK(cudaMemcpy(got_g1, d_g1, count * 32, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(got_g2, d_g2, count * 32, cudaMemcpyDeviceToHost));

    bool ok = true;
    for (uint32_t i = 0; i < count; ++i) {
        uint8_t expected[32];
        if (!parse_hex_32(kKnown[i].hex, expected)) {
            ok = false;
            break;
        }
        if (std::memcmp(got_g1 + i * 32, expected, 32) != 0 ||
            std::memcmp(got_g2 + i * 32, expected, 32) != 0) {
            ok = false;
            break;
        }
    }

    cudaFree(d_nonces);
    cudaFree(d_g1);
    cudaFree(d_g2);
    return ok;
}

bool parse_options(int argc, char** argv, Options& options) {
    options = Options{};
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--iter") == 0 && i + 1 < argc) {
            options.iter = std::atoi(argv[++i]);
            continue;
        }
        if (std::strcmp(argv[i], "--no-check") == 0) {
            options.run_check = false;
            continue;
        }
    }
    if (options.iter <= 0) {
        options.iter = 1;
    }
    return true;
}

bool run_g5_multistream_overlap(int iter, double& seconds) {
    constexpr int kStreams = 3;
    constexpr int kThreads = 256;
    constexpr int kChunk = 65536;

    cudaStream_t streams[kStreams] = {};
    uint32_t* d_out[kStreams] = {};
    uint32_t* h_out[kStreams] = {};
    bool stream_used[kStreams] = {false, false, false};

    for (int i = 0; i < kStreams; ++i) {
        CUDA_CHECK(cudaStreamCreate(&streams[i]));
        CUDA_CHECK(cudaMalloc(&d_out[i], static_cast<size_t>(kChunk) * sizeof(uint32_t)));
        CUDA_CHECK(cudaHostAlloc(&h_out[i], static_cast<size_t>(kChunk) * sizeof(uint32_t), cudaHostAllocDefault));
    }

    CUDA_CHECK(cudaDeviceSynchronize());
    auto t0 = std::chrono::high_resolution_clock::now();

    for (int offset = 0; offset < iter; offset += kChunk * kStreams) {
        for (int s = 0; s < kStreams; ++s) {
            const int start = offset + s * kChunk;
            if (start >= iter) {
                continue;
            }
            const int remaining = iter - start;
            const int count = remaining < kChunk ? remaining : kChunk;
            const int blocks = (count + kThreads - 1) / kThreads;
            stream_used[s] = true;

            g5_stream_kernel<<<blocks, kThreads, 0, streams[s]>>>(
                d_out[s], static_cast<uint32_t>(count), static_cast<uint32_t>(start));
            CUDA_CHECK(cudaGetLastError());

            CUDA_CHECK(cudaMemcpyAsync(h_out[s], d_out[s],
                                       static_cast<size_t>(count) * sizeof(uint32_t),
                                       cudaMemcpyDeviceToHost, streams[s]));
        }
    }

    for (int i = 0; i < kStreams; ++i) {
        CUDA_CHECK(cudaStreamSynchronize(streams[i]));
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    seconds = std::chrono::duration<double>(t1 - t0).count();

    volatile uint32_t sink = 0;
    for (int i = 0; i < kStreams; ++i) {
        if (stream_used[i]) {
            sink ^= h_out[i][0];
        }
    }
    (void)sink;

    for (int i = 0; i < kStreams; ++i) {
        if (h_out[i]) {
            cudaFreeHost(h_out[i]);
        }
        if (d_out[i]) {
            cudaFree(d_out[i]);
        }
        if (streams[i]) {
            cudaStreamDestroy(streams[i]);
        }
    }
    return true;
}

bool run_benchmarks(int iter, Timing& timing) {
    const int threads = 256;
    const int blocks = (iter + threads - 1) / threads;
    const int iter_g4 = (iter + 3) / 4;
    const int blocks_g4 = (iter_g4 + threads - 1) / threads;

    uint32_t* d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_out, static_cast<size_t>(iter) * sizeof(uint32_t)));

    cudaDeviceSynchronize();
    auto t0 = std::chrono::high_resolution_clock::now();
    g1_naive_kernel<<<blocks, threads>>>(d_out, static_cast<uint32_t>(iter), 0u);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    auto t1 = std::chrono::high_resolution_clock::now();

    g2_midstate_kernel<<<blocks, threads>>>(d_out, static_cast<uint32_t>(iter), 0u);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    auto t2 = std::chrono::high_resolution_clock::now();

    g3_schedule_kernel<<<blocks, threads>>>(d_out, static_cast<uint32_t>(iter), 0u);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    auto t3 = std::chrono::high_resolution_clock::now();

    g4_batched_kernel<<<blocks_g4, threads>>>(d_out, static_cast<uint32_t>(iter), 0u);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    auto t4 = std::chrono::high_resolution_clock::now();

    CUDA_CHECK(cudaFree(d_out));
    d_out = nullptr;
    if (!run_g5_multistream_overlap(iter, timing.g5_seconds)) {
        return false;
    }
    auto t5 = std::chrono::high_resolution_clock::now();

    CUDA_CHECK(cudaMalloc(&d_out, static_cast<size_t>(iter) * sizeof(uint32_t)));

    int dev = 0;
    CUDA_CHECK(cudaGetDevice(&dev));
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
    const int threads_g6 = 256;
    int blocks_g6 = prop.multiProcessorCount * 12;
    if (blocks_g6 < 1) {
        blocks_g6 = 1;
    }
    g6_tuned_kernel<<<blocks_g6, threads_g6>>>(d_out, static_cast<uint32_t>(iter), 0u);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    auto t6 = std::chrono::high_resolution_clock::now();

    cudaFree(d_out);

    timing.g1_seconds = std::chrono::duration<double>(t1 - t0).count();
    timing.g2_seconds = std::chrono::duration<double>(t2 - t1).count();
    timing.g3_seconds = std::chrono::duration<double>(t3 - t2).count();
    timing.g4_seconds = std::chrono::duration<double>(t4 - t3).count();
    timing.g6_seconds = std::chrono::duration<double>(t6 - t5).count();
    return true;
}

void print_row(const char* label, double seconds, int iter) {
    const double safe_seconds = seconds > 0.0 ? seconds : 1e-9;
    const double hs = static_cast<double>(iter) / safe_seconds;
    const double ns = (safe_seconds * 1e9) / static_cast<double>(iter);
    std::printf("%s,%.6f,%.6f\n", label, hs, ns);
}

}  // namespace

int main(int argc, char** argv) {
    Options options{};
    parse_options(argc, argv, options);

    int dev_count = 0;
    cudaError_t dev_rc = cudaGetDeviceCount(&dev_count);
    if (dev_rc != cudaSuccess || dev_count <= 0) {
        std::fprintf(stderr, "No CUDA device available: %s\n", cudaGetErrorString(dev_rc));
        return 2;
    }

    uint8_t header[80];
    uint32_t midstate[8];
    init_example_header(header);
    compute_midstate_host(header, midstate);

    if (cudaMemcpyToSymbol(c_header, header, sizeof(header)) != cudaSuccess ||
        cudaMemcpyToSymbol(c_midstate, midstate, sizeof(midstate)) != cudaSuccess) {
        std::fprintf(stderr, "Failed to upload header/midstate constants to CUDA device\n");
        return 3;
    }

    if (options.run_check) {
        if (!run_correctness_check()) {
            std::fprintf(stderr, "Correctness check failed for G1/G2\n");
            return 4;
        }
    }

    Timing timing{};
    if (!run_benchmarks(options.iter, timing)) {
        return 5;
    }

    print_row("G1", timing.g1_seconds, options.iter);
    print_row("G2", timing.g2_seconds, options.iter);
    print_row("G3", timing.g3_seconds, options.iter);
    print_row("G4", timing.g4_seconds, options.iter);
    print_row("G5", timing.g5_seconds, options.iter);
    print_row("G6", timing.g6_seconds, options.iter);
    return 0;
}
