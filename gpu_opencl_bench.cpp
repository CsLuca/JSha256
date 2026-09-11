#include <CL/cl.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace {

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
    bool check_all = false;
};

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

inline uint32_t load_be(const uint8_t* p);

inline uint32_t rotr32_host(uint32_t x, uint32_t n) {
    return (x >> n) | (x << (32u - n));
}

void sha256_compress_words_host(uint32_t state[8], const uint32_t w16[16]) {
    uint32_t w[64];
    for (int i = 0; i < 16; ++i) w[i] = w16[i];
    for (int i = 16; i < 64; ++i) {
        const uint32_t s0 = rotr32_host(w[i - 15], 7) ^ rotr32_host(w[i - 15], 18) ^ (w[i - 15] >> 3u);
        const uint32_t s1 = rotr32_host(w[i - 2], 17) ^ rotr32_host(w[i - 2], 19) ^ (w[i - 2] >> 10u);
        w[i] = w[i - 16] + s0 + w[i - 7] + s1;
    }

    uint32_t a = state[0], b = state[1], c = state[2], d = state[3];
    uint32_t e = state[4], f = state[5], g = state[6], h = state[7];
    for (int i = 0; i < 64; ++i) {
        const uint32_t S1 = rotr32_host(e, 6) ^ rotr32_host(e, 11) ^ rotr32_host(e, 25);
        const uint32_t ch = (e & f) ^ ((~e) & g);
        const uint32_t t1 = h + S1 + ch + kShaKHost[i] + w[i];
        const uint32_t S0 = rotr32_host(a, 2) ^ rotr32_host(a, 13) ^ rotr32_host(a, 22);
        const uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
        const uint32_t t2 = S0 + maj;
        h = g; g = f; f = e; e = d + t1;
        d = c; c = b; b = a; a = t1 + t2;
    }
    state[0] += a; state[1] += b; state[2] += c; state[3] += d;
    state[4] += e; state[5] += f; state[6] += g; state[7] += h;
}

void compute_midstate_host(const uint8_t header[80], uint32_t midstate[8]) {
    for (int i = 0; i < 8; ++i) midstate[i] = kShaInit[i];
    uint32_t w16[16];
    for (int i = 0; i < 16; ++i) {
        w16[i] = load_be(header + i * 4);
    }
    sha256_compress_words_host(midstate, w16);
}

inline uint32_t load_be(const uint8_t* p) {
    return (static_cast<uint32_t>(p[0]) << 24u) |
           (static_cast<uint32_t>(p[1]) << 16u) |
           (static_cast<uint32_t>(p[2]) << 8u) |
           static_cast<uint32_t>(p[3]);
}

inline void store_be(uint8_t* p, uint32_t x) {
    p[0] = static_cast<uint8_t>((x >> 24u) & 0xffu);
    p[1] = static_cast<uint8_t>((x >> 16u) & 0xffu);
    p[2] = static_cast<uint8_t>((x >> 8u) & 0xffu);
    p[3] = static_cast<uint8_t>(x & 0xffu);
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

int hex_nibble(char c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return 10 + (c - 'a');
    if (c >= 'A' && c <= 'F') return 10 + (c - 'A');
    return -1;
}

bool parse_hex_32(const char* hex, uint8_t out[32]) {
    if (std::strlen(hex) != 64) return false;
    for (size_t i = 0; i < 32; ++i) {
        const int hi = hex_nibble(hex[i * 2]);
        const int lo = hex_nibble(hex[i * 2 + 1]);
        if (hi < 0 || lo < 0) return false;
        out[i] = static_cast<uint8_t>((hi << 4) | lo);
    }
    return true;
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
        if (std::strcmp(argv[i], "--check-all") == 0) {
            options.check_all = true;
            options.run_check = true;
            continue;
        }
    }
    if (options.iter <= 0) options.iter = 1;
    return true;
}

const char* kKernelSource = R"CLC(
uint rotr32(uint x, uint n) {
    return (x >> n) | (x << (32u - n));
}

uint load_be_u32(const __global uchar* p) {
    return ((uint)p[0] << 24u) | ((uint)p[1] << 16u) | ((uint)p[2] << 8u) | (uint)p[3];
}

void sha256_compress_words(__private uint state[8], __private uint w[64]) {
    const uint K[64] = {
        0x428a2f98u,0x71374491u,0xb5c0fbcfu,0xe9b5dba5u,0x3956c25bu,0x59f111f1u,0x923f82a4u,0xab1c5ed5u,
        0xd807aa98u,0x12835b01u,0x243185beu,0x550c7dc3u,0x72be5d74u,0x80deb1feu,0x9bdc06a7u,0xc19bf174u,
        0xe49b69c1u,0xefbe4786u,0x0fc19dc6u,0x240ca1ccu,0x2de92c6fu,0x4a7484aau,0x5cb0a9dcu,0x76f988dau,
        0x983e5152u,0xa831c66du,0xb00327c8u,0xbf597fc7u,0xc6e00bf3u,0xd5a79147u,0x06ca6351u,0x14292967u,
        0x27b70a85u,0x2e1b2138u,0x4d2c6dfcu,0x53380d13u,0x650a7354u,0x766a0abbu,0x81c2c92eu,0x92722c85u,
        0xa2bfe8a1u,0xa81a664bu,0xc24b8b70u,0xc76c51a3u,0xd192e819u,0xd6990624u,0xf40e3585u,0x106aa070u,
        0x19a4c116u,0x1e376c08u,0x2748774cu,0x34b0bcb5u,0x391c0cb3u,0x4ed8aa4au,0x5b9cca4fu,0x682e6ff3u,
        0x748f82eeu,0x78a5636fu,0x84c87814u,0x8cc70208u,0x90befffau,0xa4506cebu,0xbef9a3f7u,0xc67178f2u
    };

    for (int i = 16; i < 64; ++i) {
        uint s0 = rotr32(w[i - 15], 7) ^ rotr32(w[i - 15], 18) ^ (w[i - 15] >> 3u);
        uint s1 = rotr32(w[i - 2], 17) ^ rotr32(w[i - 2], 19) ^ (w[i - 2] >> 10u);
        w[i] = w[i - 16] + s0 + w[i - 7] + s1;
    }

    uint a = state[0], b = state[1], c = state[2], d = state[3];
    uint e = state[4], f = state[5], g = state[6], h = state[7];
    for (int i = 0; i < 64; ++i) {
        uint S1 = rotr32(e, 6) ^ rotr32(e, 11) ^ rotr32(e, 25);
        uint ch = (e & f) ^ ((~e) & g);
        uint t1 = h + S1 + ch + K[i] + w[i];
        uint S0 = rotr32(a, 2) ^ rotr32(a, 13) ^ rotr32(a, 22);
        uint maj = (a & b) ^ (a & c) ^ (b & c);
        uint t2 = S0 + maj;
        h = g; g = f; f = e; e = d + t1;
        d = c; c = b; b = a; a = t1 + t2;
    }
    state[0] += a; state[1] += b; state[2] += c; state[3] += d;
    state[4] += e; state[5] += f; state[6] += g; state[7] += h;
}

void double_sha256_header_g1(const __global uchar* header80, uint nonce, __private uint out_digest[8]) {
    uint state1[8] = {0x6a09e667u,0xbb67ae85u,0x3c6ef372u,0xa54ff53au,0x510e527fu,0x9b05688cu,0x1f83d9abu,0x5be0cd19u};
    uint w1[64];
    for (int i = 0; i < 16; ++i) {
        w1[i] = load_be_u32(header80 + i * 4);
    }
    sha256_compress_words(state1, w1);

    uint w2[64];
    for (int i = 0; i < 16; ++i) w2[i] = 0u;
    w2[0] = load_be_u32(header80 + 64);
    w2[1] = load_be_u32(header80 + 68);
    w2[2] = load_be_u32(header80 + 72);
    w2[3] = nonce;
    w2[4] = 0x80000000u;
    w2[15] = 640u;
    sha256_compress_words(state1, w2);

    uint state2[8] = {0x6a09e667u,0xbb67ae85u,0x3c6ef372u,0xa54ff53au,0x510e527fu,0x9b05688cu,0x1f83d9abu,0x5be0cd19u};
    uint w3[64];
    for (int i = 0; i < 16; ++i) w3[i] = 0u;
    for (int i = 0; i < 8; ++i) w3[i] = state1[i];
    w3[8] = 0x80000000u;
    w3[15] = 256u;
    sha256_compress_words(state2, w3);

    for (int i = 0; i < 8; ++i) out_digest[i] = state2[i];
}

void double_sha256_header_g2(const __global uchar* header80,
                             const __global uint* midstate,
                             uint nonce,
                             __private uint out_digest[8]) {
    uint state1[8];
    for (int i = 0; i < 8; ++i) state1[i] = midstate[i];

    uint w2[64];
    for (int i = 0; i < 16; ++i) w2[i] = 0u;
    w2[0] = load_be_u32(header80 + 64);
    w2[1] = load_be_u32(header80 + 68);
    w2[2] = load_be_u32(header80 + 72);
    w2[3] = nonce;
    w2[4] = 0x80000000u;
    w2[15] = 640u;
    sha256_compress_words(state1, w2);

    uint state2[8] = {0x6a09e667u,0xbb67ae85u,0x3c6ef372u,0xa54ff53au,0x510e527fu,0x9b05688cu,0x1f83d9abu,0x5be0cd19u};
    uint w3[64];
    for (int i = 0; i < 16; ++i) w3[i] = 0u;
    for (int i = 0; i < 8; ++i) w3[i] = state1[i];
    w3[8] = 0x80000000u;
    w3[15] = 256u;
    sha256_compress_words(state2, w3);

    for (int i = 0; i < 8; ++i) out_digest[i] = state2[i];
}

void double_sha256_header_g3(const __global uchar* header80,
                             const __global uint* midstate,
                             uint nonce,
                             __private uint out_digest[8]) {
    uint state1[8];
    for (int i = 0; i < 8; ++i) state1[i] = midstate[i];

    uint w2[64];
    w2[0] = load_be_u32(header80 + 64);
    w2[1] = load_be_u32(header80 + 68);
    w2[2] = load_be_u32(header80 + 72);
    w2[3] = nonce;
    w2[4] = 0x80000000u;
    for (int i = 5; i < 15; ++i) w2[i] = 0u;
    w2[15] = 640u;
    sha256_compress_words(state1, w2);

    uint state2[8] = {0x6a09e667u,0xbb67ae85u,0x3c6ef372u,0xa54ff53au,0x510e527fu,0x9b05688cu,0x1f83d9abu,0x5be0cd19u};
    uint w3[64];
    for (int i = 0; i < 8; ++i) w3[i] = state1[i];
    w3[8] = 0x80000000u;
    for (int i = 9; i < 15; ++i) w3[i] = 0u;
    w3[15] = 256u;
    sha256_compress_words(state2, w3);

    for (int i = 0; i < 8; ++i) out_digest[i] = state2[i];
}

__kernel void g1_bench(__global const uchar* header80, uint nonce_base, uint iter, __global uint* out) {
    const uint idx = get_global_id(0);
    if (idx >= iter) return;
    uint digest[8];
    double_sha256_header_g1(header80, nonce_base + idx, digest);
    out[idx] = digest[0];
}

__kernel void g2_bench(__global const uchar* header80,
                       __global const uint* midstate,
                       uint nonce_base,
                       uint iter,
                       __global uint* out) {
    const uint idx = get_global_id(0);
    if (idx >= iter) return;
    uint digest[8];
    double_sha256_header_g2(header80, midstate, nonce_base + idx, digest);
    out[idx] = digest[0];
}

__kernel void g3_bench(__global const uchar* header80,
                       __global const uint* midstate,
                       uint nonce_base,
                       uint iter,
                       __global uint* out) {
    const uint idx = get_global_id(0);
    if (idx >= iter) return;
    uint digest[8];
    double_sha256_header_g3(header80, midstate, nonce_base + idx, digest);
    out[idx] = digest[0];
}

__kernel void g4_bench(__global const uchar* header80,
                       __global const uint* midstate,
                       uint nonce_base,
                       uint iter,
                       __global uint* out) {
    const uint idx = get_global_id(0);
    const uint base = nonce_base + idx * 4u;
    const uint end_nonce = nonce_base + iter;
    if (base >= end_nonce) return;

    uint acc = 0u;
    for (uint lane = 0u; lane < 4u; ++lane) {
        const uint n = base + lane;
        if (n >= end_nonce) break;
        uint digest[8];
        double_sha256_header_g3(header80, midstate, n, digest);
        acc ^= digest[0] ^ (digest[1] + lane);
    }
    out[idx] = acc;
}

__kernel void g1_check(__global const uchar* header80, __global const uint* nonces, uint count, __global uint* out_words) {
    const uint idx = get_global_id(0);
    if (idx >= count) return;
    uint digest[8];
    double_sha256_header_g1(header80, nonces[idx], digest);
    const uint base = idx * 8u;
    for (int i = 0; i < 8; ++i) out_words[base + i] = digest[i];
}

__kernel void g2_check(__global const uchar* header80,
                       __global const uint* midstate,
                       __global const uint* nonces,
                       uint count,
                       __global uint* out_words) {
    const uint idx = get_global_id(0);
    if (idx >= count) return;
    uint digest[8];
    double_sha256_header_g2(header80, midstate, nonces[idx], digest);
    const uint base = idx * 8u;
    for (int i = 0; i < 8; ++i) out_words[base + i] = digest[i];
}

__kernel void g3_check(__global const uchar* header80,
                       __global const uint* midstate,
                       __global const uint* nonces,
                       uint count,
                       __global uint* out_words) {
    const uint idx = get_global_id(0);
    if (idx >= count) return;
    uint digest[8];
    double_sha256_header_g3(header80, midstate, nonces[idx], digest);
    const uint base = idx * 8u;
    for (int i = 0; i < 8; ++i) out_words[base + i] = digest[i];
}

__kernel void g4_check(__global const uchar* header80,
                       __global const uint* midstate,
                       __global const uint* nonces,
                       uint count,
                       __global uint* out_words) {
    const uint idx = get_global_id(0);
    if (idx >= count) return;
    uint digest[8];
    double_sha256_header_g3(header80, midstate, nonces[idx], digest);
    const uint base = idx * 8u;
    for (int i = 0; i < 8; ++i) out_words[base + i] = digest[i];
}
)CLC";

bool check_status(cl_int status, const char* where) {
    if (status == CL_SUCCESS) return true;
    std::fprintf(stderr, "OpenCL error at %s: %d\n", where, static_cast<int>(status));
    return false;
}

}  // namespace

int main(int argc, char** argv) {
    Options options{};
    parse_options(argc, argv, options);

    cl_int st = CL_SUCCESS;
    cl_uint platform_count = 0;
    st = clGetPlatformIDs(0, nullptr, &platform_count);
    if (!check_status(st, "clGetPlatformIDs(count)") || platform_count == 0) return 2;

    std::vector<cl_platform_id> platforms(platform_count);
    st = clGetPlatformIDs(platform_count, platforms.data(), nullptr);
    if (!check_status(st, "clGetPlatformIDs(list)")) return 2;

    cl_device_id device = nullptr;
    for (cl_uint i = 0; i < platform_count && !device; ++i) {
        cl_uint dev_count = 0;
        st = clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_GPU, 0, nullptr, &dev_count);
        if (st != CL_SUCCESS || dev_count == 0) continue;
        std::vector<cl_device_id> devs(dev_count);
        st = clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_GPU, dev_count, devs.data(), nullptr);
        if (st == CL_SUCCESS && !devs.empty()) device = devs[0];
    }
    if (!device) return 2;

    cl_context context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &st);
    if (!check_status(st, "clCreateContext")) return 3;
    cl_command_queue queue = clCreateCommandQueue(context, device, 0, &st);
    if (!check_status(st, "clCreateCommandQueue")) return 3;

    const char* src = kKernelSource;
    const size_t src_len = std::strlen(kKernelSource);
    cl_program program = clCreateProgramWithSource(context, 1, &src, &src_len, &st);
    if (!check_status(st, "clCreateProgramWithSource")) return 3;
    st = clBuildProgram(program, 1, &device, "", nullptr, nullptr);
    if (st != CL_SUCCESS) {
        size_t log_size = 0;
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);
        std::vector<char> log(log_size + 1, 0);
        if (log_size > 0) {
            clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, log.data(), nullptr);
            std::fprintf(stderr, "%s\n", log.data());
        }
        return 3;
    }

    cl_kernel kernel_g1_bench = clCreateKernel(program, "g1_bench", &st);
    if (!check_status(st, "clCreateKernel(g1_bench)")) return 3;
    cl_kernel kernel_g2_bench = clCreateKernel(program, "g2_bench", &st);
    if (!check_status(st, "clCreateKernel(g2_bench)")) return 3;
    cl_kernel kernel_g3_bench = clCreateKernel(program, "g3_bench", &st);
    if (!check_status(st, "clCreateKernel(g3_bench)")) return 3;
    cl_kernel kernel_g4_bench = clCreateKernel(program, "g4_bench", &st);
    if (!check_status(st, "clCreateKernel(g4_bench)")) return 3;
    cl_kernel kernel_g1_check = clCreateKernel(program, "g1_check", &st);
    if (!check_status(st, "clCreateKernel(g1_check)")) return 3;
    cl_kernel kernel_g2_check = clCreateKernel(program, "g2_check", &st);
    if (!check_status(st, "clCreateKernel(g2_check)")) return 3;
    cl_kernel kernel_g3_check = clCreateKernel(program, "g3_check", &st);
    if (!check_status(st, "clCreateKernel(g3_check)")) return 3;
    cl_kernel kernel_g4_check = clCreateKernel(program, "g4_check", &st);
    if (!check_status(st, "clCreateKernel(g4_check)")) return 3;

    uint8_t header[80];
    uint32_t midstate[8];
    init_example_header(header);
    compute_midstate_host(header, midstate);

    cl_mem d_header = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                     sizeof(header), header, &st);
    if (!check_status(st, "clCreateBuffer(header)")) return 3;
    cl_mem d_midstate = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                       sizeof(midstate), midstate, &st);
    if (!check_status(st, "clCreateBuffer(midstate)")) return 3;

    if (options.run_check) {
        uint32_t nonces[kKnownCount];
        for (size_t i = 0; i < kKnownCount; ++i) nonces[i] = kKnown[i].nonce;

        cl_mem d_nonces = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                         sizeof(nonces), nonces, &st);
        if (!check_status(st, "clCreateBuffer(nonces)")) return 4;
        cl_mem d_words_g1 = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
                                           kKnownCount * 8 * sizeof(uint32_t), nullptr, &st);
        if (!check_status(st, "clCreateBuffer(words_g1)")) return 4;
        cl_mem d_words_g2 = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
                                           kKnownCount * 8 * sizeof(uint32_t), nullptr, &st);
        if (!check_status(st, "clCreateBuffer(words_g2)")) return 4;
        cl_mem d_words_g3 = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
                                           kKnownCount * 8 * sizeof(uint32_t), nullptr, &st);
        if (!check_status(st, "clCreateBuffer(words_g3)")) return 4;
        cl_mem d_words_g4 = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
                                           kKnownCount * 8 * sizeof(uint32_t), nullptr, &st);
        if (!check_status(st, "clCreateBuffer(words_g4)")) return 4;

        const uint32_t count_u32 = static_cast<uint32_t>(kKnownCount);
        st = clSetKernelArg(kernel_g1_check, 0, sizeof(cl_mem), &d_header);
        st |= clSetKernelArg(kernel_g1_check, 1, sizeof(cl_mem), &d_nonces);
        st |= clSetKernelArg(kernel_g1_check, 2, sizeof(uint32_t), &count_u32);
        st |= clSetKernelArg(kernel_g1_check, 3, sizeof(cl_mem), &d_words_g1);
        if (!check_status(st, "clSetKernelArg(g1_check)")) return 4;

        if (options.check_all) {
            st = clSetKernelArg(kernel_g2_check, 0, sizeof(cl_mem), &d_header);
            st |= clSetKernelArg(kernel_g2_check, 1, sizeof(cl_mem), &d_midstate);
            st |= clSetKernelArg(kernel_g2_check, 2, sizeof(cl_mem), &d_nonces);
            st |= clSetKernelArg(kernel_g2_check, 3, sizeof(uint32_t), &count_u32);
            st |= clSetKernelArg(kernel_g2_check, 4, sizeof(cl_mem), &d_words_g2);
            if (!check_status(st, "clSetKernelArg(g2_check)")) return 4;

            st = clSetKernelArg(kernel_g3_check, 0, sizeof(cl_mem), &d_header);
            st |= clSetKernelArg(kernel_g3_check, 1, sizeof(cl_mem), &d_midstate);
            st |= clSetKernelArg(kernel_g3_check, 2, sizeof(cl_mem), &d_nonces);
            st |= clSetKernelArg(kernel_g3_check, 3, sizeof(uint32_t), &count_u32);
            st |= clSetKernelArg(kernel_g3_check, 4, sizeof(cl_mem), &d_words_g3);
            if (!check_status(st, "clSetKernelArg(g3_check)")) return 4;

            st = clSetKernelArg(kernel_g4_check, 0, sizeof(cl_mem), &d_header);
            st |= clSetKernelArg(kernel_g4_check, 1, sizeof(cl_mem), &d_midstate);
            st |= clSetKernelArg(kernel_g4_check, 2, sizeof(cl_mem), &d_nonces);
            st |= clSetKernelArg(kernel_g4_check, 3, sizeof(uint32_t), &count_u32);
            st |= clSetKernelArg(kernel_g4_check, 4, sizeof(cl_mem), &d_words_g4);
            if (!check_status(st, "clSetKernelArg(g4_check)")) return 4;
        }

        const size_t local = 64;
        const size_t global = ((kKnownCount + local - 1) / local) * local;
        st = clEnqueueNDRangeKernel(queue, kernel_g1_check, 1, nullptr, &global, &local, 0, nullptr, nullptr);
        if (!check_status(st, "clEnqueueNDRangeKernel(g1_check)")) return 4;
        if (options.check_all) {
            st = clEnqueueNDRangeKernel(queue, kernel_g2_check, 1, nullptr, &global, &local, 0, nullptr, nullptr);
            if (!check_status(st, "clEnqueueNDRangeKernel(g2_check)")) return 4;
            st = clEnqueueNDRangeKernel(queue, kernel_g3_check, 1, nullptr, &global, &local, 0, nullptr, nullptr);
            if (!check_status(st, "clEnqueueNDRangeKernel(g3_check)")) return 4;
            st = clEnqueueNDRangeKernel(queue, kernel_g4_check, 1, nullptr, &global, &local, 0, nullptr, nullptr);
            if (!check_status(st, "clEnqueueNDRangeKernel(g4_check)")) return 4;
        }
        st = clFinish(queue);
        if (!check_status(st, "clFinish(check)")) return 4;

        std::vector<uint32_t> words_g1(kKnownCount * 8, 0);
        std::vector<uint32_t> words_g2(kKnownCount * 8, 0);
        std::vector<uint32_t> words_g3(kKnownCount * 8, 0);
        std::vector<uint32_t> words_g4(kKnownCount * 8, 0);
        st = clEnqueueReadBuffer(queue, d_words_g1, CL_TRUE, 0,
                                 words_g1.size() * sizeof(uint32_t), words_g1.data(), 0, nullptr, nullptr);
        if (!check_status(st, "clEnqueueReadBuffer(g1_check)")) return 4;
        if (options.check_all) {
            st = clEnqueueReadBuffer(queue, d_words_g2, CL_TRUE, 0,
                                     words_g2.size() * sizeof(uint32_t), words_g2.data(), 0, nullptr, nullptr);
            if (!check_status(st, "clEnqueueReadBuffer(g2_check)")) return 4;
            st = clEnqueueReadBuffer(queue, d_words_g3, CL_TRUE, 0,
                                     words_g3.size() * sizeof(uint32_t), words_g3.data(), 0, nullptr, nullptr);
            if (!check_status(st, "clEnqueueReadBuffer(g3_check)")) return 4;
            st = clEnqueueReadBuffer(queue, d_words_g4, CL_TRUE, 0,
                                     words_g4.size() * sizeof(uint32_t), words_g4.data(), 0, nullptr, nullptr);
            if (!check_status(st, "clEnqueueReadBuffer(g4_check)")) return 4;
        }

        bool ok = true;
        for (size_t i = 0; i < kKnownCount; ++i) {
            uint8_t expected[32];
            uint8_t got[32];
            if (!parse_hex_32(kKnown[i].hex, expected)) {
                ok = false;
                break;
            }
            for (int j = 0; j < 8; ++j) {
                store_be(got + j * 4, words_g1[i * 8 + j]);
            }
            if (std::memcmp(got, expected, 32) != 0) {
                ok = false;
                break;
            }
            if (options.check_all) {
                for (int j = 0; j < 8; ++j) {
                    store_be(got + j * 4, words_g2[i * 8 + j]);
                }
                if (std::memcmp(got, expected, 32) != 0) {
                    ok = false;
                    break;
                }
                for (int j = 0; j < 8; ++j) {
                    store_be(got + j * 4, words_g3[i * 8 + j]);
                }
                if (std::memcmp(got, expected, 32) != 0) {
                    ok = false;
                    break;
                }
                for (int j = 0; j < 8; ++j) {
                    store_be(got + j * 4, words_g4[i * 8 + j]);
                }
                if (std::memcmp(got, expected, 32) != 0) {
                    ok = false;
                    break;
                }
            }
        }
        clReleaseMemObject(d_nonces);
        clReleaseMemObject(d_words_g1);
        clReleaseMemObject(d_words_g2);
        clReleaseMemObject(d_words_g3);
        clReleaseMemObject(d_words_g4);
        if (!ok) {
            std::fprintf(stderr, "Correctness check failed for OpenCL G1%s\n",
                         options.check_all ? "/G2/G3/G4" : "");
            return 4;
        }
    }

    cl_mem d_out = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
                                  static_cast<size_t>(options.iter) * sizeof(uint32_t), nullptr, &st);
    if (!check_status(st, "clCreateBuffer(out)")) return 5;

    const uint32_t nonce_base = 0;
    const uint32_t iter_u32 = static_cast<uint32_t>(options.iter);
    st = clSetKernelArg(kernel_g1_bench, 0, sizeof(cl_mem), &d_header);
    st |= clSetKernelArg(kernel_g1_bench, 1, sizeof(uint32_t), &nonce_base);
    st |= clSetKernelArg(kernel_g1_bench, 2, sizeof(uint32_t), &iter_u32);
    st |= clSetKernelArg(kernel_g1_bench, 3, sizeof(cl_mem), &d_out);
    if (!check_status(st, "clSetKernelArg(g1_bench)")) return 5;

    st = clSetKernelArg(kernel_g2_bench, 0, sizeof(cl_mem), &d_header);
    st |= clSetKernelArg(kernel_g2_bench, 1, sizeof(cl_mem), &d_midstate);
    st |= clSetKernelArg(kernel_g2_bench, 2, sizeof(uint32_t), &nonce_base);
    st |= clSetKernelArg(kernel_g2_bench, 3, sizeof(uint32_t), &iter_u32);
    st |= clSetKernelArg(kernel_g2_bench, 4, sizeof(cl_mem), &d_out);
    if (!check_status(st, "clSetKernelArg(g2_bench)")) return 5;

    st = clSetKernelArg(kernel_g3_bench, 0, sizeof(cl_mem), &d_header);
    st |= clSetKernelArg(kernel_g3_bench, 1, sizeof(cl_mem), &d_midstate);
    st |= clSetKernelArg(kernel_g3_bench, 2, sizeof(uint32_t), &nonce_base);
    st |= clSetKernelArg(kernel_g3_bench, 3, sizeof(uint32_t), &iter_u32);
    st |= clSetKernelArg(kernel_g3_bench, 4, sizeof(cl_mem), &d_out);
    if (!check_status(st, "clSetKernelArg(g3_bench)")) return 5;

    st = clSetKernelArg(kernel_g4_bench, 0, sizeof(cl_mem), &d_header);
    st |= clSetKernelArg(kernel_g4_bench, 1, sizeof(cl_mem), &d_midstate);
    st |= clSetKernelArg(kernel_g4_bench, 2, sizeof(uint32_t), &nonce_base);
    st |= clSetKernelArg(kernel_g4_bench, 3, sizeof(uint32_t), &iter_u32);
    st |= clSetKernelArg(kernel_g4_bench, 4, sizeof(cl_mem), &d_out);
    if (!check_status(st, "clSetKernelArg(g4_bench)")) return 5;

    const size_t local = 256;
    const size_t global = ((static_cast<size_t>(options.iter) + local - 1) / local) * local;

    auto t0 = std::chrono::high_resolution_clock::now();
    st = clEnqueueNDRangeKernel(queue, kernel_g1_bench, 1, nullptr, &global, &local, 0, nullptr, nullptr);
    if (!check_status(st, "clEnqueueNDRangeKernel(g1_bench)")) return 5;
    st = clFinish(queue);
    if (!check_status(st, "clFinish(g1_bench)")) return 5;
    auto t1 = std::chrono::high_resolution_clock::now();

    auto t2 = std::chrono::high_resolution_clock::now();
    st = clEnqueueNDRangeKernel(queue, kernel_g2_bench, 1, nullptr, &global, &local, 0, nullptr, nullptr);
    if (!check_status(st, "clEnqueueNDRangeKernel(g2_bench)")) return 5;
    st = clFinish(queue);
    if (!check_status(st, "clFinish(g2_bench)")) return 5;
    auto t3 = std::chrono::high_resolution_clock::now();

    auto t4 = std::chrono::high_resolution_clock::now();
    st = clEnqueueNDRangeKernel(queue, kernel_g3_bench, 1, nullptr, &global, &local, 0, nullptr, nullptr);
    if (!check_status(st, "clEnqueueNDRangeKernel(g3_bench)")) return 5;
    st = clFinish(queue);
    if (!check_status(st, "clFinish(g3_bench)")) return 5;
    auto t5 = std::chrono::high_resolution_clock::now();

    const size_t iter_g4 = (static_cast<size_t>(options.iter) + 3u) / 4u;
    const size_t global_g4 = ((iter_g4 + local - 1) / local) * local;
    auto t6 = std::chrono::high_resolution_clock::now();
    st = clEnqueueNDRangeKernel(queue, kernel_g4_bench, 1, nullptr, &global_g4, &local, 0, nullptr, nullptr);
    if (!check_status(st, "clEnqueueNDRangeKernel(g4_bench)")) return 5;
    st = clFinish(queue);
    if (!check_status(st, "clFinish(g4_bench)")) return 5;
    auto t7 = std::chrono::high_resolution_clock::now();

    uint32_t sink = 0;
    st = clEnqueueReadBuffer(queue, d_out, CL_TRUE, 0, sizeof(uint32_t), &sink, 0, nullptr, nullptr);
    if (!check_status(st, "clEnqueueReadBuffer(out)")) return 5;
    (void)sink;

    const double seconds_g1 = std::chrono::duration<double>(t1 - t0).count();
    const double safe_seconds_g1 = seconds_g1 > 0.0 ? seconds_g1 : 1e-9;
    const double hs_g1 = static_cast<double>(options.iter) / safe_seconds_g1;
    const double ns_g1 = (safe_seconds_g1 * 1e9) / static_cast<double>(options.iter);

    const double seconds_g2 = std::chrono::duration<double>(t3 - t2).count();
    const double safe_seconds_g2 = seconds_g2 > 0.0 ? seconds_g2 : 1e-9;
    const double hs_g2 = static_cast<double>(options.iter) / safe_seconds_g2;
    const double ns_g2 = (safe_seconds_g2 * 1e9) / static_cast<double>(options.iter);

    const double seconds_g3 = std::chrono::duration<double>(t5 - t4).count();
    const double safe_seconds_g3 = seconds_g3 > 0.0 ? seconds_g3 : 1e-9;
    const double hs_g3 = static_cast<double>(options.iter) / safe_seconds_g3;
    const double ns_g3 = (safe_seconds_g3 * 1e9) / static_cast<double>(options.iter);

    const double seconds_g4 = std::chrono::duration<double>(t7 - t6).count();
    const double safe_seconds_g4 = seconds_g4 > 0.0 ? seconds_g4 : 1e-9;
    const double hs_g4 = static_cast<double>(options.iter) / safe_seconds_g4;
    const double ns_g4 = (safe_seconds_g4 * 1e9) / static_cast<double>(options.iter);

    std::printf("G1,%.6f,%.6f\n", hs_g1, ns_g1);
    std::printf("G2,%.6f,%.6f\n", hs_g2, ns_g2);
    std::printf("G3,%.6f,%.6f\n", hs_g3, ns_g3);
    std::printf("G4,%.6f,%.6f\n", hs_g4, ns_g4);

    clReleaseMemObject(d_out);
    clReleaseMemObject(d_midstate);
    clReleaseMemObject(d_header);
    clReleaseKernel(kernel_g4_check);
    clReleaseKernel(kernel_g3_check);
    clReleaseKernel(kernel_g2_check);
    clReleaseKernel(kernel_g1_check);
    clReleaseKernel(kernel_g4_bench);
    clReleaseKernel(kernel_g3_bench);
    clReleaseKernel(kernel_g2_bench);
    clReleaseKernel(kernel_g1_bench);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
    return 0;
}
