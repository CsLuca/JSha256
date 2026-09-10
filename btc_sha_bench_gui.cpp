#define NOMINMAX
#include <windows.h>
#include <intrin.h>

#include <array>
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <immintrin.h>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#if defined(__AVX2__) || defined(_M_AVX2)
#define HAVE_AVX2_INTRIN 1
#else
#define HAVE_AVX2_INTRIN 0
#endif

#if defined(__SHA__) || (defined(_MSC_VER) && defined(_M_X64))
#define HAVE_SHA_INTRIN 1
#else
#define HAVE_SHA_INTRIN 0
#endif

namespace bench {

static inline uint32_t rotr(uint32_t x, unsigned n) {
    return (x >> n) | (x << (32 - n));
}

static inline uint32_t Ch(uint32_t x, uint32_t y, uint32_t z) {
    return (x & y) ^ (~x & z);
}

static inline uint32_t Maj(uint32_t x, uint32_t y, uint32_t z) {
    return (x & y) ^ (x & z) ^ (y & z);
}

static inline uint32_t Sigma0(uint32_t x) {
    return rotr(x, 2) ^ rotr(x, 13) ^ rotr(x, 22);
}

static inline uint32_t Sigma1(uint32_t x) {
    return rotr(x, 6) ^ rotr(x, 11) ^ rotr(x, 25);
}

static inline uint32_t sigma0(uint32_t x) {
    return rotr(x, 7) ^ rotr(x, 18) ^ (x >> 3);
}

static inline uint32_t sigma1(uint32_t x) {
    return rotr(x, 17) ^ rotr(x, 19) ^ (x >> 10);
}

static constexpr uint32_t K[64] = {
    0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u,
    0x3956c25bu, 0x59f111f1u, 0x923f82a4u, 0xab1c5ed5u,
    0xd807aa98u, 0x12835b01u, 0x243185beu, 0x550c7dc3u,
    0x72be5d74u, 0x80deb1feu, 0x9bdc06a7u, 0xc19bf174u,
    0xe49b69c1u, 0xefbe4786u, 0x0fc19dc6u, 0x240ca1ccu,
    0x2de92c6fu, 0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau,
    0x983e5152u, 0xa831c66du, 0xb00327c8u, 0xbf597fc7u,
    0xc6e00bf3u, 0xd5a79147u, 0x06ca6351u, 0x14292967u,
    0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu, 0x53380d13u,
    0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u,
    0xa2bfe8a1u, 0xa81a664bu, 0xc24b8b70u, 0xc76c51a3u,
    0xd192e819u, 0xd6990624u, 0xf40e3585u, 0x106aa070u,
    0x19a4c116u, 0x1e376c08u, 0x2748774cu, 0x34b0bcb5u,
    0x391c0cb3u, 0x4ed8aa4au, 0x5b9cca4fu, 0x682e6ff3u,
    0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u,
    0x90befffau, 0xa4506cebu, 0xbef9a3f7u, 0xc67178f2u
};

static constexpr uint32_t H0[8] = {
    0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u, 0xa54ff53au,
    0x510e527fu, 0x9b05688cu, 0x1f83d9abu, 0x5be0cd19u
};

static inline uint32_t load_be(const uint8_t* p) {
    return (uint32_t(p[0]) << 24) |
           (uint32_t(p[1]) << 16) |
           (uint32_t(p[2]) << 8) |
            uint32_t(p[3]);
}

static inline void store_be(uint8_t* p, uint32_t x) {
    p[0] = static_cast<uint8_t>(x >> 24);
    p[1] = static_cast<uint8_t>(x >> 16);
    p[2] = static_cast<uint8_t>(x >> 8);
    p[3] = static_cast<uint8_t>(x);
}

struct Midstate {
    uint32_t h[8];
};

static void compress_generic(uint32_t state[8], const uint32_t W[64]) {
    uint32_t a = state[0], b = state[1], c = state[2], d = state[3];
    uint32_t e = state[4], f = state[5], g = state[6], h = state[7];
    for (int i = 0; i < 64; ++i) {
        const uint32_t T1 = h + Sigma1(e) + Ch(e, f, g) + K[i] + W[i];
        const uint32_t T2 = Sigma0(a) + Maj(a, b, c);
        h = g; g = f; f = e; e = d + T1;
        d = c; c = b; b = a; a = T1 + T2;
    }
    state[0] += a; state[1] += b; state[2] += c; state[3] += d;
    state[4] += e; state[5] += f; state[6] += g; state[7] += h;
}

#define SHA256_ROUND(a,b,c,d,e,f,g,h,w,k) \
    do { \
        const uint32_t T1 = (h) + Sigma1(e) + Ch(e,f,g) + (k) + (w); \
        const uint32_t T2 = Sigma0(a) + Maj(a,b,c); \
        (h) = (g); (g) = (f); (f) = (e); (e) = (d) + T1; \
        (d) = (c); (c) = (b); (b) = (a); (a) = T1 + T2; \
    } while (0)

static void compress_unrolled(uint32_t state[8], const uint32_t W[64]) {
    uint32_t a = state[0], b = state[1], c = state[2], d = state[3];
    uint32_t e = state[4], f = state[5], g = state[6], h = state[7];
    SHA256_ROUND(a,b,c,d,e,f,g,h,W[0],K[0]);   SHA256_ROUND(a,b,c,d,e,f,g,h,W[1],K[1]);
    SHA256_ROUND(a,b,c,d,e,f,g,h,W[2],K[2]);   SHA256_ROUND(a,b,c,d,e,f,g,h,W[3],K[3]);
    SHA256_ROUND(a,b,c,d,e,f,g,h,W[4],K[4]);   SHA256_ROUND(a,b,c,d,e,f,g,h,W[5],K[5]);
    SHA256_ROUND(a,b,c,d,e,f,g,h,W[6],K[6]);   SHA256_ROUND(a,b,c,d,e,f,g,h,W[7],K[7]);
    SHA256_ROUND(a,b,c,d,e,f,g,h,W[8],K[8]);   SHA256_ROUND(a,b,c,d,e,f,g,h,W[9],K[9]);
    SHA256_ROUND(a,b,c,d,e,f,g,h,W[10],K[10]); SHA256_ROUND(a,b,c,d,e,f,g,h,W[11],K[11]);
    SHA256_ROUND(a,b,c,d,e,f,g,h,W[12],K[12]); SHA256_ROUND(a,b,c,d,e,f,g,h,W[13],K[13]);
    SHA256_ROUND(a,b,c,d,e,f,g,h,W[14],K[14]); SHA256_ROUND(a,b,c,d,e,f,g,h,W[15],K[15]);
    SHA256_ROUND(a,b,c,d,e,f,g,h,W[16],K[16]); SHA256_ROUND(a,b,c,d,e,f,g,h,W[17],K[17]);
    SHA256_ROUND(a,b,c,d,e,f,g,h,W[18],K[18]); SHA256_ROUND(a,b,c,d,e,f,g,h,W[19],K[19]);
    SHA256_ROUND(a,b,c,d,e,f,g,h,W[20],K[20]); SHA256_ROUND(a,b,c,d,e,f,g,h,W[21],K[21]);
    SHA256_ROUND(a,b,c,d,e,f,g,h,W[22],K[22]); SHA256_ROUND(a,b,c,d,e,f,g,h,W[23],K[23]);
    SHA256_ROUND(a,b,c,d,e,f,g,h,W[24],K[24]); SHA256_ROUND(a,b,c,d,e,f,g,h,W[25],K[25]);
    SHA256_ROUND(a,b,c,d,e,f,g,h,W[26],K[26]); SHA256_ROUND(a,b,c,d,e,f,g,h,W[27],K[27]);
    SHA256_ROUND(a,b,c,d,e,f,g,h,W[28],K[28]); SHA256_ROUND(a,b,c,d,e,f,g,h,W[29],K[29]);
    SHA256_ROUND(a,b,c,d,e,f,g,h,W[30],K[30]); SHA256_ROUND(a,b,c,d,e,f,g,h,W[31],K[31]);
    SHA256_ROUND(a,b,c,d,e,f,g,h,W[32],K[32]); SHA256_ROUND(a,b,c,d,e,f,g,h,W[33],K[33]);
    SHA256_ROUND(a,b,c,d,e,f,g,h,W[34],K[34]); SHA256_ROUND(a,b,c,d,e,f,g,h,W[35],K[35]);
    SHA256_ROUND(a,b,c,d,e,f,g,h,W[36],K[36]); SHA256_ROUND(a,b,c,d,e,f,g,h,W[37],K[37]);
    SHA256_ROUND(a,b,c,d,e,f,g,h,W[38],K[38]); SHA256_ROUND(a,b,c,d,e,f,g,h,W[39],K[39]);
    SHA256_ROUND(a,b,c,d,e,f,g,h,W[40],K[40]); SHA256_ROUND(a,b,c,d,e,f,g,h,W[41],K[41]);
    SHA256_ROUND(a,b,c,d,e,f,g,h,W[42],K[42]); SHA256_ROUND(a,b,c,d,e,f,g,h,W[43],K[43]);
    SHA256_ROUND(a,b,c,d,e,f,g,h,W[44],K[44]); SHA256_ROUND(a,b,c,d,e,f,g,h,W[45],K[45]);
    SHA256_ROUND(a,b,c,d,e,f,g,h,W[46],K[46]); SHA256_ROUND(a,b,c,d,e,f,g,h,W[47],K[47]);
    SHA256_ROUND(a,b,c,d,e,f,g,h,W[48],K[48]); SHA256_ROUND(a,b,c,d,e,f,g,h,W[49],K[49]);
    SHA256_ROUND(a,b,c,d,e,f,g,h,W[50],K[50]); SHA256_ROUND(a,b,c,d,e,f,g,h,W[51],K[51]);
    SHA256_ROUND(a,b,c,d,e,f,g,h,W[52],K[52]); SHA256_ROUND(a,b,c,d,e,f,g,h,W[53],K[53]);
    SHA256_ROUND(a,b,c,d,e,f,g,h,W[54],K[54]); SHA256_ROUND(a,b,c,d,e,f,g,h,W[55],K[55]);
    SHA256_ROUND(a,b,c,d,e,f,g,h,W[56],K[56]); SHA256_ROUND(a,b,c,d,e,f,g,h,W[57],K[57]);
    SHA256_ROUND(a,b,c,d,e,f,g,h,W[58],K[58]); SHA256_ROUND(a,b,c,d,e,f,g,h,W[59],K[59]);
    SHA256_ROUND(a,b,c,d,e,f,g,h,W[60],K[60]); SHA256_ROUND(a,b,c,d,e,f,g,h,W[61],K[61]);
    SHA256_ROUND(a,b,c,d,e,f,g,h,W[62],K[62]); SHA256_ROUND(a,b,c,d,e,f,g,h,W[63],K[63]);
    state[0] += a; state[1] += b; state[2] += c; state[3] += d;
    state[4] += e; state[5] += f; state[6] += g; state[7] += h;
}
#undef SHA256_ROUND

static void make_schedule_from_block(const uint8_t block[64], uint32_t W[64]) {
    for (int i = 0; i < 16; ++i) {
        W[i] = load_be(block + i * 4);
    }
    for (int i = 16; i < 64; ++i) {
        W[i] = sigma1(W[i - 2]) + W[i - 7] + sigma0(W[i - 15]) + W[i - 16];
    }
}

static void make_second_block_schedule(const uint8_t header[80], uint32_t nonce, uint32_t W[64]) {
    W[0] = load_be(header + 64); // merkle_root[28..31]
    W[1] = load_be(header + 68); // timestamp
    W[2] = load_be(header + 72); // nBits
    W[3] = nonce;
    W[4] = 0x80000000u;
    for (int i = 5; i < 15; ++i) {
        W[i] = 0;
    }
    W[15] = 640;
    for (int i = 16; i < 64; ++i) {
        W[i] = sigma1(W[i - 2]) + W[i - 7] + sigma0(W[i - 15]) + W[i - 16];
    }
}

static Midstate make_midstate(const uint8_t header[80]) {
    Midstate m{};
    std::memcpy(m.h, H0, sizeof(H0));
    uint8_t block0[64]{};
    std::memcpy(block0, header, 64);
    uint32_t W[64]{};
    make_schedule_from_block(block0, W);
    compress_generic(m.h, W);
    return m;
}

static void sha256_32bytes_specialized(const uint8_t digest32[32], uint8_t out32[32], bool unroll) {
    uint32_t state[8]{};
    std::memcpy(state, H0, sizeof(H0));
    uint32_t W[64]{};
    for (int i = 0; i < 8; ++i) {
        W[i] = load_be(digest32 + i * 4);
    }
    W[8] = 0x80000000u;
    for (int i = 9; i < 15; ++i) {
        W[i] = 0;
    }
    W[15] = 256;
    for (int i = 16; i < 64; ++i) {
        W[i] = sigma1(W[i - 2]) + W[i - 7] + sigma0(W[i - 15]) + W[i - 16];
    }
    if (unroll) {
        compress_unrolled(state, W);
    } else {
        compress_generic(state, W);
    }
    for (int i = 0; i < 8; ++i) {
        store_be(out32 + i * 4, state[i]);
    }
}

enum class Version {
    V1 = 1,
    V2 = 2,
    V3 = 3,
    V4 = 4,
    V5 = 5,
    V6 = 6,
    V7 = 7,
    V8 = 8
};

static void hash_v1_to_v4(const uint8_t header[80], const Midstate& mid, uint32_t nonce, Version version, uint8_t out32[32]) {
    uint32_t state[8]{};
    std::memcpy(state, mid.h, sizeof(state));

    uint32_t W1[64]{};
    if (version == Version::V1) {
        uint8_t b1[64]{};
        std::memcpy(b1, header + 64, 16);
        store_be(b1 + 12, nonce);
        b1[16] = 0x80;
        b1[62] = 0x02;
        b1[63] = 0x80;
        make_schedule_from_block(b1, W1);
    } else {
        make_second_block_schedule(header, nonce, W1);
    }

    if (version == Version::V3 || version == Version::V4) {
        compress_unrolled(state, W1);
    } else {
        compress_generic(state, W1);
    }

    uint8_t digest1[32]{};
    for (int i = 0; i < 8; ++i) {
        store_be(digest1 + i * 4, state[i]);
    }

    if (version == Version::V4) {
        sha256_32bytes_specialized(digest1, out32, true);
        return;
    }

    sha256_32bytes_specialized(digest1, out32, false);
}

struct SimdInfo {
    bool sse2 = false;
    bool sse41 = false;
    bool avx = false;
    bool avx2 = false;
    bool avx512f = false;
    bool sha = false;
};

struct CpuInfo {
    std::string vendor;
    std::string brand;
    std::string arch;
    unsigned int logical_cores = 0;
};

static std::string trim_spaces(std::string s) {
    while (!s.empty() && (s.back() == ' ' || s.back() == '\0')) {
        s.pop_back();
    }
    size_t start = 0;
    while (start < s.size() && s[start] == ' ') {
        ++start;
    }
    return s.substr(start);
}

static CpuInfo detect_cpu_info() {
    CpuInfo out{};

    int r0[4]{};
    __cpuid(r0, 0);
    const int max_leaf = r0[0];

    char vendor[13]{};
    std::memcpy(vendor + 0, &r0[1], 4);
    std::memcpy(vendor + 4, &r0[3], 4);
    std::memcpy(vendor + 8, &r0[2], 4);
    vendor[12] = '\0';
    out.vendor = vendor;

    if (max_leaf >= 0x80000004) {
        int b2[4]{};
        int b3[4]{};
        int b4[4]{};
        __cpuid(b2, 0x80000002);
        __cpuid(b3, 0x80000003);
        __cpuid(b4, 0x80000004);
        char brand[49]{};
        std::memcpy(brand + 0, b2, sizeof(b2));
        std::memcpy(brand + 16, b3, sizeof(b3));
        std::memcpy(brand + 32, b4, sizeof(b4));
        brand[48] = '\0';
        out.brand = trim_spaces(std::string(brand));
    }

    SYSTEM_INFO si{};
    GetNativeSystemInfo(&si);
    switch (si.wProcessorArchitecture) {
        case PROCESSOR_ARCHITECTURE_AMD64:
            out.arch = "x64";
            break;
        case PROCESSOR_ARCHITECTURE_INTEL:
            out.arch = "x86";
            break;
        case PROCESSOR_ARCHITECTURE_ARM64:
            out.arch = "ARM64";
            break;
        default:
            out.arch = "Unknown";
            break;
    }
    out.logical_cores = si.dwNumberOfProcessors;

    return out;
}

static bool os_supports_avx_state() {
    int regs[4]{};
    __cpuid(regs, 1);
    const bool xsave = (regs[2] & (1 << 27)) != 0;
    const bool avx = (regs[2] & (1 << 28)) != 0;
    if (!xsave || !avx) {
        return false;
    }
    const unsigned long long xcr0 = _xgetbv(0);
    return (xcr0 & 0x6ull) == 0x6ull;
}

static bool os_supports_avx512_state() {
    const unsigned long long xcr0 = _xgetbv(0);
    return (xcr0 & 0xe6ull) == 0xe6ull;
}

static SimdInfo detect_simd() {
    SimdInfo info{};
    int r1[4]{};
    int r7[4]{};
    __cpuid(r1, 1);
    __cpuidex(r7, 7, 0);

    info.sse2 = (r1[3] & (1 << 26)) != 0;
    info.sse41 = (r1[2] & (1 << 19)) != 0;

    const bool avx_hw = (r1[2] & (1 << 28)) != 0;
    const bool avx_state = os_supports_avx_state();
    info.avx = avx_hw && avx_state;
    info.avx2 = info.avx && ((r7[1] & (1 << 5)) != 0);
    info.avx512f = info.avx && ((r7[1] & (1 << 16)) != 0) && os_supports_avx512_state();
    info.sha = (r7[1] & (1 << 29)) != 0;
    return info;
}

struct BenchmarkResult {
    std::string simd;
    std::string version;
    std::string backend;
    int lanes = 1;
    double hashes_per_sec = 0.0;
    double cycles_per_hash = 0.0;
};

struct TimerResult {
    double seconds = 0.0;
    unsigned long long cycles = 0;
};

template <typename F>
static TimerResult timed_run(F&& fn) {
    LARGE_INTEGER freq{};
    LARGE_INTEGER t0{};
    LARGE_INTEGER t1{};
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&t0);
    _mm_lfence();
    const unsigned long long c0 = __rdtsc();
    fn();
    _mm_lfence();
    const unsigned long long c1 = __rdtsc();
    QueryPerformanceCounter(&t1);

    TimerResult tr{};
    tr.seconds = static_cast<double>(t1.QuadPart - t0.QuadPart) / static_cast<double>(freq.QuadPart);
    tr.cycles = c1 - c0;
    return tr;
}

static const char* version_name(Version v) {
    switch (v) {
        case Version::V1: return "V1";
        case Version::V2: return "V2";
        case Version::V3: return "V3";
        case Version::V4: return "V4";
        case Version::V5: return "V5";
        case Version::V6: return "V6";
        case Version::V7: return "V7";
        case Version::V8: return "V8";
        default: return "?";
    }
}

#if HAVE_AVX2_INTRIN
static inline __m256i rotr256(__m256i x, int n) {
    return _mm256_or_si256(_mm256_srli_epi32(x, n), _mm256_slli_epi32(x, 32 - n));
}

static inline __m256i Ch256(__m256i x, __m256i y, __m256i z) {
    return _mm256_xor_si256(_mm256_and_si256(x, y), _mm256_andnot_si256(x, z));
}

static inline __m256i Maj256(__m256i x, __m256i y, __m256i z) {
    return _mm256_xor_si256(_mm256_xor_si256(_mm256_and_si256(x, y), _mm256_and_si256(x, z)), _mm256_and_si256(y, z));
}

static inline __m256i Sigma0256(__m256i x) {
    return _mm256_xor_si256(_mm256_xor_si256(rotr256(x, 2), rotr256(x, 13)), rotr256(x, 22));
}

static inline __m256i Sigma1256(__m256i x) {
    return _mm256_xor_si256(_mm256_xor_si256(rotr256(x, 6), rotr256(x, 11)), rotr256(x, 25));
}

static inline __m256i sigma0256(__m256i x) {
    return _mm256_xor_si256(_mm256_xor_si256(rotr256(x, 7), rotr256(x, 18)), _mm256_srli_epi32(x, 3));
}

static inline __m256i sigma1256(__m256i x) {
    return _mm256_xor_si256(_mm256_xor_si256(rotr256(x, 17), rotr256(x, 19)), _mm256_srli_epi32(x, 10));
}

static void hash_v5_avx2_batch8(const uint8_t header[80], const Midstate& mid, uint32_t nonce_base, uint8_t out_last_hash[32]) {
    __m256i W[64]{};
    W[0] = _mm256_set1_epi32(static_cast<int>(load_be(header + 64)));
    W[1] = _mm256_set1_epi32(static_cast<int>(load_be(header + 68)));
    W[2] = _mm256_set1_epi32(static_cast<int>(load_be(header + 72)));
    W[3] = _mm256_set_epi32(
        static_cast<int>(nonce_base + 7),
        static_cast<int>(nonce_base + 6),
        static_cast<int>(nonce_base + 5),
        static_cast<int>(nonce_base + 4),
        static_cast<int>(nonce_base + 3),
        static_cast<int>(nonce_base + 2),
        static_cast<int>(nonce_base + 1),
        static_cast<int>(nonce_base + 0));
    W[4] = _mm256_set1_epi32(static_cast<int>(0x80000000u));
    for (int i = 5; i < 15; ++i) {
        W[i] = _mm256_setzero_si256();
    }
    W[15] = _mm256_set1_epi32(640);

    for (int i = 16; i < 64; ++i) {
        W[i] = _mm256_add_epi32(
            _mm256_add_epi32(sigma1256(W[i - 2]), W[i - 7]),
            _mm256_add_epi32(sigma0256(W[i - 15]), W[i - 16]));
    }

    __m256i a = _mm256_set1_epi32(static_cast<int>(mid.h[0]));
    __m256i b = _mm256_set1_epi32(static_cast<int>(mid.h[1]));
    __m256i c = _mm256_set1_epi32(static_cast<int>(mid.h[2]));
    __m256i d = _mm256_set1_epi32(static_cast<int>(mid.h[3]));
    __m256i e = _mm256_set1_epi32(static_cast<int>(mid.h[4]));
    __m256i f = _mm256_set1_epi32(static_cast<int>(mid.h[5]));
    __m256i g = _mm256_set1_epi32(static_cast<int>(mid.h[6]));
    __m256i h = _mm256_set1_epi32(static_cast<int>(mid.h[7]));

    for (int i = 0; i < 64; ++i) {
        const __m256i kval = _mm256_set1_epi32(static_cast<int>(K[i]));
        const __m256i T1 = _mm256_add_epi32(
            _mm256_add_epi32(_mm256_add_epi32(h, Sigma1256(e)), Ch256(e, f, g)),
            _mm256_add_epi32(kval, W[i]));
        const __m256i T2 = _mm256_add_epi32(Sigma0256(a), Maj256(a, b, c));
        h = g;
        g = f;
        f = e;
        e = _mm256_add_epi32(d, T1);
        d = c;
        c = b;
        b = a;
        a = _mm256_add_epi32(T1, T2);
    }

    a = _mm256_add_epi32(a, _mm256_set1_epi32(static_cast<int>(mid.h[0])));
    b = _mm256_add_epi32(b, _mm256_set1_epi32(static_cast<int>(mid.h[1])));
    c = _mm256_add_epi32(c, _mm256_set1_epi32(static_cast<int>(mid.h[2])));
    d = _mm256_add_epi32(d, _mm256_set1_epi32(static_cast<int>(mid.h[3])));
    e = _mm256_add_epi32(e, _mm256_set1_epi32(static_cast<int>(mid.h[4])));
    f = _mm256_add_epi32(f, _mm256_set1_epi32(static_cast<int>(mid.h[5])));
    g = _mm256_add_epi32(g, _mm256_set1_epi32(static_cast<int>(mid.h[6])));
    h = _mm256_add_epi32(h, _mm256_set1_epi32(static_cast<int>(mid.h[7])));

    alignas(32) uint32_t va[8], vb[8], vc[8], vd[8], ve[8], vf[8], vg[8], vh[8];
    _mm256_store_si256(reinterpret_cast<__m256i*>(va), a);
    _mm256_store_si256(reinterpret_cast<__m256i*>(vb), b);
    _mm256_store_si256(reinterpret_cast<__m256i*>(vc), c);
    _mm256_store_si256(reinterpret_cast<__m256i*>(vd), d);
    _mm256_store_si256(reinterpret_cast<__m256i*>(ve), e);
    _mm256_store_si256(reinterpret_cast<__m256i*>(vf), f);
    _mm256_store_si256(reinterpret_cast<__m256i*>(vg), g);
    _mm256_store_si256(reinterpret_cast<__m256i*>(vh), h);

    uint8_t digest1[8][32]{};
    for (int lane = 0; lane < 8; ++lane) {
        store_be(digest1[lane] + 0, va[lane]);
        store_be(digest1[lane] + 4, vb[lane]);
        store_be(digest1[lane] + 8, vc[lane]);
        store_be(digest1[lane] + 12, vd[lane]);
        store_be(digest1[lane] + 16, ve[lane]);
        store_be(digest1[lane] + 20, vf[lane]);
        store_be(digest1[lane] + 24, vg[lane]);
        store_be(digest1[lane] + 28, vh[lane]);
        sha256_32bytes_specialized(digest1[lane], out_last_hash, true);
    }
}
#endif

#if HAVE_SHA_INTRIN
#if defined(__GNUC__) || defined(__clang__)
__attribute__((target("sha,ssse3,sse4.1")))
#endif
static void sha256_compress_block_shani(uint32_t state[8], const uint8_t block[64]) {
    const __m128i bswap = _mm_set_epi8(
        12, 13, 14, 15,
        8, 9, 10, 11,
        4, 5, 6, 7,
        0, 1, 2, 3);

    auto kvec = [](uint32_t a, uint32_t b, uint32_t c, uint32_t d) {
        return _mm_setr_epi32(static_cast<int>(a), static_cast<int>(b), static_cast<int>(c), static_cast<int>(d));
    };

    __m128i s0, s1, save0, save1, tmp;
    __m128i m0, m1, m2, m3;

    tmp = _mm_loadu_si128(reinterpret_cast<const __m128i*>(&state[0]));
    s1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(&state[4]));
    tmp = _mm_shuffle_epi32(tmp, 0xB1);
    s1 = _mm_shuffle_epi32(s1, 0x1B);
    s0 = _mm_alignr_epi8(tmp, s1, 8);
    s1 = _mm_blend_epi16(s1, tmp, 0xF0);
    save0 = s0;
    save1 = s1;

    m0 = _mm_shuffle_epi8(_mm_loadu_si128(reinterpret_cast<const __m128i*>(block + 0)), bswap);
    m1 = _mm_shuffle_epi8(_mm_loadu_si128(reinterpret_cast<const __m128i*>(block + 16)), bswap);
    m2 = _mm_shuffle_epi8(_mm_loadu_si128(reinterpret_cast<const __m128i*>(block + 32)), bswap);
    m3 = _mm_shuffle_epi8(_mm_loadu_si128(reinterpret_cast<const __m128i*>(block + 48)), bswap);

    auto rounds4 = [&](const __m128i& w, const __m128i& k) {
        __m128i t = _mm_add_epi32(w, k);
        s1 = _mm_sha256rnds2_epu32(s1, s0, t);
        t = _mm_shuffle_epi32(t, 0x0E);
        s0 = _mm_sha256rnds2_epu32(s0, s1, t);
    };

    rounds4(m0, kvec(0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u));
    rounds4(m1, kvec(0x3956c25bu, 0x59f111f1u, 0x923f82a4u, 0xab1c5ed5u));
    m0 = _mm_sha256msg1_epu32(m0, m1);

    rounds4(m2, kvec(0xd807aa98u, 0x12835b01u, 0x243185beu, 0x550c7dc3u));
    m1 = _mm_sha256msg1_epu32(m1, m2);

    rounds4(m3, kvec(0x72be5d74u, 0x80deb1feu, 0x9bdc06a7u, 0xc19bf174u));
    m2 = _mm_sha256msg1_epu32(m2, m3);

    m0 = _mm_add_epi32(m0, _mm_alignr_epi8(m3, m2, 4));
    m0 = _mm_sha256msg2_epu32(m0, m3);
    rounds4(m0, kvec(0xe49b69c1u, 0xefbe4786u, 0x0fc19dc6u, 0x240ca1ccu));

    m1 = _mm_add_epi32(m1, _mm_alignr_epi8(m0, m3, 4));
    m1 = _mm_sha256msg2_epu32(m1, m0);
    rounds4(m1, kvec(0x2de92c6fu, 0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau));

    m2 = _mm_add_epi32(m2, _mm_alignr_epi8(m1, m0, 4));
    m2 = _mm_sha256msg2_epu32(m2, m1);
    rounds4(m2, kvec(0x983e5152u, 0xa831c66du, 0xb00327c8u, 0xbf597fc7u));

    m3 = _mm_add_epi32(m3, _mm_alignr_epi8(m2, m1, 4));
    m3 = _mm_sha256msg2_epu32(m3, m2);
    rounds4(m3, kvec(0xc6e00bf3u, 0xd5a79147u, 0x06ca6351u, 0x14292967u));

    m0 = _mm_sha256msg1_epu32(m0, m1);
    m0 = _mm_add_epi32(m0, _mm_alignr_epi8(m3, m2, 4));
    m0 = _mm_sha256msg2_epu32(m0, m3);
    rounds4(m0, kvec(0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu, 0x53380d13u));

    m1 = _mm_sha256msg1_epu32(m1, m2);
    m1 = _mm_add_epi32(m1, _mm_alignr_epi8(m0, m3, 4));
    m1 = _mm_sha256msg2_epu32(m1, m0);
    rounds4(m1, kvec(0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u));

    m2 = _mm_sha256msg1_epu32(m2, m3);
    m2 = _mm_add_epi32(m2, _mm_alignr_epi8(m1, m0, 4));
    m2 = _mm_sha256msg2_epu32(m2, m1);
    rounds4(m2, kvec(0xa2bfe8a1u, 0xa81a664bu, 0xc24b8b70u, 0xc76c51a3u));

    m3 = _mm_sha256msg1_epu32(m3, m0);
    m3 = _mm_add_epi32(m3, _mm_alignr_epi8(m2, m1, 4));
    m3 = _mm_sha256msg2_epu32(m3, m2);
    rounds4(m3, kvec(0xd192e819u, 0xd6990624u, 0xf40e3585u, 0x106aa070u));

    m0 = _mm_sha256msg1_epu32(m0, m1);
    m0 = _mm_add_epi32(m0, _mm_alignr_epi8(m3, m2, 4));
    m0 = _mm_sha256msg2_epu32(m0, m3);
    rounds4(m0, kvec(0x19a4c116u, 0x1e376c08u, 0x2748774cu, 0x34b0bcb5u));

    m1 = _mm_sha256msg1_epu32(m1, m2);
    m1 = _mm_add_epi32(m1, _mm_alignr_epi8(m0, m3, 4));
    m1 = _mm_sha256msg2_epu32(m1, m0);
    rounds4(m1, kvec(0x391c0cb3u, 0x4ed8aa4au, 0x5b9cca4fu, 0x682e6ff3u));

    m2 = _mm_sha256msg1_epu32(m2, m3);
    m2 = _mm_add_epi32(m2, _mm_alignr_epi8(m1, m0, 4));
    m2 = _mm_sha256msg2_epu32(m2, m1);
    rounds4(m2, kvec(0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u));

    m3 = _mm_sha256msg1_epu32(m3, m0);
    m3 = _mm_add_epi32(m3, _mm_alignr_epi8(m2, m1, 4));
    m3 = _mm_sha256msg2_epu32(m3, m2);
    rounds4(m3, kvec(0x90befffau, 0xa4506cebu, 0xbef9a3f7u, 0xc67178f2u));

    s0 = _mm_add_epi32(s0, save0);
    s1 = _mm_add_epi32(s1, save1);

    tmp = _mm_shuffle_epi32(s0, 0x1B);
    s1 = _mm_shuffle_epi32(s1, 0xB1);
    s0 = _mm_blend_epi16(tmp, s1, 0xF0);
    s1 = _mm_alignr_epi8(s1, tmp, 8);

    _mm_storeu_si128(reinterpret_cast<__m128i*>(&state[0]), s0);
    _mm_storeu_si128(reinterpret_cast<__m128i*>(&state[4]), s1);
}

#if defined(__GNUC__) || defined(__clang__)
__attribute__((target("sha,ssse3,sse4.1")))
#endif
static void sha256_32bytes_shani(const uint8_t digest32[32], uint8_t out32[32]) {
    uint8_t block[64]{};
    std::memcpy(block, digest32, 32);
    block[32] = 0x80;
    block[62] = 0x01;
    block[63] = 0x00;

    uint32_t state[8];
    std::memcpy(state, H0, sizeof(H0));
    sha256_compress_block_shani(state, block);

    for (int i = 0; i < 8; ++i) {
        store_be(out32 + i * 4, state[i]);
    }
}

static void hash_v4_shani(const uint8_t header[80], const Midstate& mid, uint32_t nonce, uint8_t out32[32]) {
    uint32_t state[8]{};
    std::memcpy(state, mid.h, sizeof(state));

    uint32_t W1[64]{};
    make_second_block_schedule(header, nonce, W1);
    compress_unrolled(state, W1);

    uint8_t digest1[32]{};
    for (int i = 0; i < 8; ++i) {
        store_be(digest1 + i * 4, state[i]);
    }
    sha256_32bytes_shani(digest1, out32);
}

static void hash_v7_shani_full(const uint8_t header[80], const Midstate& mid, uint32_t nonce, uint8_t out32[32]) {
    uint32_t state[8]{};
    std::memcpy(state, mid.h, sizeof(state));

    uint8_t block1[64]{};
    std::memcpy(block1, header + 64, 16);
    store_be(block1 + 12, nonce);
    block1[16] = 0x80;
    block1[62] = 0x02;
    block1[63] = 0x80;

    sha256_compress_block_shani(state, block1);

    uint8_t digest1[32]{};
    for (int i = 0; i < 8; ++i) {
        store_be(digest1 + i * 4, state[i]);
    }
    sha256_32bytes_shani(digest1, out32);
}
#endif

static void init_example_header(uint8_t header[80]) {
    std::memset(header, 0, 80);
    header[0] = 0x01;
    for (int i = 0; i < 32; ++i) {
        header[4 + i] = static_cast<uint8_t>(i);
        header[36 + i] = static_cast<uint8_t>(0x20 + i);
    }
    store_be(header + 68, 0x65F00000u); // timestamp
    store_be(header + 72, 0x1d00ffffu); // nBits
    store_be(header + 76, 0u);          // nonce
}

static std::vector<BenchmarkResult> run_all_benchmarks() {
    std::vector<BenchmarkResult> out;
    const SimdInfo simd = detect_simd();

    struct SimdTier {
        const char* name;
        bool enabled;
        int lanes;
    };

    const std::vector<SimdTier> tiers = {
        {"SCALAR", true, 1},
        {"SSE2", simd.sse2, 4},
        {"SSE4.1", simd.sse41, 4},
        {"AVX", simd.avx, 8},
        {"AVX2", simd.avx2, 8},
        {"AVX512F", simd.avx512f, 16},
        {"SHA-NI", simd.sha, 4}
    };

    uint8_t header[80]{};
    init_example_header(header);
    const Midstate mid = make_midstate(header);

    constexpr uint32_t ITER = 120000;
    uint8_t sink[32]{};

    for (const auto& tier : tiers) {
        if (!tier.enabled) {
            continue;
        }

        const bool is_sha_tier = (std::strcmp(tier.name, "SHA-NI") == 0);

        for (Version v : {Version::V1, Version::V2, Version::V3, Version::V4}) {
            auto tr = timed_run([&]() {
                for (uint32_t i = 0; i < ITER; ++i) {
                    if (is_sha_tier && v == Version::V4) {
#if HAVE_SHA_INTRIN
                        hash_v4_shani(header, mid, i, sink);
#else
                        hash_v1_to_v4(header, mid, i, v, sink);
#endif
                    } else {
                        hash_v1_to_v4(header, mid, i, v, sink);
                    }
                }
            });
            BenchmarkResult br{};
            br.simd = tier.name;
            br.version = version_name(v);
            if (is_sha_tier && v == Version::V4) {
                br.backend = HAVE_SHA_INTRIN ? "sha-ni-real" : "sha-fallback";
            } else {
                br.backend = "scalar-core";
            }
            br.lanes = 1;
            br.hashes_per_sec = static_cast<double>(ITER) / tr.seconds;
            br.cycles_per_hash = static_cast<double>(tr.cycles) / static_cast<double>(ITER);
            out.push_back(br);
        }

        auto tr = timed_run([&]() {
            uint32_t nonce = 0;
#if HAVE_AVX2_INTRIN
            if (std::strcmp(tier.name, "AVX2") == 0) {
                for (uint32_t i = 0; i + 8 <= ITER; i += 8) {
                    hash_v5_avx2_batch8(header, mid, nonce, sink);
                    nonce += 8;
                }
                for (; nonce < ITER; ++nonce) {
                    if (is_sha_tier) {
#if HAVE_SHA_INTRIN
                        hash_v4_shani(header, mid, nonce, sink);
#else
                        hash_v1_to_v4(header, mid, nonce, Version::V4, sink);
#endif
                    } else {
                        hash_v1_to_v4(header, mid, nonce, Version::V4, sink);
                    }
                }
                return;
            }
#endif
            for (uint32_t i = 0; i < ITER; i += static_cast<uint32_t>(tier.lanes)) {
                for (int lane = 0; lane < tier.lanes; ++lane) {
                    const uint32_t n = nonce + static_cast<uint32_t>(lane);
                    if (is_sha_tier) {
#if HAVE_SHA_INTRIN
                        hash_v4_shani(header, mid, n, sink);
#else
                        hash_v1_to_v4(header, mid, n, Version::V4, sink);
#endif
                    } else {
                        hash_v1_to_v4(header, mid, n, Version::V4, sink);
                    }
                }
                nonce += static_cast<uint32_t>(tier.lanes);
            }
        });
        BenchmarkResult br{};
        br.simd = tier.name;
        br.version = version_name(Version::V5);
        br.backend =
#if HAVE_SHA_INTRIN
            is_sha_tier ? "sha-ni-real" :
#endif
#if HAVE_AVX2_INTRIN
            (std::strcmp(tier.name, "AVX2") == 0) ? "avx2-8lane-real" : "lane-model";
#else
            "lane-model";
#endif
        br.lanes = tier.lanes;
        br.hashes_per_sec = static_cast<double>(ITER) / tr.seconds;
        br.cycles_per_hash = static_cast<double>(tr.cycles) / static_cast<double>(ITER);
        out.push_back(br);
    }

    {
        const CpuInfo cpu = detect_cpu_info();
        const unsigned int threads = std::max(1u, std::min(cpu.logical_cores, 32u));
        auto tr = timed_run([&]() {
            std::vector<std::thread> workers;
            std::vector<uint8_t> local_sinks(threads, 0);
            workers.reserve(threads);

            const uint32_t chunk = ITER / threads;
            uint32_t start = 0;
            for (unsigned int t = 0; t < threads; ++t) {
                const uint32_t extra = (t < (ITER % threads)) ? 1u : 0u;
                const uint32_t begin = start;
                const uint32_t end = begin + chunk + extra;
                start = end;

                workers.emplace_back([&, t, begin, end]() {
                    uint8_t local[32]{};
                    for (uint32_t n = begin; n < end; ++n) {
                        hash_v1_to_v4(header, mid, n, Version::V4, local);
                    }
                    local_sinks[t] = local[0];
                });
            }

            for (auto& th : workers) {
                th.join();
            }
            for (uint8_t v : local_sinks) {
                sink[0] ^= v;
            }
        });

        BenchmarkResult br{};
        br.simd = "SCALAR";
        br.version = version_name(Version::V6);
        br.backend = "mt-v4";
        br.lanes = static_cast<int>(threads);
        br.hashes_per_sec = static_cast<double>(ITER) / tr.seconds;
        br.cycles_per_hash = static_cast<double>(tr.cycles) / static_cast<double>(ITER);
        out.push_back(br);
    }

#if HAVE_SHA_INTRIN
    if (simd.sha) {
        auto tr = timed_run([&]() {
            for (uint32_t n = 0; n < ITER; ++n) {
                hash_v7_shani_full(header, mid, n, sink);
            }
        });
        BenchmarkResult br{};
        br.simd = "SHA-NI";
        br.version = version_name(Version::V7);
        br.backend = "sha-ni-full";
        br.lanes = 1;
        br.hashes_per_sec = static_cast<double>(ITER) / tr.seconds;
        br.cycles_per_hash = static_cast<double>(tr.cycles) / static_cast<double>(ITER);
        out.push_back(br);
    }
#endif

#if HAVE_AVX2_INTRIN
    if (simd.avx2) {
        auto tr = timed_run([&]() {
            uint32_t nonce = 0;
            for (; nonce + 16 <= ITER; nonce += 16) {
                hash_v5_avx2_batch8(header, mid, nonce, sink);
                hash_v5_avx2_batch8(header, mid, nonce + 8, sink);
            }
            for (; nonce < ITER; ++nonce) {
                hash_v1_to_v4(header, mid, nonce, Version::V4, sink);
            }
        });
        BenchmarkResult br{};
        br.simd = "AVX2";
        br.version = version_name(Version::V8);
        br.backend = "avx2-2x8-pipeline";
        br.lanes = 16;
        br.hashes_per_sec = static_cast<double>(ITER) / tr.seconds;
        br.cycles_per_hash = static_cast<double>(tr.cycles) / static_cast<double>(ITER);
        out.push_back(br);
    }
#endif

    sink[0] ^= 1;
    return out;
}

static std::string format_cpu_features() {
    const SimdInfo simd = detect_simd();
    const CpuInfo cpu = detect_cpu_info();
    std::ostringstream oss;
    oss << "CPU info\r\n";
    oss << "- Vendor:   " << (cpu.vendor.empty() ? "n/a" : cpu.vendor) << "\r\n";
    oss << "- Model:    " << (cpu.brand.empty() ? "n/a" : cpu.brand) << "\r\n";
    oss << "- Arch:     " << cpu.arch << "\r\n";
    oss << "- Logical:  " << cpu.logical_cores << "\r\n\r\n";
    oss << "Detected SIMD support\r\n";
    oss << "- SSE2:    " << (simd.sse2 ? "yes" : "no") << "\r\n";
    oss << "- SSE4.1:  " << (simd.sse41 ? "yes" : "no") << "\r\n";
    oss << "- AVX:     " << (simd.avx ? "yes" : "no") << "\r\n";
    oss << "- AVX2:    " << (simd.avx2 ? "yes" : "no") << "\r\n";
    oss << "- AVX512F: " << (simd.avx512f ? "yes" : "no") << "\r\n";
    oss << "- SHA-NI:  " << (simd.sha ? "yes" : "no") << "\r\n\r\n";
    return oss.str();
}

static std::string format_results(const std::vector<BenchmarkResult>& rows) {
    std::vector<BenchmarkResult> sorted = rows;
    std::sort(sorted.begin(), sorted.end(), [](const BenchmarkResult& a, const BenchmarkResult& b) {
        return a.hashes_per_sec > b.hashes_per_sec;
    });

    std::ostringstream oss;
    oss << format_cpu_features();
    oss << std::left << std::setw(10) << "SIMD"
        << std::setw(6) << "Ver"
        << std::setw(17) << "Backend"
        << std::setw(8) << "Lanes"
        << std::setw(16) << "Hash/s"
        << std::setw(14) << "Cycles/hash"
        << "\r\n";
    oss << "-----------------------------------------------------------------------\r\n";
    for (const auto& r : sorted) {
        oss << std::left << std::setw(10) << r.simd
            << std::setw(6) << r.version
            << std::setw(17) << r.backend
            << std::setw(8) << r.lanes
            << std::setw(16) << std::fixed << std::setprecision(2) << r.hashes_per_sec
            << std::setw(14) << std::fixed << std::setprecision(2) << r.cycles_per_hash
            << "\r\n";
    }
    return oss.str();
}

} // namespace bench

static HWND g_output = nullptr;

static void set_output_text(const std::string& s) {
    SetWindowTextA(g_output, s.c_str());
}

static LRESULT CALLBACK WndProc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam) {
    switch (msg) {
        case WM_CREATE: {
            CreateWindowExA(
                0,
                "BUTTON",
                "Run Benchmark V1..V8",
                WS_CHILD | WS_VISIBLE | BS_PUSHBUTTON,
                12,
                12,
                220,
                32,
                hwnd,
                reinterpret_cast<HMENU>(1001),
                GetModuleHandle(nullptr),
                nullptr);

            g_output = CreateWindowExA(
                WS_EX_CLIENTEDGE,
                "EDIT",
                "Click 'Run Benchmark V1..V8' to start.",
                WS_CHILD | WS_VISIBLE | WS_VSCROLL | ES_LEFT | ES_MULTILINE | ES_AUTOVSCROLL | ES_READONLY,
                12,
                56,
                960,
                560,
                hwnd,
                reinterpret_cast<HMENU>(1002),
                GetModuleHandle(nullptr),
                nullptr);
            return 0;
        }
        case WM_COMMAND: {
            if (LOWORD(wParam) == 1001) {
                set_output_text("Running benchmark...\r\n");
                auto results = bench::run_all_benchmarks();
                auto text = bench::format_results(results);
                set_output_text(text);
            }
            return 0;
        }
        case WM_SIZE: {
            RECT rc{};
            GetClientRect(hwnd, &rc);
            if (g_output) {
                MoveWindow(g_output, 12, 56, rc.right - 24, rc.bottom - 68, TRUE);
            }
            return 0;
        }
        case WM_DESTROY:
            PostQuitMessage(0);
            return 0;
        default:
            return DefWindowProc(hwnd, msg, wParam, lParam);
    }
}

int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE, LPSTR, int nCmdShow) {
    const char* kClassName = "BtcShaBenchGuiWnd";

    WNDCLASSA wc{};
    wc.lpfnWndProc = WndProc;
    wc.hInstance = hInstance;
    wc.lpszClassName = kClassName;
    wc.hCursor = LoadCursor(nullptr, IDC_ARROW);
    wc.hbrBackground = reinterpret_cast<HBRUSH>(COLOR_WINDOW + 1);

    RegisterClassA(&wc);

    HWND hwnd = CreateWindowExA(
        0,
        kClassName,
        "Bitcoin SHA256 Benchmark V1..V5 + SIMD Scoreboard",
        WS_OVERLAPPEDWINDOW,
        CW_USEDEFAULT,
        CW_USEDEFAULT,
        1024,
        720,
        nullptr,
        nullptr,
        hInstance,
        nullptr);

    if (!hwnd) {
        return 1;
    }

    ShowWindow(hwnd, nCmdShow);
    UpdateWindow(hwnd);

    MSG msg{};
    while (GetMessage(&msg, nullptr, 0, 0)) {
        TranslateMessage(&msg);
        DispatchMessage(&msg);
    }

    return 0;
}
