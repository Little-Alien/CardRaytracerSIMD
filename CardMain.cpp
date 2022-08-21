/**
 *  Special thanks to Paul Heckbert
 *
 *  http://tproger.ru/translations/business-card-raytracer
    http://fabiensanglard.net/rayTracing_back_of_business_card/
 */
// based on card-raytracer-opt version
// added: vectorization, multi-threading, simplified Random, some small algo optimisations
// options: -O3 -march=native -ffast-math
// for Clang (probably old versions only): -mllvm -align-all-nofallthru-blocks=5
// https://dendibakh.github.io/blog/2018/01/25/Code_alignment_options_in_llvm
// some gcc versions need -flax-vector-conversions
// gcc/openmp: -fopenmp
// clang/openmp: -fopenmp=libomp
// AVX512: -march=skylake-avx512
// Ryzen ver.3 (5XXX): -march=znver3

#if _MSC_VER // MSVC compiler
  #define _CRT_SECURE_NO_WARNINGS
#endif

#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <conio.h>
#include <math.h>
#include <time.h>
#include  <immintrin.h>
//#include <omp.h>

#if __clang__
    #define vExt_vector 0  // use vector extensions for main vector structure (clang only)
#else
    #define vExt_vector 0
#endif
#define dpps 0 // 1 - mm_dp_ps for dot product

#if _MSC_VER //  MSVC compiler
    #define vExt_proc 0 // use tracer4 proc with vector extensions - disable for default MSVC, enable for Clang/MSVC
    #define vExt_ClearUnused 1 // must be 1 for Clang/MSVC
#else
    #define vExt_proc 1
    #define vExt_ClearUnused 0
#endif

// main options
#define tracerProc tracer4
// tracer1 - scalar, tracer2 - scalar with 1D-array, tracer3 - autovec, tracer4 - vector ext.
#define mThread 1 // 1 - OpenMP multithreading, -fopemp must be in parameters
#define SimpleRand 1 // simplified Random()


#define WIDTH  512
#define HEIGHT 512
#define RayCnt 64 // lower value can be set for faster testing

const int TestCycles = 32;
const float MaxTestTime = 2.0f;

// "float" accuracy seems to be enough for graphics
typedef float fltype;
//typedef double fltype;
fltype const RayK = (fltype) 224 / RayCnt;

#if vExt_vector
    typedef fltype Vector __attribute__((ext_vector_type(4)));

    static const Vector ZERO_VECTOR = (Vector){0, 0, 0};
    static const Vector Z_ORTHO_VECTOR = (Vector){0, 0, 1};
    static const Vector CAMERA_ASPECT_VECTOR = (Vector){17, 16, 8};
    static const Vector CAMERA_DEST_VECTOR = (Vector){-6, -16, 0};
    static const Vector COLOR_CELL1_VECTOR = (Vector){3, 1, 1};
    static const Vector COLOR_CELL2_VECTOR = (Vector){3, 3, 3};
    static const Vector COLOR_DARK_GRAY_VECTOR = (Vector){13, 13, 13};
    static const Vector COLOR_SKY = (Vector){.7, .6, 1};
#else
    struct Vector {
        Vector() {
        }
        Vector(fltype a, fltype b, fltype c)
            : x(a), y(b), z(c) {
        }
        fltype x, y, z;
        friend Vector operator+(const Vector &l, const Vector &r);
        friend Vector operator*(const Vector &l, fltype r);
    };

    inline Vector operator+(const Vector &l, const Vector &r) {
        return Vector(l.x + r.x, l.y + r.y, l.z + r.z);
    }
    inline Vector operator*(const Vector &l, fltype r) {
        return Vector(l.x * r, l.y * r, l.z * r);
    }

    static const Vector ZERO_VECTOR = Vector(0, 0, 0);
    static const Vector Z_ORTHO_VECTOR = Vector(0, 0, 1);
    static const Vector CAMERA_ASPECT_VECTOR = Vector(17, 16, 8);
    static const Vector CAMERA_DEST_VECTOR = Vector(-6, -16, 0);

    static const Vector COLOR_CELL1_VECTOR = Vector(3, 1, 1);
    static const Vector COLOR_CELL2_VECTOR = Vector(3, 3, 3);
    static const Vector COLOR_DARK_GRAY_VECTOR = Vector(13, 13, 13);
    static const Vector COLOR_SKY = Vector(.7, .6, 1);
#endif

// was %
inline
//  fltype dot(const Vector &l, const Vector &r)
  fltype dot(Vector l, Vector r)
{
#if dpps
    fltype v;
    _mm_store_ss(&v, _mm_dp_ps((__m128)l, (__m128)r, 0x71)); // faster a bit
    return v;
#else
	return l.x * r.x + l.y * r.y + l.z * r.z;
//    Vector v = l * r;
//    return v.x + v.y + v.z;
#endif
}

// was ^
inline
  Vector cross(const Vector &a, const Vector &b)
//  Vector cross(Vector a, Vector b)
{
#if _MSC_VER
    return Vector(a.y * b.z - a.z * b.y,
                  a.z * b.x - a.x * b.z,
                  a.x * b.y - a.y * b.x);
#else
    return (Vector) { a.y* b.z - a.z * b.y,
                      a.z* b.x - a.x * b.z,
                      a.x* b.y - a.y * b.x };
#endif
}
// was !
inline
  Vector norm(const Vector &v)
//  Vector norm(Vector v)
{
    return v * (1 / sqrt(dot(v,v)));
}

// Position vectors:

int G[] =
      { 0x0003C712,  // 00111100011100010010
		0x00044814,  // 01000100100000010100
		0x00044818,  // 01000100100000011000
		0x0003CF94,  // 00111100111110010100
		0x00004892,  // 00000100100010010010
		0x00004891,  // 00000100100010010001
		0x00038710,  // 00111000011100010000
		0x00000010,  // 00000000000000010000
		0x00000010,  // 00000000000000010000
		};
int const SpCnt  = 49;
int const SpCntA = 64; // aligned

#if _MSC_VER
  #define pre_align __declspec(align(32))
  #define post_align
#else
  #define pre_align
  #define post_align __attribute__ ((aligned(32)))
#endif

pre_align fltype Gxa[SpCntA] post_align;//__attribute__ ((aligned(32)));
pre_align fltype Gza[SpCntA] post_align;//__attribute__ ((aligned(32)));

typedef unsigned char byte;

struct Color {
    byte b,g,r;
};

Color Dst[WIDTH][HEIGHT];

uint32_t g_seed = 16007;

static inline
fltype Random()
{
	return (fltype) rand() / RAND_MAX;
//	return (fltype) 0.0f;                               // 1.95
//    g_seed *= 16807;
//    return (fltype) g_seed / (float)0xFFFFFFFF;       // 2.45
}


static inline
void Random8(uint32_t* __restrict seed, float* __restrict res)
{
  for (int i = 0; i < 8; i++) {
    #if SimpleRand
        seed[i] = seed[i] * 16807;
        res[i] = (fltype)(seed[i]) / (fltype)0xFFFFFFFF;
    #else
        res[i] = Random();
    #endif
  }
}


// The intersection test for line [o,v].
// Return 2 if a hit was found (and also return distance t and bouncing ray n).
// Return 0 if no hit was found but ray goes upward
// Return 1 if no hit was found but ray goes downward

static
//#if __clang__
  inline
//#endif
int tracer1(const Vector &o, const Vector &d, fltype &t, Vector& n) {
	t = 1e9;
	int m = 0;

	fltype p = -o.z / d.z;
	if (.01f < p) {
		t = p;
		n = Z_ORTHO_VECTOR;
		m = 1;
	}
	//The world is encoded in G, with 10 lines and 20 columns
	for (int k = 19; k--;)
		for (int j = 9; j--;)
			if (G[j] & 1 << k) {
				Vector p = o + Vector{(float)-k, 0.0, (float)-j - 4};
				fltype b = dot(p, d);
				fltype c = dot(p, p) - 1;
				fltype q = b * b - c;
				//Does the ray hit the sphere ?
				if (q > 0) {
					//It does, compute the distance camera-sphere
					fltype s = -b - sqrt(q);
					if (s < t && s > .01f)
                    {
						// So far this is the minimum distance, save it. And also
                        // compute the bouncing ray vector into 'n'
						t = s;
						n = norm(p + d * t);
						m = 2;
					}
				}
			}

	return m;
}

// non-vectorized version with 1-D array
static inline
int tracer2(const Vector &o, const Vector &d, fltype &t, Vector& n) {
	t = 1e9;
	int m = 0;

	fltype p = -o.z / d.z;
	if (.01f < p) {
		t = p;
		n = Z_ORTHO_VECTOR;
		m = 1;
	}

	//The world is encoded in G, with 10 lines and 20 columns
    for (int j = 0; j < SpCnt; j++) {
        Vector p = o + Vector{Gxa[j], 0.0, Gza[j]};
        fltype b = dot(p, d);
        fltype c = dot(p, p) - 1;
        fltype q = b * b - c;
        //Does the ray hit the sphere ?
        if (q > 0) {
            //It does, compute the distance camera-sphere
            fltype s = -b - sqrt(q);
            if (s < t && s > .01f)
            {
                // So far this is the minimum distance, save it. And also
                // compute the bouncing ray vector into 'n'
                t = s;
                n = norm(p + d * t);
                m = 2;
            }
        }
    }

	return m;
}

// autovectorization with loops separating
static inline
int tracer3(const Vector &o, const Vector &d, fltype &t, Vector& n)
{
	t = 1e9;
	int m = 0;
    pre_align fltype qa[SpCntA] post_align;
    pre_align fltype ba[SpCntA] post_align;
    int rHitCnt = 0;
    //bool rHitCnt = false;

    // calculating koeffs for sphere hit test, vectorized loop

#if _MSC_VER // version for MSVC compiler
    fltype dx = d.x, dy = d.y, dz = d.z;
    for (int j = 0; j < SpCnt; j++) {
        fltype px = o.x + Gxa[j];
        fltype py = o.y;
        fltype pz = o.z + Gza[j];
        fltype b = px * dx + py * dy + pz * dz;
        fltype c = px * px + py * py + pz * pz - 1.0f;
        fltype q = b * b - c;
        qa[j] = q; ba[j] = b;
        if (q > 0.0f) rHitCnt++;
    }
#else
    //#pragma clang loop unroll(disable)
    //#pragma clang loop interleave_count(2)
    for (int j = 0; j < SpCnt; j++) {
        Vector p = o + (Vector) { Gxa[j], 0, Gza[j] };
		fltype b = dot(p, d);
		fltype c = dot(p, p) - 1.0f;
		fltype q = b * b - c;
        qa[j] = q; ba[j] = b;
        //counting spheres that we hit by this ray
        if (q > 0.0f) rHitCnt++;
        //rHitCnt = rHitCnt | (q > 0.0f);
    }
#endif
    int mj = -1;
    // hit something?
    if (rHitCnt > 0)  // > 0
        // checking spheres - non-vectorized loop (because of (s < t))
        for (int j = 0; j < SpCnt; j++)
            //Does the ray hit the sphere ?
            if (qa[j] > 0.0f) {
                //It does, compute the distance camera-sphere
                fltype s = -ba[j] - sqrt(qa[j]);
                // find the minimum distance
                if ((s > .01f) && (s < t)) {
                    t = s; mj = j;
                }
            }


    // compute the bouncing ray vector into 'n'
    // (can be done only for nearest sphere)
    if (mj != -1) {
        Vector p = o + Vector{Gxa[mj], 0.0f, Gza[mj]};
        n = norm(p + d * t);
        m = 2;
    }
    else {
        // we miss the spheres, check bottom plane
        fltype p = -o.z / d.z;
        if (.01 < p) {
            t = p;
            n = Z_ORTHO_VECTOR;
            m = 1;
        }
    }

	return m;
}


#if vExt_proc

#if __AVX512F__
    #define VSize 16
#else
  #if __AVX__
    #define VSize 8 // 4 or 16 can also be used, but 8 is optimal
  #else
    #define VSize 4
  #endif // __AVX__
#endif // __AVX512F__

#if __clang__
  typedef int int8 __attribute__ ((ext_vector_type(VSize)));
  typedef int int4 __attribute__ ((ext_vector_type(VSize / 2)));
  typedef float float8 __attribute__ ((ext_vector_type(VSize)));
  typedef float float4 __attribute__ ((ext_vector_type(VSize / 2)));
#else
  typedef int int8 __attribute__ ((vector_size(VSize * 4)));
  typedef int int4 __attribute__ ((vector_size(VSize * 2)));
  typedef float float8 __attribute__ ((vector_size(VSize * 4)));
  typedef float float4 __attribute__ ((vector_size(VSize * 2)));
#endif


static inline bool test_bits_any(const int8 a) {
#if __AVX512F__
    return _mm512_movepi32_mask((__m512i)a) != 0;
#else
  #if __AVX__
    #if (VSize == 8)
        return (_mm256_movemask_ps(_mm256_castsi256_ps((__m256i)a))) != 0;
    #endif
  #else
    return (_mm_movemask_ps((__m128)a)) != 0;
  #endif // __AVX__
#endif // __AVX512F__
}

static inline bool any_positive(const float8 a) {
#if __AVX512F__
    return _mm512_movepi32_mask((__m512i)a) != 0xFFFF;
#else
  #if ((__AVX__) && (VSize >= 8))
    #if (VSize == 16) // ugly code to test 16-float vectors with 8-float AVX - no acceleration, can be removed
        __m256 m1 = { a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7] };
        __m256 m2 = { a[8], a[9], a[10], a[11], a[12], a[13], a[14], a[15] };
        return (_mm256_movemask_ps(m1) != 0xFF) || (_mm256_movemask_ps(m2) != 0xFF);
    #else
        return (_mm256_movemask_ps(a)) != 0xFF;
    #endif
  #else
    return (_mm_movemask_ps(a)) != 0xF;
  #endif // __AVX__
#endif // __AVX512F__
}
static inline float8 sqrt8(const float8 a) {
#if __AVX512F__
    return _mm512_sqrt_ps(a);
#else
  #if ((__AVX__) && (VSize >= 8))
    #if (VSize == 16)
        __m256 m1 = { a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7] };
        __m256 m2 = { a[8], a[9], a[10], a[11], a[12], a[13], a[14], a[15] };
        m1 = _mm256_sqrt_ps(m1);
        m2 = _mm256_sqrt_ps(m2);
        float8 res = { m1[0], m1[1], m1[2], m1[3], m1[4], m1[5], m1[6], m1[7], m2[0], m2[1], m2[2], m2[3], m2[4], m2[5], m2[6], m2[7] };
        return res;
    #else
        return _mm256_sqrt_ps(a);
    #endif
  #else
    return _mm_sqrt_ps(a);
  #endif // __AVX__
#endif // __AVX512F__
}

static inline float8 blendf(const float8 a, const float8 b, const int8 mask)
{
//#if ((__AVX__) && (!__AVX512F__))
//    return _mm256_blendv_ps((__m256)b, (__m256)a, (__m256)mask);
//#else
    return (float8)((mask & (int8)a) | (~mask & (int8)b));
//#endif
}
static inline int8 blend(const int8 a, const int8 b, const int8 mask)
{
//#if ((__AVX__) && (!__AVX512F__))
//    return (int8)_mm256_blendv_ps((__m256)b, (__m256)a, (__m256)mask);
//#else
    return (mask & a) | (~mask & b);
//#endif
}

struct vector8 {
    float8 x, y, z;
};

inline float8 dot8(const vector8 l, const vector8 r) {
    return l.x * r.x + l.y * r.y + l.z * r.z;
}

// vector extensions
static inline
int tracer4(const Vector &o, const Vector &d, fltype &t, Vector& n)
{
	t = 1e6;
	int m = 0;

    float8 * Gx8 = (float8*)Gxa;
    float8 * Gz8 = (float8*)Gza;
    vector8 d8, p8;
    float8 ta;
    int8 ia, ja;
    for (int i = 0; i < VSize; i++) {
        d8.x[i] = d.x; d8.y[i] = d.y; d8.z[i] = d.z;
        p8.y[i] = o.y;
        ta[i] = t; ia[i] = -1; ja[i] = i;
    }

    // vectorized loop
    //#pragma clang loop unroll(disable)
    for (int j = 0; j < (SpCnt / VSize); j++) {
        p8.x = o.x + Gx8[j];
        p8.z = o.z + Gz8[j];
		float8 b = dot8(p8, d8);
		float8 c = dot8(p8, p8) - 1.0f;
		float8 q = (b * b - c);

        #if vExt_ClearUnused
        int8 m = q > 0.0f;
        if (test_bits_any(m)) {
        #else
        if (any_positive(q)) {
        #endif
            float8 sa = -b - sqrt8(q); // compute the distance camera-sphere
            #if vExt_ClearUnused
                sa = (float8)((int8)sa & m); // mask used numbers - seems to be not needed for native clang/gcc
            #endif
            // find the minimum distance
            int8 c2 = ((sa > .01f) && (sa < ta));
            ta = blendf(sa, ta, c2);
            ia = blend(ja, ia, c2);
        }
        ja += VSize;
    }

    int mj = -1;

    if (any_positive((float8)ia)) // any value not 0xFF (-1)

        for (int i = 0; i < VSize; i++) {
        //    if ((ia[i] >= 0) && (ta[i] < t)) {
            if (ta[i] < t) {
                t = ta[i]; mj = ia[i];
            }
        }

    // scalar loop - processing array "tail"
    for (int j = (SpCnt / VSize) * VSize; j < SpCnt; j++) {
        Vector p = o + (Vector) { Gxa[j], 0, Gza[j] };
		fltype b = dot(p, d);
		fltype c = dot(p, p) - 1;
		fltype q = (b * b - c);
        if (q > 0.0f) {
            fltype s = -b - sqrt(q);
            if ((s > .01f) && (s < t)) { t = s; mj = j; }
        }
    }

    // compute the bouncing ray vector into 'n'
    // (can be done only for nearest sphere)
    if (mj != -1) {
        Vector p = o + Vector{Gxa[mj], 0.0f, Gza[mj]};
        n = norm(p + d * t);
        m = 2;
    }
    else {
        // we miss the spheres, check bottom plane
        fltype p = -o.z / d.z;
        if (.01 < p) {
            t = p;
            n = Z_ORTHO_VECTOR;
            m = 1;
        }
    }

	return m;
}

#endif


// Sample the world and return the pixel color for
// a ray passing by point o (Origin) and d (Direction)

static //inline
Vector sampler(const Vector o, const Vector d, float rnd1, float rnd2) {
	fltype t;
	Vector n(ZERO_VECTOR);

	// Search for an intersection ray Vs World.
	int m = tracerProc(o, d, t, n);
	//return COLOR_SKY * pow(1.0f - d.z * m, 4);

	// m == 0 -> No sphere found and the ray goes upward: Generate a sky color
	if (!m) {
		return COLOR_SKY * pow(1.0f - d.z, 4);
	}
	// A sphere was maybe hit.
	Vector h = o + d * t; // h = intersection coordinate
	// 'l' = direction to light (with random delta for soft-shadows).
	Vector l = norm( Vector{9 + rnd1, 9 + rnd2, 16} + h * -1 );
	Vector r = n;

	// Calculate the lambertian factor
	fltype b = dot(l, n);
	// Calculate illumination factor (lambertian coefficient > 0 or in shadow)?
	if (b < 0 || tracerProc(h, l, t, n)) {
		b = 0;
	}

    // m == 1 -> No sphere was hit and the ray was going downward:
    // Generate a floor color
	if (m & 1) {
		return ((int) (ceil(h.x * .2) + ceil(h.y * .2)) & 1
			? COLOR_CELL1_VECTOR
			: COLOR_CELL2_VECTOR
		) * (b * .2 + .1);
	}

	// r = The half-vector
	r = d + r * (dot(r, d) * -2);
	// Calculate the color 'p' with diffuse and specular component
	fltype p = (b > 0)
		? pow(dot(l, r), 99)
		: 0;

	// m == 2 A sphere was hit. Cast an ray bouncing from the sphere surface.
	// (recursive sampler call)
    #if (!SimpleRand)
        rnd1 = Random(); rnd2 = Random();
	#endif
	// Attenuate color by 50% since it is bouncing (* .5)
	#if vExt_vector
        return Vector(p) + sampler(h, r, rnd2, rnd1) * .5;
	#else
        return Vector(p, p, p) + sampler(h, r, rnd2, rnd1) * .5;
	#endif
}


int main(int argc, char **argv)
{
    // converting spheres' position from bitmasks to 1D-arrays
    int i = 0;
    for (int j = 9; j--;)
        for (int k = 19; k--;)
            if (G[j] & 1 << k) {
                Gxa[i] = -k; Gza[i] = -j-4;
                i++;
            };

	Vector g = norm(CAMERA_DEST_VECTOR);
	Vector a = norm(cross(Z_ORTHO_VECTOR, g)) * .002;
	Vector b = norm(cross(g, a)) * .002;
	Vector c = (a + b) * -256 + g;

	float MinTime = 1e6;
	float SumTime = 0.0f;
	int RealCycles = 0;

    for (int n = 0; n < TestCycles; n++) {

        clock_t tStart = clock();

        // For each pixel
        #if mThread
            #pragma omp parallel for schedule(dynamic)
        #endif
        //for (int y = HEIGHT-1; y >= 0; y--) {
        for (int y = 0; y < HEIGHT; y++) {
            Color* pc = &Dst[HEIGHT-1-y][0];
            //Color* pc = &Dst[y][0];

            //for (int x = 0; x < WIDTH; x++) {
            for (int x = WIDTH; x--;) {
                Vector p(COLOR_DARK_GRAY_VECTOR);

                uint32_t RSeed[8];
                if (SimpleRand)
                    for (int i = 0; i < 8; i++) RSeed[i] = rand();

                // Cast 64 rays per pixel, for blur (stochastic sampling) and soft-shadows.
                for (int r = RayCnt; r--;) {
                    float Rnd[8];
                    Random8(RSeed, Rnd);
                    // A little bit of delta up/down and left/right
                    Vector t = a * (Rnd[0] - .5) * 99 + b * (Rnd[1] - .5) * 99;
                    // Ray Direction with random deltas for stochastic sampling
                    Vector d = norm(t * -1 + (a * (x + Rnd[2]) + b * (y + Rnd[3]) + c) * 16);
                    // Set the camera focal point v(17,16,8) and cast the ray
                    // Accumulate the color returned in the p variable
                    p = p + sampler(CAMERA_ASPECT_VECTOR + t, d, Rnd[4], Rnd[5]) * RayK;
                }
                *pc++ = {(byte) p.x, (byte) p.y, (byte) p.z};
            }
        }

        float cTime = (float)(clock() - tStart)/CLOCKS_PER_SEC;
        if (cTime < MinTime) MinTime = cTime;
        printf("Time taken: %.3fs\n", cTime);
        SumTime+= cTime; RealCycles++;
        if (SumTime > MaxTestTime) break;
    }

	printf("Min time: %.3fs\n", MinTime);


	FILE *out = fopen("dst.ppm", "w");
	fprintf(out, "P6 %d %d 255 ", WIDTH, HEIGHT);
/*
	// writing bmp file
	FILE *out = fopen("dst.bmp", "w");
	char hdr[]= {0x42 ,0x4D ,0x36 ,0x00 ,0x0C ,0x00 ,0x00 ,0x00 ,0x00 ,0x00 ,0x36 ,0x00 ,0x00 ,0x00 ,0x28 ,0x00
				,0x00 ,0x00 ,0x00 ,0x02 ,0x00 ,0x00 ,0x00 ,0x02 ,0x00 ,0x00 ,0x01 ,0x00 ,0x18 ,0x00 ,0x00 ,0x00
				,0x00 ,0x00 ,0x00 ,0x00 ,0x0C ,0x00 ,0x00 ,0x00 ,0x00 ,0x00 ,0x00 ,0x00 ,0x00 ,0x00 ,0x00 ,0x00
				,0x00 ,0x00 ,0x00 ,0x00 ,0x00 ,0x00};
	fwrite(hdr, sizeof(hdr), 1, out);
*/
	fwrite(Dst, sizeof(Dst), 1, out);
	fclose(out);
#if _MSC_VER
    _getch();
#else
    getch();
#endif
	return 0;
}
