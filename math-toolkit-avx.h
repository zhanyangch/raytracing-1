#ifndef __RAY_MATH_TOOLKIT_H
#define __RAY_MATH_TOOLKIT_H

#include <math.h>
#include <stdio.h>
#include <assert.h>
#include <immintrin.h>

__attribute__((always_inline)) static inline
void normalize(double *v)
{
    double temp[4] = {0, 0, 0, 0};
    register __m256d ymm0, ymm1, ymm2, ymm3;
    ymm0 = _mm256_loadu_pd (v); // ymm0 = {v0, v1, v2, 0}
    ymm1 = _mm256_mul_pd (ymm0, ymm0);
    ymm2 = _mm256_setzero_pd();
    ymm3 = _mm256_hadd_pd (ymm1, ymm2); // ymm3 = { v0^2+v1^2, 0, v2^2, 0}
    const __m128d valupper = _mm256_extractf128_pd(ymm3, 1); // valupper = { v2^2, 0}
    ymm2 = _mm256_castpd128_pd256 (valupper); // ymm2 = {v2^2, 0, X, X} X:undefined
    ymm3 = _mm256_add_pd (ymm3, ymm2); // ymm3 = {v0^2+v1^2+v2^2, 0, X, X}
    ymm3 = _mm256_sqrt_pd (ymm3); // ymm4 = {sqrt(v0^2+v1^2+v2^2), 0, X, X}
    _mm256_storeu_pd (temp, ymm3);
    ymm2 = _mm256_set_pd (*temp, *temp, *temp, *temp);
    ymm0 = _mm256_div_pd (ymm0, ymm2);
    _mm256_storeu_pd (v, ymm0);
}

__attribute__((always_inline)) static inline
double length(const double *v)
{
    double temp[4] = {0, 0, 0, 0};
    register __m256d ymm0, ymm1, ymm2, ymm3;
    ymm0 = _mm256_loadu_pd (v); // ymm0 = {v0, v1, v2, 0}
    ymm1 = _mm256_mul_pd (ymm0, ymm0);
    ymm2 = _mm256_setzero_pd();
    ymm3 = _mm256_hadd_pd (ymm1, ymm2); // ymm3 = { v0^2+v1^2, 0, v2^2, 0}
    const __m128d valupper = _mm256_extractf128_pd(ymm3, 1); // valupper = { v2^2, 0}
    ymm2 = _mm256_castpd128_pd256 (valupper); // ymm2 = {v2^2, 0, X, X} X:undefined
    ymm3 = _mm256_add_pd (ymm3, ymm2); // ymm3 = {v0^2+v1^2+v2^2, 0, X, X}
    ymm3 = _mm256_sqrt_pd (ymm3); // ymm4 = {sqrt(v0^2+v1^2+v2^2), 0, X, X}
    _mm256_storeu_pd (temp, ymm3);
    return *temp;
}

__attribute__((always_inline)) static inline
void add_vector(const double *a, const double *b, double *out)
{
    register __m256d ymm0, ymm1;
    ymm0 = _mm256_loadu_pd (a); // ymm0 = {a0, a1, a2, 0}
    ymm1 = _mm256_loadu_pd (b);	// ymm1 = {b0, b1, b2, 0}
    ymm0 = _mm256_add_pd (ymm0, ymm1); // ymm0 = {a0+b0, a1+b1, a2+b2, 0}
    _mm256_storeu_pd (out, ymm0); // temp = {a0+b0, a1+b1, a2+b2, 0}
}

__attribute__((always_inline)) static inline
void subtract_vector(const double *a, const double *b, double *out)
{
    register __m256d ymm0, ymm1;
    ymm0 = _mm256_loadu_pd (a); // ymm0 = {a0, a1, a2, 0}
    ymm1 = _mm256_loadu_pd (b);	// ymm1 = {b0, b1, b2, 0}
    ymm0 = _mm256_sub_pd (ymm0, ymm1); // ymm0 = {a0+b0, a1+b1, a2+b2, 0}
    _mm256_storeu_pd (out, ymm0); // temp = {a0+b0, a1+b1, a2+b2, 0}
}

__attribute__((always_inline)) static inline
void multiply_vectors(const double *a, const double *b, double *out)
{
    register __m256d ymm0, ymm1;
    ymm0 = _mm256_loadu_pd (a); // ymm0 = {a0, a1, a2, 0}
    ymm1 = _mm256_loadu_pd (b);	// ymm1 = {b0, b1, b2, 0}
    ymm0 = _mm256_mul_pd (ymm0, ymm1);
    _mm256_storeu_pd (out, ymm0);
}

__attribute__((always_inline)) static inline
void multiply_vector(const double *a, double b, double *out)
{
    register __m256d ymm0, ymm1;
    ymm0 = _mm256_loadu_pd (a);
    ymm1 = _mm256_set_pd (b, b, b, b);
    ymm0 = _mm256_mul_pd (ymm0, ymm1);
    _mm256_storeu_pd (out, ymm0);
}

__attribute__((always_inline)) static inline
void cross_product(const double *a, const double *b, double *out)
{
    register __m256d ymm0, ymm1, ymm2, ymm3;
    double temp = 0;
    ymm0 = _mm256_set_pd (*(a+1), *a, *(a+2), *(a+1)); // ymm0 = {a1, a2, a2, a0}
    ymm1 = _mm256_set_pd (*b, *(b+1), *(b+1), *(b+2)); // ymm1 = {b2, b1, b0, b2}
    ymm2 = _mm256_set_pd (temp, temp, *a, *(a+2)); // ymm2 = {a0, a1, 0, 0}
    ymm3 = _mm256_set_pd (temp, temp, *(b+2), *b); // ymm3 = {b1, b0, 0, 0}
    ymm0 = _mm256_mul_pd (ymm0, ymm1);
    ymm2 = _mm256_mul_pd (ymm2, ymm3);
    ymm3 = _mm256_hsub_pd (ymm0, ymm2);
    _mm256_storeu_pd (out, ymm3);
}

__attribute__((always_inline)) static inline
double dot_product(const double *a, const double *b)
{
    double temp[4] = {0, 0, 0, 0};
    register __m256d ymm0, ymm1, ymm2;
    ymm0 = _mm256_loadu_pd (a); // ymm0 = {a0, a1, a2, 0}
    ymm1 = _mm256_loadu_pd (b);	// ymm1 = {b0, b1, b2, 0}
    ymm2 = _mm256_setzero_pd(); // ymm2 = {0, 0, 0, 0}
    ymm0 = _mm256_mul_pd (ymm0, ymm1); // ymm0 = {a0*b0, a1*b1, a2*b2, 0}
    ymm0 = _mm256_hadd_pd (ymm0, ymm2); // ymm0 = {a0*b0+a1*b1, 0, a2*b2, 0}
    const __m128d valupper = _mm256_extractf128_pd(ymm0, 1); // valupper = {a2*b2, 0}
    ymm1 = _mm256_castpd128_pd256 (valupper); // ymm2 = {a2*b2, 0, X, X} X means undefined
    ymm0 = _mm256_add_pd (ymm0, ymm1); // ymm0 = {a0*b0+a1*b1+a2*b2, 0, X, X}
    _mm256_storeu_pd (temp, ymm0);
    return *temp;
}

__attribute__((always_inline)) static inline
void scalar_triple_product(const double *u, const double *v, const double *w,
                           double *out)
{
    cross_product(v, w, out);
    multiply_vectors(u, out, out);
}

__attribute__((always_inline)) static inline
double scalar_triple(const double *u, const double *v, const double *w)
{
    double tmp[3];
    cross_product(w, u, tmp);
    return dot_product(v, tmp);
}

#endif
