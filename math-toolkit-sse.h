#ifndef __RAY_MATH_TOOLKIT_SSE_H
#define __RAY_MATH_TOOLKIT_SSE_H


#include <emmintrin.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>

__attribute__((always_inline)) static inline
void normalize(double *v)
{

    __m128d I0 = _mm_loadu_pd(v);//v0 v1
    __m128d I1 = _mm_loadu_pd(v+1);//v1 v2
    __m128d I2 = _mm_mul_pd (I0,I0);//v0^2 v1^2
    __m128d I3 = _mm_mul_pd (I1,I1);//v1^2 v2^2
    __m128d I4 = _mm_add_pd (I2,I3);//v0^2+v1^2 v1^2+v2^2
    __m128d T0 = _mm_unpackhi_pd(I3,I3); // v2^2 v2^2
    __m128d I5 = _mm_add_pd(I4,T0);//v0^2+v1^2+v2^2
    __m128d I6 = _mm_unpacklo_pd(I5,I5);
    __m128d I7 =_mm_sqrt_pd(I6);

    I0=_mm_div_pd(I0,I7);
    I1=_mm_div_pd(I1,I7);
    _mm_storeu_pd(v,I0);
    _mm_storeu_pd(v+1,I1);
}
__attribute__((always_inline)) static inline
double length(const double *v)
{
    __m128d I0 = _mm_loadu_pd(v);//v0 v1
    __m128d I1 = _mm_loadu_pd(v+1);//v1 v2
    __m128d I2 = _mm_mul_pd (I0,I0);//v0^2 v1^2
    __m128d I3 = _mm_mul_pd (I1,I1);//v1^2 v2^2
    __m128d I4 = _mm_add_pd (I2,I3);//v0^2+v1^2 v1^2+v2^2
    __m128d T0 = _mm_unpackhi_pd(I3,I3); // v2^2 v2^2
    __m128d I5 = _mm_add_pd(I4,T0);//v0^2+v1^2+v2^2
    __m128d I6 = _mm_sqrt_pd(I5);
    return _mm_cvtsd_f64(I6);
}


__attribute__((always_inline)) static inline
void add_vector(const double *a, const double *b, double *out)
{
    __m128d I0 = _mm_loadu_pd(a);//a0 a1
    __m128d I1 = _mm_loadu_pd(a+1);//a1 a2
    __m128d I2 = _mm_loadu_pd(b);//b0 b1
    __m128d I3 = _mm_loadu_pd(b+1);//b1 b2
    __m128d T0 = _mm_add_pd(I0,I2);
    __m128d T1 = _mm_add_pd(I1,I3);
    _mm_storeu_pd(out,T0);
    _mm_storeu_pd(out+1,T1);
}
__attribute__((always_inline)) static inline
void subtract_vector(const double *a, const double *b, double *out)
{
    __m128d I0 = _mm_loadu_pd(a);//a0 a1
    __m128d I1 = _mm_loadu_pd(a+1);//a1 a2
    __m128d I2 = _mm_loadu_pd(b);//b0 b1
    __m128d I3 = _mm_loadu_pd(b+1);//b1 b2
    __m128d T0 = _mm_sub_pd(I0,I2);
    __m128d T1 = _mm_sub_pd(I1,I3);
    _mm_storeu_pd(out,T0);
    _mm_storeu_pd(out+1,T1);
}


__attribute__((always_inline)) static inline
void multiply_vectors(const double *a, const double *b, double *out)
{
    __m128d I0 = _mm_loadu_pd(a);//a0 a1
    __m128d I1 = _mm_loadu_pd(a+1);//a1 a2
    __m128d I2 = _mm_loadu_pd(b);//b0 b1
    __m128d I3 = _mm_loadu_pd(b+1);//b1 b2
    __m128d T0 = _mm_mul_pd(I0,I2);
    __m128d T1 = _mm_mul_pd(I1,I3);
    _mm_storeu_pd(out,T0);
    _mm_storeu_pd(out+1,T1);
}

static inline
void multiply_vector(const double *a, double b, double *out)
{
    __m128d I0 = _mm_loadu_pd(a);//a0 a1
    __m128d I1 = _mm_loadu_pd(a+1);//a1 a2
    __m128d I2 = _mm_load_pd1(&b);//b0 b1
    __m128d T0 = _mm_mul_pd(I0,I2);
    __m128d T1 = _mm_mul_pd(I1,I2);
    _mm_storeu_pd(out,T0);
    _mm_storeu_pd(out+1,T1);
}

__attribute__((always_inline)) static inline
void cross_product(const double *v1, const double *v2, double *out)
{
    __m128d I0 = _mm_loadu_pd(v1);//v1[0] v1[1]
    __m128d I1 = _mm_loadu_pd(v1+1);//v1[1] v1[2]
    __m128d I2 = _mm_loadu_pd(v2);//v2[0] v2[1]
    __m128d I3 = _mm_loadu_pd(v2+1);//v2[1] v2[2]

    __m128d I4 = _mm_shuffle_pd(I1,I1,1);//v1[2]
    __m128d I5 = _mm_shuffle_pd(I3,I3,1);//v2[2]

    __m128d T1 = _mm_mul_pd(I0,I3);
    __m128d T2 = _mm_mul_pd(I1,I2);
    __m128d T3 = _mm_sub_pd(T1,T2);// out[2] out[0]

    __m128d T4 = _mm_mul_pd(I4,I2);
    __m128d T5 = _mm_mul_pd(I5,I0);
    __m128d T6 = _mm_sub_pd(T4,T5);// out[1]

    __m128d T7 =_mm_shuffle_pd(T3,T6,1);//out[0] out[1]

    _mm_storeu_pd(out,T7);
    *(out+2)=_mm_cvtsd_f64(T3);
}

__attribute__((always_inline)) static inline
double dot_product(const double *v1, const double *v2)
{
    __m128d I0 = _mm_loadu_pd(v1);//v1[0] v1[1]
    __m128d I1 = _mm_loadu_pd(v1+1);//v1[1] v1[2]
    __m128d I2 = _mm_loadu_pd(v2);//v2[0] v2[1]
    __m128d I3 = _mm_loadu_pd(v2+1);//v2[1] v2[2]

    __m128d T1 = _mm_mul_pd(I0,I2);
    __m128d T2 = _mm_mul_pd(I1,I3);
    __m128d T3 = _mm_add_pd(T1,T2);

    __m128d T4 = _mm_unpackhi_pd(T2,T2);
    __m128d T5 = _mm_add_pd(T3,T4);
    return _mm_cvtsd_f64(T5);
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
