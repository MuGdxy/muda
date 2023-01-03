#pragma once
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>

#ifdef __INTELLISENSE__
void         __syncthreads();
int          atomicAdd(int* address, int val);
unsigned int atomicAdd(unsigned int* address, unsigned int val);
unsigned long long int atomicAdd(unsigned long long int* address, unsigned long long int val);
float   atomicAdd(float* address, float val);
double  atomicAdd(double* address, double val);
__half2 atomicAdd(__half2* address, __half2 val);
__half  atomicAdd(__half* address, __half val);
//__nv_bfloat162 atomicAdd(__nv_bfloat162* address, __nv_bfloat162 val);
//__nv_bfloat16 atomicAdd(__nv_bfloat16* address, __nv_bfloat16 val);
int          atomicSub(int* address, int val);
unsigned int atomicSub(unsigned int* address, unsigned int val);
int          atomicExch(int* address, int val);
unsigned int atomicExch(unsigned int* address, unsigned int val);
unsigned long long int atomicExch(unsigned long long int* address, unsigned long long int val);
float        atomicExch(float* address, float val);
int          atomicMin(int* address, int val);
unsigned int atomicMin(unsigned int* address, unsigned int val);
unsigned long long int atomicMin(unsigned long long int* address, unsigned long long int val);
long long int atomicMin(long long int* address, long long int val);
int           atomicMax(int* address, int val);
unsigned int  atomicMax(unsigned int* address, unsigned int val);
unsigned long long int atomicMax(unsigned long long int* address, unsigned long long int val);
long long int atomicMax(long long int* address, long long int val);
unsigned int  atomicInc(unsigned int* address, unsigned int val);
unsigned int  atomicDec(unsigned int* address, unsigned int val);
int           atomicCAS(int* address, int compare, int val);
unsigned int atomicCAS(unsigned int* address, unsigned int compare, unsigned int val);
unsigned long long int atomicCAS(unsigned long long int* address,
                                 unsigned long long int  compare,
                                 unsigned long long int  val);
unsigned short int     atomicCAS(unsigned short int* address,
                                 unsigned short int  compare,
                                 unsigned short int  val);
int                    atomicAnd(int* address, int val);
unsigned int           atomicAnd(unsigned int* address, unsigned int val);
unsigned long long int atomicAnd(unsigned long long int* address, unsigned long long int val);
int          atomicOr(int* address, int val);
unsigned int atomicOr(unsigned int* address, unsigned int val);
unsigned long long int atomicOr(unsigned long long int* address, unsigned long long int val);
int          atomicXor(int* address, int val);
unsigned int atomicXor(unsigned int* address, unsigned int val);
unsigned long long int atomicXor(unsigned long long int* address, unsigned long long int val);
#endif

#include <Eigen/Core>

template <typename T, int N>
inline __device__ Eigen::Vector<T, N> atomicAdd(Eigen::Vector<T, N>& v, const Eigen::Vector<T, N>& val) noexcept
{
#pragma unroll
    Eigen::Vector<T, N> ret;
    for(int i = 0; i < N; ++i)
        ret(i) = atomicAdd(&v(i), val(i));
    return ret;
}