#ifndef HW_DEF_H
#define HW_DEF_H

#include<cstdint>
#include <cmath>
#include "./common/common.h"
#include "./common/deviceQuery.h"
__forceinline__ __device__ unsigned lane_id()
{
    unsigned ret; 
    asm volatile ("mov.u32 %0, %laneid;" : "=r"(ret));
    return ret;
}

__forceinline__ __device__ unsigned warp_id()
{
    // this is not equal to threadIdx.x / 32
    unsigned ret; 
    asm volatile ("mov.u32 %0, %warpid;" : "=r"(ret));
    return ret;
}


#endif
