#ifndef HW_DEF_H
#define HW_DEF_H

#include<cstdint>
#include <cmath>
#include "./common/common.h"
#include "./common/deviceQuery.h"

// note this is just fake meta data, used for performance microbenchmarking
void initialize_fake_metadata_2_4(uint32_t* metadata, int row_size, int col_size) {
    int range = 6;
    uint32_t FourToTwoMeta[6] = { 0x4, 0x8, 0x9, 0xc, 0xd, 0xe };
    for (int i = 0; i < row_size * col_size / 16; i++) { // 32 bit can represent 16 indexes , each index has 2 bit
        uint32_t result = 0x0;
        for (int n = 0; n < 32 / 4; ++n) {
            double rnd = double(std::rand()) / double(RAND_MAX);
            rnd = range * rnd;
            uint32_t meta = FourToTwoMeta[(int)rnd];

            result = (uint32_t)(result | ((uint32_t)(meta << (i * 4))));
        }
        metadata[i] = result;
    }
}

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
