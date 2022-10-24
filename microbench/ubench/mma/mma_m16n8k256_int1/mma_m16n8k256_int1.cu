#include <algorithm>
#include <cuda.h>
#include <iostream>
#include <mma.h>
#include <stdio.h>
#include <stdlib.h>
#include <cstdlib>
#include "../../../hw_def/hw_def.h"

// #define SHARED_MEM_SIZE (32 * 1024 / 4) // 32 KB
// Launch only one thread to calcaulte the latency using a pointer-chasing
// array technique
//#define THREADS_NUM 32
// iterate over the array ITERS times
#ifndef ITERS
#define ITERS  (1024 )
#endif


#ifndef ILPconfig
#define ILPconfig 1
#endif

static_assert(ILPconfig<=8, "ILP > 8 is not implemented\n");


__global__ void mma_ubench(uint64_t *startClk, uint64_t *stopClk, char *a, char *b, int *res,
          uint32_t strid) { // strid set to 0 used to prevent optimization
  // thread index
  uint32_t tid = threadIdx.x;
  uint32_t gid = blockIdx.x * blockDim.x + tid;
  uint32_t warpid = gid / warpSize;

  a = a + warpid * 16*32; // m*k = 16*32
  b = b + warpid * 8*32; // n*k = 8*32
  res = res + warpid * 16*8;// m*n = 16*8

   /** step 1: create register for each thread **/
  char frag_A[16*ILPconfig]; // four int8 registers, 
  char frag_B[8*ILPconfig];  // one .f16x2 registers, 2 half  elements
  int frag_D[4*ILPconfig]; //result(fp32) 2 f32 registers
  // fake load, we are focusing on mma latency/throughput. So no need to care about loading
  for(int i = 0;i<16*ILPconfig;i++){
    frag_A[i] = a[i + lane_id()*16]; 
    
  }
  for(int i =0;i<8*ILPconfig;i++){
    frag_B[i] = b[i + lane_id()*8]; 
    //frag_D[i] = 0.0f;
  }
  for(int i =0;i<4*ILPconfig;i++){
    //frag_B[i] = b[i + lane_id()*4]; 
    frag_D[i] = 0;
  }

  uint32_t const *A = reinterpret_cast<uint32_t const *>(&frag_A[0]);
  uint32_t const *B = reinterpret_cast<uint32_t const *>(&frag_B[0]);//?
  int *C = reinterpret_cast<int *>(&frag_D[0]);
  int *D = C;  // D = A*B + D. 


  uint64_t start = 0;
  uint64_t stop = 0;
  // synchronize all threads
  asm volatile("bar.sync 0;");
  // start timing
  asm volatile("mov.u64 %0, %%clock64;" : "=l"(start)::"memory");
  //#pragma unroll
  for (int j = 0; j < ITERS; ++j) {
    asm volatile(
        "mma.sync.aligned.m16n8k256.row.col.s32.b1.b1.s32.and.popc {%0,%1,%2,%3}, "
        "{%4,%5,%6,%7}, "
        "{%8,%9}, {%10,%11,%12,%13};\n"
        : "=r"(D[0]), "=r"(D[1]) , "=r"(D[2]), "=r"(D[3])
        : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), 
          "r"(B[0]), "r"(B[1]),
          "r"(C[0]), "r"(C[1]) ,"r"(C[2]), "r"(C[3])
    ); 

    #if ILPconfig >= 2
    asm volatile(
        "mma.sync.aligned.m16n8k256.row.col.s32.b1.b1.s32.and.popc {%0,%1,%2,%3}, "
        "{%4,%5,%6,%7}, "
        "{%8,%9}, {%10,%11,%12,%13};\n"
      : "=r"(D[4]), "=r"(D[5]) , "=r"(D[6]), "=r"(D[7])
      : "r"(A[4]), "r"(A[5]), "r"(A[6]), "r"(A[7]), 
        "r"(B[2]), "r"(B[3]),
        "r"(C[4]), "r"(C[5]) ,"r"(C[6]), "r"(C[7])
    ); 
    #endif

    #if ILPconfig >= 3
    asm volatile(
        "mma.sync.aligned.m16n8k256.row.col.s32.b1.b1.s32.and.popc {%0,%1,%2,%3}, "
        "{%4,%5,%6,%7}, "
        "{%8,%9}, {%10,%11,%12,%13};\n"
      : "=r"(D[8]), "=r"(D[9]) , "=r"(D[10]), "=r"(D[11])
      : "r"(A[8]), "r"(A[9]), "r"(A[10]), "r"(A[11]), 
        "r"(B[4]), "r"(B[5]),
        "r"(C[8]), "r"(C[9]) ,"r"(C[10]), "r"(C[11])
    ); 
    #endif
    #if ILPconfig >= 4
    asm volatile(
        "mma.sync.aligned.m16n8k256.row.col.s32.b1.b1.s32.and.popc {%0,%1,%2,%3}, "
        "{%4,%5,%6,%7}, "
        "{%8,%9}, {%10,%11,%12,%13};\n"
      : "=r"(D[12]), "=r"(D[13]) , "=r"(D[14]), "=r"(D[15])
      : "r"(A[12]), "r"(A[13]), "r"(A[14]), "r"(A[15]), 
        "r"(B[6]), "r"(B[7]),
        "r"(C[12]), "r"(C[13]) ,"r"(C[14]), "r"(C[15])
    ); 
    #endif

    #if ILPconfig >= 5
    asm volatile(
        "mma.sync.aligned.m16n8k256.row.col.s32.b1.b1.s32.and.popc {%0,%1,%2,%3}, "
        "{%4,%5,%6,%7}, "
        "{%8,%9}, {%10,%11,%12,%13};\n"
      : "=r"(D[16]), "=r"(D[17]) , "=r"(D[18]), "=r"(D[19])
      : "r"(A[16]), "r"(A[17]), "r"(A[18]), "r"(A[19]), 
        "r"(B[8]), "r"(B[9]),
        "r"(C[16]), "r"(C[17]) ,"r"(C[18]), "r"(C[19])
    ); 
    #endif

    #if ILPconfig >= 6
    asm volatile(
        "mma.sync.aligned.m16n8k256.row.col.s32.b1.b1.s32.and.popc {%0,%1,%2,%3}, "
        "{%4,%5,%6,%7}, "
        "{%8,%9}, {%10,%11,%12,%13};\n"
      : "=r"(D[20]), "=r"(D[21]) , "=r"(D[22]), "=r"(D[23])
      : "r"(A[20]), "r"(A[21]), "r"(A[22]), "r"(A[23]), 
        "r"(B[10]), "r"(B[11]),
        "r"(C[20]), "r"(C[21]) ,"r"(C[22]), "r"(C[23])
    ); 
    #endif
    #if ILPconfig >= 7
    asm volatile(
        "mma.sync.aligned.m16n8k256.row.col.s32.b1.b1.s32.and.popc {%0,%1,%2,%3}, "
        "{%4,%5,%6,%7}, "
        "{%8,%9}, {%10,%11,%12,%13};\n"
      : "=r"(D[24]), "=r"(D[25]) , "=r"(D[26]), "=r"(D[27])
      : "r"(A[24]), "r"(A[25]), "r"(A[26]), "r"(A[27]), 
        "r"(B[12]), "r"(B[13]),
        "r"(C[24]), "r"(C[25]) ,"r"(C[26]), "r"(C[27])
    ); 
    #endif
    #if ILPconfig >= 8
    asm volatile(
        "mma.sync.aligned.m16n8k256.row.col.s32.b1.b1.s32.and.popc {%0,%1,%2,%3}, "
        "{%4,%5,%6,%7}, "
        "{%8,%9}, {%10,%11,%12,%13};\n"
      : "=r"(D[28]), "=r"(D[29]) , "=r"(D[30]), "=r"(D[31])
      : "r"(A[28]), "r"(A[28]), "r"(A[30]), "r"(A[31]), 
        "r"(B[14]), "r"(B[15]),
        "r"(C[28]), "r"(C[29]) ,"r"(C[30]), "r"(C[31])
    ); 
    #endif

    __syncwarp();

  }

  // synchronize warps
 
  // stop timing
  asm volatile("mov.u64 %0, %%clock64;" : "=l"(stop)::"memory");//around 1 cycle overhead
  for(int i=0; i < 4*ILPconfig;i++){
    res[i] = frag_D[i]; 

  }

  //res[0] += fpuC;
  startClk[gid] = start;
  stopClk[gid] = stop;
}


template <class T, class R> 
float run(int THREADS_PER_BLOCK, bool report_fma_bw = false) {
    intilizeDeviceProp(0);
  
    int BLOCKS_NUM = 1;
    int TOTAL_THREADS = THREADS_PER_BLOCK * BLOCKS_NUM;
    int WARP_SIZE = 32;
  
    unsigned total_A_SIZE =
        16*32 * (TOTAL_THREADS / WARP_SIZE); // asume one 16x8 matrix per warp
    unsigned total_B_SIZE =
        8*32 * (TOTAL_THREADS / WARP_SIZE); // asume one 8*8 matrix per warp
    unsigned total_R_SIZE =
        16*8 * (TOTAL_THREADS / WARP_SIZE); // asume one 16x16 matrix per warp
  
    uint64_t *startClk = (uint64_t *)malloc(TOTAL_THREADS * sizeof(uint64_t));
    uint64_t *stopClk = (uint64_t *)malloc(TOTAL_THREADS * sizeof(uint64_t));
    T *data1 = (T *)malloc(total_A_SIZE * sizeof(T));
    T *data2 = (T *)malloc(total_B_SIZE * sizeof(T));
    R *res = (R *)malloc(total_R_SIZE * sizeof(R));
  
    uint64_t *startClk_g;
    uint64_t *stopClk_g;
    T *data1_g;
    T *data2_g;
    R *res_g;
  
    for (uint32_t i = 0; i < 16*32; i++) {
      data1[i] = (T)i;
    }
  
    for (uint32_t i = 0; i < 8*32; i++) {
      data2[i] = (T)i;
    }
  
    gpuErrchk(cudaMalloc(&startClk_g, TOTAL_THREADS * sizeof(uint64_t)));
    gpuErrchk(cudaMalloc(&stopClk_g, TOTAL_THREADS * sizeof(uint64_t)));
    gpuErrchk(cudaMalloc(&data1_g, total_A_SIZE * sizeof(T)));
    gpuErrchk(cudaMalloc(&data2_g, total_B_SIZE * sizeof(T)));
    gpuErrchk(cudaMalloc(&res_g, total_R_SIZE * sizeof(R)));
  
    gpuErrchk(cudaMemcpy(data1_g, data1, total_A_SIZE * sizeof(T),
                         cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(data2_g, data2, total_B_SIZE * sizeof(T),
                         cudaMemcpyHostToDevice));
  
    mma_ubench<<<BLOCKS_NUM, THREADS_PER_BLOCK>>>(
        startClk_g, stopClk_g, data1_g, data2_g, res_g, 0);
    gpuErrchk(cudaPeekAtLastError());
  
    gpuErrchk(cudaMemcpy(startClk, startClk_g, TOTAL_THREADS * sizeof(uint64_t),
                         cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(stopClk, stopClk_g, TOTAL_THREADS * sizeof(uint64_t),
                         cudaMemcpyDeviceToHost));
    gpuErrchk(
        cudaMemcpy(res, res_g, total_R_SIZE * sizeof(R), cudaMemcpyDeviceToHost));
  
    float mma_bw, fma_bw;
    uint64_t total_time =
        *std::max_element(&stopClk[0], &stopClk[TOTAL_THREADS]) -
        *std::min_element(&startClk[0], &startClk[TOTAL_THREADS]);

    float fpuFMA = (float)(ITERS * TOTAL_THREADS * 1 * 1 * 1 * 0 ) /
          ((float)total_time);  // max 64FMA/clk/SM on RTX3070Ti

    mma_bw = ((float)(ITERS * TOTAL_THREADS)) / (float)total_time;
    // hmma_bw = ((float)(REPEAT_TIMES * TOTAL_THREADS * SASS_hmma_per_PTX_wmma)) /
    //           (float)total_time;
    fma_bw = ((float)(ITERS * 16 * 8 * 256 * ILPconfig * //0 *
                      (TOTAL_THREADS / WARP_SIZE))) /
             (float)total_time;
  
    // std::cout << "wmma PTX issue bandwidth = " << wmma_bw << "(thread/clk/SM) \n";
    //std::cout << "mma issue bandwidth = " << mma_bw << "(thread/clk/SM)\n";
    std::cout << "mma.sync.aligned.m16n8k256.row.col.s32.b1.b1.s32.and.popc latency " << (float)total_time/(float)ITERS << " cycles\n";
    std::cout << "FMA tensor bandwidth = " << fma_bw + fpuFMA << "(FMA/clk/SM)\n";
  
    std::cout << "Total Clk number = " << total_time << "\n";
  
    if (report_fma_bw)
      return fma_bw;
    else
      return mma_bw;
}

int main() {
    intilizeDeviceProp(0);
    std::cout<<"***********************************"<<std::endl;
    std::cout << "mma.sync.aligned.m16n8k256.row.col.s32.b1.b1.s32.and.popc microbenchmark with ILP = " << ILPconfig << std::endl;
    for(int i = 1; i <= 32; i = i*2){
        std::cout << "Number of warps = "<< i <<std::endl;
        run<char, int>(32*i);
        std::cout << std::endl;
    }

    // std::cout << "Number of warps = "<< 1 <<std::endl;
    // tensor1688_max_flops<half, float>(32);
    return 0;
  }
  