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


static_assert(ILPconfig<=5," ILP>5 is not implemented\n");



__global__ void mma_ubench(uint64_t *startClk, uint64_t *stopClk, half *a, half *b, float *res,
          uint32_t strid) { // strid set to 0 used to prevent optimization
  // thread index
  uint32_t tid = threadIdx.x;
  uint32_t gid = blockIdx.x * blockDim.x + tid;
  uint32_t warpid = gid / warpSize;

  a = a + warpid * 16*16; // m*k = 16*16
  b = b + warpid * 8*16; // n*k = 8*16
  res = res + warpid * 16*8;// m*n = 16*16

   /** step 1: create register for each thread **/
  half frag_A[8*ILPconfig]; // two .f16x2 registers, 8 half elements, 
  half frag_B[4*ILPconfig];  // one .f16x2 registers, 4 half  elements
  float frag_D[4*ILPconfig]; //result(fp32) 4 f32 registers
  // fake load, we are focusing on mma latency/throughput. So no need to care about loading
  for(int i = 0;i<8 * ILPconfig;i++){
    frag_A[i] = a[i + lane_id()*8]; 
    
  }
  for(int i =0;i<4 * ILPconfig;i++){
    frag_B[i] = b[i + lane_id()*4]; 
    frag_D[i] = 0.0f;
  }

  uint32_t const *A = reinterpret_cast<uint32_t const *>(&frag_A[0]);
  uint32_t const *B = reinterpret_cast<uint32_t const *>(&frag_B[0]);//?
  float *C = reinterpret_cast<float *>(&frag_D[0]);
  float *D = C;  // D = A*B + D. 

  // float fpuA = frag_A[0];
  // float fpuB = frag_B[0];
  float fpuC = frag_D[0];

  // int intA = threadIdx.x;
  // int intB = threadIdx.x + 1;
  int intC = threadIdx.x + 2;

  uint64_t start = 0;
  uint64_t stop = 0;
  // synchronize all threads
  asm volatile("bar.sync 0;");
  // start timing
  asm volatile("mov.u64 %0, %%clock64;" : "=l"(start)::"memory");
  //#pragma unroll
  for (int j = 0; j < ITERS; ++j) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
        : "=f"(D[0]), "=f"(D[1]), "=f"(D[2]), "=f"(D[3])
        : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), 
          "r"(B[0]), "r"(B[1]),
          "f"(C[0]), "f"(C[1]), "f"(C[2]), "f"(C[3])
    );
    #if ILPconfig >= 2
    asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
      : "=f"(D[4]), "=f"(D[5]), "=f"(D[6]), "=f"(D[7])
      : "r"(A[4]), "r"(A[5]), "r"(A[6]), "r"(A[7]), 
        "r"(B[2]), "r"(B[3]),
        "f"(C[4]), "f"(C[5]), "f"(C[6]), "f"(C[7])
    );
    #endif
    #if ILPconfig >= 3
    asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
      : "=f"(D[8]), "=f"(D[9]), "=f"(D[10]), "=f"(D[11])
      : "r"(A[8]), "r"(A[9]), "r"(A[10]), "r"(A[11]), 
        "r"(B[4]), "r"(B[5]),
        "f"(C[8]), "f"(C[9]), "f"(C[10]), "f"(C[11])
    );
    #endif
    #if ILPconfig >= 4
    asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
      : "=f"(D[12]), "=f"(D[13]), "=f"(D[14]), "=f"(D[15])
      : "r"(A[12]), "r"(A[13]), "r"(A[14]), "r"(A[15]), 
        "r"(B[6]), "r"(B[7]),
        "f"(C[12]), "f"(C[13]), "f"(C[14]), "f"(C[15])
    );
    #endif
    #if ILPconfig >= 5
    asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
      : "=f"(D[16]), "=f"(D[17]), "=f"(D[18]), "=f"(D[19])
      : "r"(A[16]), "r"(A[17]), "r"(A[18]), "r"(A[19]), 
        "r"(B[8]), "r"(B[9]),
        "f"(C[16]), "f"(C[17]), "f"(C[18]), "f"(C[19])
    );
    #endif
    __syncwarp();
  }
  // stop timing
  asm volatile("mov.u64 %0, %%clock64;" : "=l"(stop)::"memory");
  for(int i=0; i < 4*ILPconfig;i++){
    res[i] += frag_D[i]; 

    res[i] += fpuC;
    res[i] += intC;
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
        16*16 * (TOTAL_THREADS / WARP_SIZE); // asume one 16x8 matrix per warp
    unsigned total_B_SIZE =
        8*16 * (TOTAL_THREADS / WARP_SIZE); // asume one 8*8 matrix per warp
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
  
    for (uint32_t i = 0; i < 16*8; i++) {
      data1[i] = (T)i;
    }
  
    for (uint32_t i = 0; i < 8*8; i++) {
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
    fma_bw = ((float)(ITERS * 16 * 8 * 16  * ILPconfig * //0 *
                      (TOTAL_THREADS / WARP_SIZE))) /
             (float)total_time;
  
    // std::cout << "wmma PTX issue bandwidth = " << wmma_bw << "(thread/clk/SM) \n";
    //std::cout << "mma issue bandwidth = " << mma_bw << "(thread/clk/SM)\n";
    std::cout << "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32  latency " << (float)total_time/(float)ITERS << " cycles\n";
    std::cout << "FMA tensor bandwidth = " << fma_bw + fpuFMA << "(FMA/clk/SM)\n";
  
    std::cout << "Total Clk number = " << total_time << "\n";
  
    if (report_fma_bw)
      return fma_bw;
    else
      return mma_bw;
}

int main() {
    std::vector<int> warps = {1,2,4,6,8,12,16,32};
    intilizeDeviceProp(0);
    std::cout<<"***********************************"<<std::endl;
    std::cout << "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 microbenchmark with ILP = " << ILPconfig << std::endl;
    for(auto& e:warps){
        std::cout << "Number of warps = "<< e <<std::endl;
        run<half, float>(32*e);
        std::cout << std::endl;
    }

    // std::cout << "Number of warps = "<< 1 <<std::endl;
    // tensor1688_max_flops<half, float>(32);
    return 0;
  }
  





    //fpuC = fpuC * fpuA;
    //asm volatile("fma.rn.f32 %0, %1, %2, %3 ;" : "=f"(fpuC) : "f"(fpuA), "f"(fpuB),"f"(fpuC)  );
    
    // asm volatile("{\t\n"
    // "fma.rn.f32 %0, %1, %2 , %0;\n\t"
    // "fma.rn.f32 %0, %1, %2 , %0;\n\t"
    // "fma.rn.f32 %0, %1, %2 , %0;\n\t"
    // "fma.rn.f32 %0, %1, %2 , %0;\n\t"
    // "}"
    // : "+f"(fpuC), "+f"(fpuA), "+f"(fpuB));

    
    // asm volatile("{\t\n"
    // "mul.rn.f32 %0, %1, %0;\n\t"
    // "mul.rn.f32 %0, %1, %0;\n\t"
    // "mul.rn.f32 %0, %1, %0;\n\t"
    // "mul.rn.f32 %0, %1, %0;\n\t"
    // "}"
    // : "+f"(fpuC), "+f"(fpuA));



    // asm volatile("{\t\n"
    // "mad.lo.s32 %0, %1, %2 , %0;\n\t"
    // "mad.lo.s32 %0, %1, %2 , %0;\n\t"
    // "mad.lo.s32 %0, %1, %2 , %0;\n\t"
    // "mad.lo.s32 %0, %1, %2 , %0;\n\t"
    // "}"
    // : "+r"(intC), "+r"(intA), "+r"(intB));
