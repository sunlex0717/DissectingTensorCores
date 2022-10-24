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

static_assert(ILPconfig<=6, "ILP>6 is not implemented\n");


__global__ void mmasp_16816(uint64_t *startClk, uint64_t *stopClk, half *a, half *b, uint32_t* meteE, half *res,
          uint32_t strid) { // strid set to 0 used to prevent optimization
  // thread index
  uint32_t tid = threadIdx.x;
  uint32_t gid = blockIdx.x * blockDim.x + tid;
  uint32_t warpid = gid / warpSize;

  a = a + warpid * 16*16; // m*k = 16*16
  b = b + warpid * 8*32; // n*k = 8*16
  res = res + warpid * 16*8;// m*n = 16*16 32*4

   /** step 1: create register for each thread **/
   half frag_A[8*ILPconfig]; // two b32 registrs, 4 half non-zero elements, 16 dense 
   half frag_B[8*ILPconfig]; // two f16x2 registers, 8 half dense elements
   half frag_D[4*ILPconfig]; //result(fp32) 4 f32 registers
   uint32_t frag_E[1*ILPconfig]; // A .b32 register containing 16 2-bit vectors to for indexing non-zero of A
  // fake load, we are focusing on mma latency/throughput. So no need to care about loading
  for(int i = 0;i<8*ILPconfig;i++){
    frag_A[i] = a[i ]; 
    frag_B[i] = b[i ]; 
  }
  for(int i =0;i<4*ILPconfig;i++){
    //frag_B[i] = b[i + lane_id()*4]; 
    frag_D[i] = 0.0f;
  }

  for(int i =0; i < 1*ILPconfig ; i ++){
    frag_E[i] = meteE[i];
  }

  //TODO: cast half to 
  uint32_t const *A = reinterpret_cast<uint32_t const *>(&frag_A[0]);
  uint32_t const *B = reinterpret_cast<uint32_t const *>(&frag_B[0]);//?
  uint32_t *C = reinterpret_cast<uint32_t *>(&frag_D[0]);
  uint32_t *D = C; 
  uint32_t const *E = reinterpret_cast<uint32_t const *>(&frag_E[0]); ;

//   float fpuA = frag_A[0];
//   float fpuB = frag_B[0];
//   float fpuC = frag_D[0];



  // int intA = threadIdx.x;
  // int intB = threadIdx.x + 1;
//   int intC = threadIdx.x + 2;

  uint64_t start = 0;
  uint64_t stop = 0;
  // synchronize all threads
  asm volatile("bar.sync 0;");
  // start timing
  asm volatile("mov.u64 %0, %%clock64;" : "=l"(start)::"memory");
  //#pragma unroll
  for (int j = 0; j < ITERS; ++j) {

    asm volatile(
        "mma.sp.sync.aligned.m16n8k32.row.col.f16.f16.f16.f16 {%0,%1}, "
        "{%2,%3,%4,%5}, {%6,%7,%8,%9}, {%10,%11}, %12, 0x0;\n"
        : "=r"(D[0]), "=r"(D[1])
        : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), 
          "r"(B[0]), "r"(B[1]), "r"(B[2]), "r"(B[3]),
          "r"(C[0]), "r"(C[1]),
          "r"(E[0])
    );
    #if ILPconfig >= 2
    asm volatile(
        "mma.sp.sync.aligned.m16n8k32.row.col.f16.f16.f16.f16 {%0,%1}, "
        "{%2,%3,%4,%5}, {%6,%7,%8,%9}, {%10,%11}, %12, 0x0;\n"
      : "=r"(D[2]), "=r"(D[3])
      : "r"(A[4]), "r"(A[5]), "r"(A[6]), "r"(A[7]), 
        "r"(B[4]), "r"(B[5]), "r"(B[6]), "r"(B[7]),
        "r"(C[2]), "r"(C[3]), 
        "r"(E[1])
    );
    #endif
    #if ILPconfig >= 3
    asm volatile(
        "mma.sp.sync.aligned.m16n8k32.row.col.f16.f16.f16.f16 {%0,%1}, "
        "{%2,%3,%4,%5}, {%6,%7,%8,%9}, {%10,%11}, %12, 0x0;\n"
      : "=r"(D[4]), "=r"(D[5])
      : "r"(A[8]), "r"(A[9]), "r"(A[10]), "r"(A[11]), 
        "r"(B[8]), "r"(B[9]), "r"(B[10]), "r"(B[11]),
        "r"(C[4]), "r"(C[5]), 
        "r"(E[2])
    );
    #endif

    #if ILPconfig >= 4
    asm volatile(
        "mma.sp.sync.aligned.m16n8k32.row.col.f16.f16.f16.f16 {%0,%1}, "
        "{%2,%3,%4,%5}, {%6,%7,%8,%9}, {%10,%11}, %12, 0x0;\n"
      : "=r"(D[6]), "=r"(D[7])
      : "r"(A[12]), "r"(A[13]), "r"(A[14]), "r"(A[15]), 
        "r"(B[12]), "r"(B[13]), "r"(B[14]), "r"(B[15]),
        "r"(C[6]), "r"(C[7]), 
        "r"(E[3])
    );
    #endif

    #if ILPconfig >= 5
    asm volatile(
        "mma.sp.sync.aligned.m16n8k32.row.col.f16.f16.f16.f16 {%0,%1}, "
        "{%2,%3,%4,%5}, {%6,%7,%8,%9}, {%10,%11}, %12, 0x0;\n"
      : "=r"(D[8]), "=r"(D[9])
      : "r"(A[16]), "r"(A[17]), "r"(A[18]), "r"(A[19]), 
        "r"(B[16]), "r"(B[17]), "r"(B[18]), "r"(B[19]),
        "r"(C[8]), "r"(C[9]), 
        "r"(E[4])
    );
    #endif
    #if ILPconfig >= 6
    asm volatile(
        "mma.sp.sync.aligned.m16n8k32.row.col.f16.f16.f16.f16 {%0,%1}, "
        "{%2,%3,%4,%5}, {%6,%7,%8,%9}, {%10,%11}, %12, 0x0;\n"
      : "=r"(D[10]), "=r"(D[11])
      : "r"(A[20]), "r"(A[21]), "r"(A[22]), "r"(A[23]), 
        "r"(B[20]), "r"(B[21]), "r"(B[22]), "r"(B[23]),
        "r"(C[10]), "r"(C[11]),
        "r"(E[5])
    );
    #endif
    __syncwarp();
  }
  // synchronize all threads
  //asm volatile("bar.sync 0;");
  // stop timing
  asm volatile("mov.u64 %0, %%clock64;" : "=l"(stop)::"memory");
  // avoid compiler optimization
  for(int i=0; i < 4*ILPconfig;i++){
    res[i] = frag_D[i]; 

    // res[i] += float(fpuC);
    // res[i] += intC;
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
    
    int nwarps = THREADS_PER_BLOCK/WARP_SIZE;

    int mma_m = 16;
    int mma_n = 8;
    int mma_k = 32;


    // T *data1 = (T *)malloc(mma_m*mma_k/2 * sizeof(T));
    // T *data2 = (T *)malloc(mma_n*mma_k * sizeof(T));
    // R *res = (R *)malloc(mma_m*mma_n * sizeof(R));
    // uint32_t *meta_e = (uint32_t *)malloc(mma_m*mma_k/16 *sizeof(uint32_t) );

    unsigned total_A_SIZE =
        mma_m*mma_k/2 * (TOTAL_THREADS / WARP_SIZE); // asume one 16x8 matrix per warp
    unsigned total_B_SIZE =
        mma_n*mma_k * (TOTAL_THREADS / WARP_SIZE); // asume one 8*8 matrix per warp
    unsigned total_R_SIZE =
        mma_m*mma_n * (TOTAL_THREADS / WARP_SIZE); // asume one 16x16 matrix per warp

    
    unsigned total_E_SIZE =
        mma_m*mma_k/16 * (TOTAL_THREADS / WARP_SIZE); // asume one 16x16 matrix per warp
  
    uint64_t *startClk = (uint64_t *)malloc(TOTAL_THREADS * sizeof(uint64_t));
    uint64_t *stopClk = (uint64_t *)malloc(TOTAL_THREADS * sizeof(uint64_t));
    T *data1 = (T *)malloc(total_A_SIZE * sizeof(T));
    T *data2 = (T *)malloc(total_B_SIZE * sizeof(T));
    R *res = (R *)malloc(total_R_SIZE * sizeof(R));

    uint32_t *meta_e = (uint32_t *)malloc(total_E_SIZE *sizeof(uint32_t) );
    //uint32_t *meta_p = meta_e;
    for(int i=0; i < nwarps; i++){

        initialize_fake_metadata_2_4(&meta_e[mma_m*mma_k/16 * i] ,mma_m,mma_k);

    }
    
  
    uint64_t *startClk_g;
    uint64_t *stopClk_g;
    T *data1_g;
    T *data2_g;
    R *res_g;
    uint32_t *meta_e_g;
  
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
    gpuErrchk(cudaMalloc(&meta_e_g, total_E_SIZE *sizeof(uint32_t)));

  
    gpuErrchk(cudaMemcpy(data1_g, data1, total_A_SIZE * sizeof(T),
                         cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(data2_g, data2, total_B_SIZE * sizeof(T),
                         cudaMemcpyHostToDevice));

    gpuErrchk(cudaMemcpy(meta_e_g, meta_e, total_E_SIZE * sizeof(uint32_t), cudaMemcpyHostToDevice));
  
    mmasp_16816<<<BLOCKS_NUM, THREADS_PER_BLOCK>>>(
        startClk_g, stopClk_g, data1_g, data2_g,meta_e_g, res_g, 0);
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

    float fpuFMA = (float)(ITERS * TOTAL_THREADS * 1 * 1 * 1 * 0) /
          ((float)total_time);

    mma_bw = ((float)(ITERS * TOTAL_THREADS)) / (float)total_time;
    // hmma_bw = ((float)(REPEAT_TIMES * TOTAL_THREADS * SASS_hmma_per_PTX_wmma)) /
    //           (float)total_time;
    fma_bw = ((float)(ITERS * mma_m * mma_n * mma_k * ILPconfig *
                      (TOTAL_THREADS / WARP_SIZE))) /
             (float)total_time;
  
    // std::cout << "wmma PTX issue bandwidth = " << wmma_bw << "(thread/clk/SM) \n";
    //std::cout << "mma issue bandwidth = " << mma_bw << "(thread/clk/SM)\n";
    std::cout << "mma.sp.sync.aligned.m16n8k32.row.col.f16.f16.f16.f16  latency " << (float)total_time/(float)ITERS << " cycles\n";
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
    std::cout << "mma.sp.sync.aligned.m16n8k32.row.col.f16.f16.f16.f16  microbenchmark with ILP = " << ILPconfig << std::endl;
    for(auto &e:warps){
        std::cout << "Number of warps = "<< e <<std::endl;
        run<half, half>(32*e);
        std::cout << std::endl;
    }

    // std::cout << "Number of warps = "<< 1 <<std::endl;
    // tensor1688_max_flops<half, float>(32);
    return 0;
  }
  