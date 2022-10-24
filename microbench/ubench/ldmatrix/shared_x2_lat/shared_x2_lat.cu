#include <cuda.h>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>

#include "../../../hw_def/hw_def.h"

#define SHARED_MEM_SIZE (32 * 1024) // 32 KB in bytes
// Launch only one thread to calcaulte the latency using a pointer-chasing
// array technique
// #define THREADS_NUM 256
// iterate over the array ITERS times
#ifndef ITERS
#define ITERS  (1024 )
#endif

//#define U32ACCESS


#ifndef ILPconfig
#define ILPconfig 1
#endif

static_assert(ILPconfig<=1,"ILP > 1 is not supported\n");

// two way bank conflict - > 23 latenct
// bank-conflict-free -> 25 latency

#ifdef U32ACCESS
typedef uint32_t shared_m;
#else
typedef uint64_t shared_m;
#endif

// Measure latency of ITERS reads.
__global__ void shared_lat(uint32_t *startClk, uint32_t *stopClk,
  shared_m *dsink, uint32_t stride) {

  // thread index
  uint32_t tid = threadIdx.x;
  uint32_t bid = blockIdx.x;
  uint32_t uid = bid * blockDim.x + tid;
  uint32_t n_threads = blockDim.x * gridDim.x;

  //__shared__ shared_m s[SHARED_MEM_SIZE]; // static shared memory

  extern __shared__ int smem[]; // dynamic 

  shared_m *s = (shared_m*)&smem[0];

  int s_smem = SHARED_MEM_SIZE/sizeof(shared_m);

  // one thread to initialize the pointer-chasing array
  // for (uint32_t i = uid; i < (s_smem - stride); i += n_threads)
  //   s[i] = (i + stride) % s_smem;

    if(uid == 0){
      for (uint32_t i = 0; i < (s_smem - stride); i ++)
        s[i] = (i + stride) % s_smem; //s[i] = (i )*16 % 2048; // s[i] is multiple of 16, because addree is aligned with 4 bytes
    }
    // 
  asm volatile("bar.sync 0;");

  // if(uid == 0){
  //   for(int i = 0; i < s_smem; i ++){
  //     printf("s[%d] = %d \t", i, int(s[i]) );

  //   }
  //   printf("\n");
  // }

  //if (uid == 0) {
    // initalize pointer chaser
    shared_m p_chaser = threadIdx.x*stride ;

    #ifdef U32ACCESS
    shared_m p_chaser_1 = threadIdx.x  + 32;
    #endif
    // start timing
    uint32_t start = 0;
    asm volatile("mov.u32 %0, %%clock;" : "=r"(start)::"memory");

    // pointer-chasing ITERS times
    for (uint32_t i = 0; i < ITERS; ++i) {
      p_chaser = s[p_chaser];

      #ifdef U32ACCESS
      p_chaser_1 =s[p_chaser_1];
      #endif

      //p_chaser_1 =s[p_chaser_1];
      //asm volatile("bar.sync 0;");
    }

    // stop timing
    asm volatile("bar.sync 0;");
    uint32_t stop = 0;
    asm volatile("mov.u32 %0, %%clock;" : "=r"(stop)::"memory");

    // write time and data back to memory
    if(uid  == 0){
      startClk[uid] = start;
      stopClk[uid] = stop;
      dsink[uid] = p_chaser;// + p_chaser_1;
      #ifdef U32ACCESS
      dsink[uid] += p_chaser_1;
      #endif

    }

  //}
}


void test_with_different_thread(int stride, int THREADS_NUM){
  //int n_warps = THREADS_NUM/32;
  BLOCKS_NUM = 1;
  TOTAL_THREADS = THREADS_NUM * BLOCKS_NUM;
  THREADS_PER_SM = THREADS_NUM * BLOCKS_NUM;

  // if( n_warps == 8 ){
  //   #define SHARED_MEM_SIZE (16 * 1024)
  // }else{
  //   #define SHARED_MEM_SIZE (32 * 1024)
  // }

  assert(SHARED_MEM_SIZE  <= MAX_SHARED_MEM_SIZE_PER_BLOCK);

  uint32_t *startClk = (uint32_t *)malloc(sizeof(uint32_t));
  uint32_t *stopClk = (uint32_t *)malloc(sizeof(uint32_t));
  shared_m *dsink = (shared_m *)malloc(sizeof(shared_m));

  uint32_t *startClk_g;
  uint32_t *stopClk_g;
  shared_m *dsink_g;

  gpuErrchk(cudaMalloc(&startClk_g, sizeof(uint32_t)));
  gpuErrchk(cudaMalloc(&stopClk_g, sizeof(uint32_t)));
  gpuErrchk(cudaMalloc(&dsink_g, sizeof(shared_m)));

  shared_lat<<<1, THREADS_NUM, SHARED_MEM_SIZE>>>(startClk_g, stopClk_g, dsink_g, stride);
  gpuErrchk(cudaPeekAtLastError());

  gpuErrchk(cudaMemcpy(startClk, startClk_g, sizeof(uint32_t),
                       cudaMemcpyDeviceToHost));
  gpuErrchk(
      cudaMemcpy(stopClk, stopClk_g, sizeof(uint32_t), cudaMemcpyDeviceToHost));
  gpuErrchk(
      cudaMemcpy(dsink, dsink_g, sizeof(shared_m), cudaMemcpyDeviceToHost));

  float lat = (float)(stopClk[0] - startClk[0]) / ITERS;

  std::cout << THREADS_NUM/32 <<" warps Shared Memory read(8B/t) latency " << lat <<" ( " <<(unsigned)(lat) << " ) " << std::endl;

  long num_bytes =  (THREADS_NUM) * 8;
  std::cout << "Shared mem throughput = " << num_bytes / lat << " bytes/clk " <<std::endl;
  std::cout << "Total Clk number " <<  stopClk[0] - startClk[0] <<std::endl;
  std::cout << std::endl;
  
  cudaDeviceSynchronize();


  // printf("Shared Memory Latency  = %f cycles\n", lat);
  // printf("Total Clk number = %u \n", stopClk[0] - startClk[0]);

  // if (ACCEL_SIM_MODE) {
  //   std::cout << "\n//Accel_Sim config: \n";
  //   std::cout << "-gpgpu_smem_latency " << (unsigned)(lat) << std::endl;
  // }
}


int main() {
  intilizeDeviceProp(0);
  for(int i = 1; i <= 32; i = i*2){
    test_with_different_thread(1,32*i);
  }

  //test_with_different_thread(1,32*14);

  return 0;
}
