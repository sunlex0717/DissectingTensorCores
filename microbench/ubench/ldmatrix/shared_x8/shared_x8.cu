#include <cuda.h>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>

#include "../../../hw_def/hw_def.h"

#define SHARED_MEM_SIZE  (32*1024) //(32 * 1024 ) // in bytes
// Launch only one thread to calcaulte the latency using a pointer-chasing
// array technique
// #define THREADS_NUM 32
// iterate over the array ITERS times
#ifndef ITERS
#define ITERS  (1024 )
#endif


#ifndef ILPconfig
#define ILPconfig 1
#endif

static_assert(ILPconfig<=1,"ILP > 1 is not supported\n");


// two way bank conflict - > 23 latenct
// bank-conflict-free -> 25 latency

#define U32ACCESS

// two way bank conflict - > 23 latenct
// bank-conflict-free -> 25 latency

#ifdef U32ACCESS
typedef uint32_t shared_m;
#else
typedef uint64_t shared_m;
#endif

//typedef uint32_t shared_m;

// typedef uint64_t shared_m;
// Measure latency of ITERS reads.
__global__ void shared_lat(uint32_t *startClk, uint32_t *stopClk,
  shared_m *dsink, uint32_t stride) {

  // thread index
  uint32_t tid = threadIdx.x;
  uint32_t bid = blockIdx.x;
  uint32_t uid = bid * blockDim.x + tid;
  uint32_t n_threads = blockDim.x * gridDim.x;

  extern __shared__ int smem[]; // dynamic 

  shared_m *s = (shared_m*)&smem[0];

  int s_smem = SHARED_MEM_SIZE/sizeof(shared_m);

  // // one thread to initialize the pointer-chasing array
  // for (uint32_t i = uid; i < (s_smem - stride); i += n_threads)
  //   s[i] = (i + stride) % s_smem;
    if(uid == 0){
      for (uint32_t i = 0; i < (s_smem - stride); i ++)
        s[i] = shared_m((i + stride) % s_smem); //s[i] = (i )*16 % 2048; // s[i] is multiple of 16, because addree is aligned with 4 bytes
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
    shared_m p_chaser = threadIdx.x * stride;
    //p_chaser = static_cast<unsigned>(__cvta_generic_to_shared(&s[p_chaser]));
    shared_m p_chaser_1 = threadIdx.x * stride + 32;

    
    
    shared_m p_chaser_2 = threadIdx.x * stride + 64;
    shared_m p_chaser_3 = threadIdx.x * stride + 96;
    


    #ifdef U32ACCESS
    shared_m p_chaser_4 = threadIdx.x * stride + 32*4;
    shared_m p_chaser_5 = threadIdx.x * stride + 32*5;
    shared_m p_chaser_6 = threadIdx.x * stride + 32*6;
    shared_m p_chaser_7 = threadIdx.x * stride + 32*7;
    #endif

    // start timing
    uint32_t start = 0;
    asm volatile("mov.u32 %0, %%clock;" : "=r"(start)::"memory");

    // pointer-chasing ITERS times
    //#pragma unroll
    for (uint32_t i = 0; i < ITERS; ++i) {
      p_chaser = s[p_chaser];  // ld.shared.u64 %0, [%1];. 
      // asm volatile("ld.shared.u32 %0, [%1];" : "=r"(p_chaser) : "r"(p_chaser*4) );

      // asm volatile("ld.shared.u32 %0, [%1];" : "=r"(p_chaser_1) : "r"(p_chaser_1*4) );
      // asm volatile("ld.shared.u32 %0, [%1];" : "=r"(p_chaser_2) : "r"(p_chaser_2*4) );
      // asm volatile("ld.shared.u32 %0, [%1];" : "=r"(p_chaser_3) : "r"(p_chaser_3*4) );

      p_chaser_1 =s[p_chaser_1];

      
      p_chaser_2 =s[p_chaser_2];
      p_chaser_3 =s[p_chaser_3];

      #ifdef U32ACCESS
      p_chaser_4 =s[p_chaser_4];
      p_chaser_5 =s[p_chaser_5];
      p_chaser_6 =s[p_chaser_6];
      p_chaser_7 =s[p_chaser_7];
      #endif
      // p_chaser_2 =s[p_chaser_2];
      // p_chaser_3 =s[p_chaser_3];
      

      
      
      // p_chaser_1 =s[p_chaser_1];
      // p_chaser_2 =s[p_chaser_2];
      // p_chaser_3 =s[p_chaser_3];
      //asm volatile("bar.sync 0;");
    }

    // stop timing
    asm volatile("bar.sync 0;");
    uint32_t stop = 0;
    
    asm volatile("mov.u32 %0, %%clock;" : "=r"(stop)::"memory");

    // write time and data back to memory
    if(uid == 0){
      startClk[uid] = start;
      stopClk[uid] = stop;
      dsink[uid] = p_chaser + p_chaser_1 +p_chaser_2 + p_chaser_3  ; // + p_chaser_2 + p_chaser_3;
  
      #ifdef U32ACCESS
        dsink[uid] += (p_chaser_4 + p_chaser_5 +p_chaser_6 + p_chaser_7);
      #endif
    }

  //}
}

void test_with_different_thread(int stride, int THREADS_NUM){
  BLOCKS_NUM = 1;
  TOTAL_THREADS = THREADS_NUM * BLOCKS_NUM;
  THREADS_PER_SM = THREADS_NUM * BLOCKS_NUM;

  assert(SHARED_MEM_SIZE <= MAX_SHARED_MEM_SIZE_PER_BLOCK);

  uint32_t *startClk = (uint32_t *)malloc(sizeof(uint32_t));
  uint32_t *stopClk = (uint32_t *)malloc(sizeof(uint32_t));
  shared_m *dsink = (shared_m *)malloc(sizeof(shared_m));

  uint32_t *startClk_g;
  uint32_t *stopClk_g;
  shared_m *dsink_g;

  gpuErrchk(cudaMalloc(&startClk_g, sizeof(uint32_t)));
  gpuErrchk(cudaMalloc(&stopClk_g, sizeof(uint32_t)));
  gpuErrchk(cudaMalloc(&dsink_g, sizeof(shared_m)));

  shared_lat<<<BLOCKS_NUM, THREADS_NUM,SHARED_MEM_SIZE>>>(startClk_g, stopClk_g, dsink_g, stride);
  gpuErrchk(cudaPeekAtLastError());

  gpuErrchk(cudaMemcpy(startClk, startClk_g, sizeof(uint32_t),
                       cudaMemcpyDeviceToHost));
  gpuErrchk(
      cudaMemcpy(stopClk, stopClk_g, sizeof(uint32_t), cudaMemcpyDeviceToHost));
  gpuErrchk(
      cudaMemcpy(dsink, dsink_g, sizeof(shared_m), cudaMemcpyDeviceToHost));

  float lat = (float)(stopClk[0] - startClk[0]) / ITERS;

  std::cout << THREADS_NUM/32 <<" warps Shared Memory read(16B/t) latency " << lat <<" ( " <<(unsigned)(lat) << " ) " << std::endl;

  long num_bytes =  (THREADS_NUM) * 32;
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
  std::cout << " ld.shared x 8"<< std::endl;
  for(int i = 1; i <= 32; i = i*2){
    test_with_different_thread(1,32*i);
  }
  //test_with_different_thread(1,32);
  //test_with_different_thread(32*6);
  return 0;
}
