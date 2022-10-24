#include <cuda.h>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>

#include "../../../hw_def/hw_def.h"

#define SHARED_MEM_SIZE (32 * 1024 ) // 32k
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

#define U32ACCESS

// two way bank conflict - > 23 latenct
// bank-conflict-free -> 25 latency

#ifdef U32ACCESS
typedef uint32_t shared_m;
#else
typedef uint64_t shared_m;
#endif
// two way bank conflict - > 23 latenct
// bank-conflict-free -> 25 latency


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
  
  if(uid == 0){
    for (uint32_t i = 0; i < (s_smem - stride); i ++)
      s[i] = (i + stride) % s_smem; //s[i] = (i )*16 % 2048; // s[i] is multiple of 16, because addree is aligned with 4 bytes
  }
  // one thread to initialize the pointer-chasing array
  // for (uint32_t i = uid; i < (SHARED_MEM_SIZE - stride); i += n_threads)
  //   s[i] = (i + stride) % SHARED_MEM_SIZE;

  asm volatile("bar.sync 0;");

  // if(uid == 0){
  //   for(int i = 0; i < SHARED_MEM_SIZE; i ++){
  //     printf("s[%d] = %d \t", i, s[i]);

  //   }
  //   printf("\n");
  // }

  //if (uid == 0) {
    // initalize pointer chaser
    shared_m p_chaser = threadIdx.x * stride;;

    // start timing
    uint32_t start = 0;
    asm volatile("mov.u32 %0, %%clock;" : "=r"(start)::"memory");

    // pointer-chasing ITERS times
    for (uint32_t i = 0; i < ITERS; ++i) {
      p_chaser = s[p_chaser];
    }

    // stop timing
    uint32_t stop = 0;
    asm volatile("mov.u32 %0, %%clock;" : "=r"(stop)::"memory");

    // write time and data back to memory
    if(uid == 0){
      startClk[uid] = start;
      stopClk[uid] = stop;
      dsink[uid] = p_chaser;
    }

  //}
}


// n-way bank conflict (n = 1,2,4,8...32)
void bank_conflict_test(int n, int THREADS_NUM){

  

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

  shared_lat<<<1, THREADS_NUM,SHARED_MEM_SIZE>>>(startClk_g, stopClk_g, dsink_g, n);
  gpuErrchk(cudaPeekAtLastError());

  gpuErrchk(cudaMemcpy(startClk, startClk_g, sizeof(uint32_t),
                       cudaMemcpyDeviceToHost));
  gpuErrchk(
      cudaMemcpy(stopClk, stopClk_g, sizeof(uint32_t), cudaMemcpyDeviceToHost));
  gpuErrchk(
      cudaMemcpy(dsink, dsink_g, sizeof(shared_m), cudaMemcpyDeviceToHost));

  float lat = (float)(stopClk[0] - startClk[0]) / ITERS;
  
  //printf("Shared Memory Latency  = %f cycles\n", lat);
  std::cout << n <<"-way bank conflict ,  " << THREADS_NUM/32 <<" warps, latency = " << lat <<" ( " <<(unsigned)(lat) << " ) " << std::endl;

  long num_bytes =  (THREADS_NUM) * 4;
  std::cout << "Shared mem throughput = " << num_bytes / lat << " bytes/clk " <<std::endl;
  std::cout << "Total Clk number " <<  stopClk[0] - startClk[0] <<std::endl;
  std::cout << std::endl;
  
  cudaDeviceSynchronize();

  //printf("Total Clk number = %u \n", stopClk[0] - startClk[0]);

  // if (ACCEL_SIM_MODE) {
  //   std::cout << "\n//Accel_Sim config: \n";
  //   std::cout << "-gpgpu_smem_latency " << (unsigned)(lat) << std::endl;
  // }

}




int main() {
  intilizeDeviceProp(0);

  std::vector<int> warps = {1,4,8};
  for(auto& e:warps){
    bank_conflict_test(1, 32*e );
    std::cout <<"***************************************"<<std::endl;
    bank_conflict_test(2, 32*e );
    std::cout <<"***************************************"<<std::endl;
    bank_conflict_test(4, 32*e );
    std::cout <<"***************************************"<<std::endl;
    bank_conflict_test(8, 32*e );
    std::cout <<"***************************************"<<std::endl;
  }

  // for(int i = 1; i<=32; i=i*2){
  //   bank_conflict_test(1, 32*i );
  // }
  // bank_conflict_test(1);
  // bank_conflict_test(2);
  return 0;
}
