#include <cuda.h>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>

#include "../../../hw_def/hw_def.h"

#define SHARED_MEM_SIZE (48 * 1024 / 4) // 32 KB
// Launch only one thread to calcaulte the latency using a pointer-chasing
// array technique
//#define THREADS_NUM 128
// iterate over the array ITERS times
#ifndef ITERS
#define ITERS  (1024 )
#endif

#ifndef ILPconfig
#define ILPconfig 1
#endif

static_assert(ILPconfig<=6,"ILP > 6 is not implemented\n");
// two way bank conflict - > 23 latenct
// bank-conflict-free -> 25 latency

typedef uint32_t shared_m;
// Measure latency of ITERS ldmatrix.x1
__global__ void shared_lat(uint32_t *startClk, uint32_t *stopClk,
  shared_m *dsink, uint32_t stride) {

  // thread index
  uint32_t tid = threadIdx.x;
  uint32_t bid = blockIdx.x;
  uint32_t uid = bid * blockDim.x + tid;
  uint32_t n_threads = blockDim.x * gridDim.x;

  __shared__ shared_m s[SHARED_MEM_SIZE]; // static shared memory

  // one thread to initialize the pointer-chasing array
  if(uid == 0){
    for (uint32_t i = 0; i < (SHARED_MEM_SIZE - stride); i ++)
      s[i] = (i )*16 % 2048; // s[i] is multiple of 16, because addree is aligned with 4 bytes
  }
    
    asm volatile("bar.sync 0;");

    unsigned addr = static_cast<unsigned>(__cvta_generic_to_shared(&s[threadIdx.x*4]));
    unsigned addr_1 = 0;

    unsigned addr2 = static_cast<unsigned>(__cvta_generic_to_shared(&s[(threadIdx.x + 32) *4]));
    unsigned addr2_1 = 0;

    unsigned addr3 = static_cast<unsigned>(__cvta_generic_to_shared(&s[(threadIdx.x + 64) *4]));
    unsigned addr3_1 = 0;

    unsigned addr4 = static_cast<unsigned>(__cvta_generic_to_shared(&s[(threadIdx.x + 96) *4]));
    unsigned addr4_1 = 0;

    unsigned addr5 = static_cast<unsigned>(__cvta_generic_to_shared(&s[(threadIdx.x + 32*4) *4]));
    unsigned addr5_1 = 0;

    unsigned addr6 = static_cast<unsigned>(__cvta_generic_to_shared(&s[(threadIdx.x + 32*5) *4]));
    unsigned addr6_1 = 0;
    //printf("thread %d , addr = %d \n", tid, addr);
    // start timing
    uint32_t start = 0;
    asm volatile("mov.u32 %0, %%clock;" : "=r"(start)::"memory");

    // pointer-chasing ITERS times
    #pragma unroll
    for (uint32_t i = 0; i < ITERS; ++i) {
        asm volatile ("ldmatrix.sync.aligned.x2.m8n8.shared.b16 {%0, %1}, [%2];" : "=r"(addr), "=r"(addr_1) : "r"(addr)); 
        #if ILPconfig >= 2
        asm volatile ("ldmatrix.sync.aligned.x2.m8n8.shared.b16 {%0, %1}, [%2];" : "=r"(addr2), "=r"(addr2_1) : "r"(addr2)); 
        #endif
        #if ILPconfig >= 3
        asm volatile ("ldmatrix.sync.aligned.x2.m8n8.shared.b16 {%0, %1}, [%2];" : "=r"(addr3), "=r"(addr3_1) : "r"(addr3)); 
        #endif
        #if ILPconfig >= 4
        asm volatile ("ldmatrix.sync.aligned.x2.m8n8.shared.b16 {%0, %1}, [%2];" : "=r"(addr4), "=r"(addr4_1) : "r"(addr4)); 
        #endif

        #if ILPconfig >= 5
        asm volatile ("ldmatrix.sync.aligned.x2.m8n8.shared.b16 {%0, %1}, [%2];" : "=r"(addr5), "=r"(addr5_1) : "r"(addr5)); 
        #endif

        #if ILPconfig >= 6
        asm volatile ("ldmatrix.sync.aligned.x2.m8n8.shared.b16 {%0, %1}, [%2];" : "=r"(addr6), "=r"(addr6_1) : "r"(addr6)); 
        #endif



        __syncwarp();
    }
    uint32_t stop = 0;
    asm volatile("mov.u32 %0, %%clock;" : "=r"(stop)::"memory");

    //printf("thread %d , x = %d \n", tid, addr);

    // write time and data back to memory
    startClk[uid] = start;
    stopClk[uid] = stop;
    dsink[uid] = addr + addr_1;

    dsink[uid] += addr2 + addr2_1;

    dsink[uid] += addr3 + addr3_1;
    dsink[uid] += addr4 + addr4_1;
    dsink[uid] += addr5 + addr5_1;
    dsink[uid] += addr6 + addr6_1;


}

void test_with_different_thread(int THREADS_NUM, int ILP){
  BLOCKS_NUM = 1;
  TOTAL_THREADS = THREADS_NUM * BLOCKS_NUM;
  THREADS_PER_SM = THREADS_NUM * BLOCKS_NUM;

  assert(SHARED_MEM_SIZE * sizeof(shared_m) <= MAX_SHARED_MEM_SIZE_PER_BLOCK);

  uint32_t *startClk = (uint32_t *)malloc(sizeof(uint32_t));
  uint32_t *stopClk = (uint32_t *)malloc(sizeof(uint32_t));
  shared_m *dsink = (shared_m *)malloc(sizeof(shared_m));

  uint32_t *startClk_g;
  uint32_t *stopClk_g;
  shared_m *dsink_g;

  gpuErrchk(cudaMalloc(&startClk_g, sizeof(uint32_t)));
  gpuErrchk(cudaMalloc(&stopClk_g, sizeof(uint32_t)));
  gpuErrchk(cudaMalloc(&dsink_g, sizeof(shared_m)));

  shared_lat<<<1, THREADS_NUM>>>(startClk_g, stopClk_g, dsink_g, 1);
  gpuErrchk(cudaPeekAtLastError());
    //printf("pass kenerl \n");
  gpuErrchk(cudaMemcpy(startClk, startClk_g, sizeof(uint32_t),
                       cudaMemcpyDeviceToHost));
  gpuErrchk(
      cudaMemcpy(stopClk, stopClk_g, sizeof(uint32_t), cudaMemcpyDeviceToHost));
  gpuErrchk(
      cudaMemcpy(dsink, dsink_g, sizeof(shared_m), cudaMemcpyDeviceToHost));

  float lat = (float)(stopClk[0] - startClk[0]) / ITERS;


  std::cout << THREADS_NUM/32 <<" warps ldmatrix.x2 latency " << lat <<" ( " <<(unsigned)(lat) << " ) " << std::endl;

  long num_bytes =  (THREADS_NUM/32) * 8 * 8 * 2 * 2 * ILP;
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
  std::vector<int> warps = {1,2,4,6,8,12,16,20,24,28,32};
  //std::vector<int> warps = {4,8,12,16};
  intilizeDeviceProp(0);
  //std::cout << "ldmatrix.x2 microbenchmark " <<std::endl;
  std::cout<<"***********************************"<<std::endl;
  std::cout << "ldmatrix.x2 microbenchmark with ILP = " << ILPconfig << std::endl;
  for(auto &e:warps){
    test_with_different_thread(32*e,ILPconfig);
  }
  

  return 0;
}
