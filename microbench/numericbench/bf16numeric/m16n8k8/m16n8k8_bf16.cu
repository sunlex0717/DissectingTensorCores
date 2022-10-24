// simple gemm  using bf16/half data types
// we do not target on optimal overall performance, so we will not use software pipepline
// pipepline or asychronous copy can speed up gemm further with cost of extra shared memory storage
// CUTLASS provides good examples of how to implement pipeline for gemm
#include <algorithm>
#include <cuda.h>
#include <iostream>
#include <mma.h>
#include <vector>
#include <stdio.h>
#include <stdlib.h>
#include <cstdlib>
#include <cuda_fp16.h>
#include <random>
#include "../../../hw_def/hw_def.h"
#include "../../cpu_base.h"

typedef __nv_bfloat16 op_AB; 
typedef float op_CD; 

#ifndef MEAN
#define MEAN (0.0)
#endif

#ifndef STDDEV
#define STDDEV (1.0)
#endif


#ifndef ITERS
#define ITERS  (1024 )
#endif

#define ROUNDS  (ITERS*10 )

const int inst_m = 16;
const int inst_n = 8;
const int inst_k = 8;

// we want to know the numeric precision of PTX instruction - the lowerst programming interface.
// Since higher-level applications are based on the PTX instruction, the numeric errors/differences higher-level applications are based on the ptx instruction. 
// __forceinline__ __device__ unsigned lane_id()
// {
//     unsigned ret; 
//     asm volatile ("mov.u32 %0, %laneid;" : "=r"(ret));
//     return ret;
// }

__forceinline__ __device__ unsigned lane_id_()
{
    unsigned ret; 
    asm volatile ("mov.u32 %0, %laneid;" : "=r"(ret));
    return ret;
}



__global__ void gemm_m16n8k8_kernel(op_AB* MatA,op_AB* MatB,op_CD* MatC, op_CD* MatD ){
    //uint32_t tid = threadIdx.x;
    //uint32_t gid = blockIdx.x * blockDim.x + tid;//global at this block
    //uint32_t warpid = gid / warpSize;
    uint32_t lane_id =  lane_id_();
    // four threads per group, group id
    uint32_t group_id = lane_id >>2;
    uint32_t tid_in_group = lane_id % 4;

    // m16 n8 k16
    op_AB frag_A[4]; // 16 * 16  / 32 = 8 * bf16
    op_AB frag_B[2]; // 8 * 16  / 32
    op_CD frag_D[4]; // float , 16*8 /32 = 4*float
    // load operand fragA
    #pragma unroll
    for(int i =0; i < 4; i++){
        uint32_t row_a = 0;
        if( i==0 || i ==1 ){
            row_a = group_id;
        }else{
            row_a = group_id + 8;
        }

        uint32_t col_a = (tid_in_group * 2) + (i & 0x1);
        // row major
        frag_A[i] = MatA[inst_k*row_a + col_a];
        
    }
    // for(int i =0; i < 8; i++){
    //     printf("laneId = %d, fragA[%d] = %f \n", lane_id, i, float(frag_A[i]));
    // }

    // load operand fragB, MatB has to be col-major
    #pragma unroll
    for(int i =0; i < 2; i++){
        uint32_t row_b =  (tid_in_group * 2) + (i);

        uint32_t col_b = group_id;
        // row-major B
        frag_B[i] = MatB[row_b*inst_n + col_b];
    }

    // for(int i =0; i < 4; i++){
    //     printf("laneId = %d, fragB[%d] = %f \n", lane_id, i, float(frag_B[i]));
    // }

    // load operand fragC, MatC has to be row-major
    #pragma unroll
    for(int i =0; i < 4; i++){
        uint32_t row_c = 0;
        if( i < 2 ){
            row_c = group_id;
        }else{
            row_c = group_id + 8;
        }
        uint32_t col_c = (tid_in_group * 2) + (i & 0x1);
        // row-major
        frag_D[i] = MatC[inst_n*row_c + col_c];
    }

    // printf("\n\n");
    // for(int i =0; i < 4; i++){
    //     printf("laneId = %d, fragC[%d] = %f \n", lane_id, i, float(frag_D[i]));
    // }

    //step 1: load data 
    // MatA => frag_A, MatB => frag_B, MatC => frag_C


    uint32_t const *A = reinterpret_cast<uint32_t const *>(&frag_A[0]);
    uint32_t const *B = reinterpret_cast<uint32_t const *>(&frag_B[0]);//?
    float *C = reinterpret_cast<float *>(&frag_D[0]);
    float *D = C;  // D = A*B + D.

    asm volatile(
        "mma.sync.aligned.m16n8k8.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%5}, {%6}, {%7,%8,%9,%10};\n"
        : "=f"(D[0]), "=f"(D[1]), "=f"(D[2]), "=f"(D[3])
        : "r"(A[0]), "r"(A[1]), // "r"(A[2]), "r"(A[3]), 
          "r"(B[0]), //"r"(B[1]),
          "f"(C[0]), "f"(C[1]), "f"(C[2]), "f"(C[3])
      );
    
    __syncwarp();

    // store back result
    // printf("\n\n");
    // for(int i =0; i < 4; i++){
    //     printf("laneId = %d, fragD[%d] = %f \n", lane_id, i, float(frag_D[i]));
    // }
    #pragma unroll
    for(int i =0; i < 4; i++){
        uint32_t row_d = 0;
        if( i < 2 ){
            row_d = group_id;
        }else{
            row_d = group_id + 8;
        }
        uint32_t col_d = (tid_in_group * 2) + (i & 0x1);
        // row-major
        MatD[inst_n*row_d + col_d] = frag_D[i];
    }

}





std::vector<double> gemm_m16n8k8_bf16(){
    int BLOCKS_NUM = 1;
    int nwarps = 1;
    int warp_size = 32;

    
    unsigned total_A_SIZE = inst_m*inst_k*nwarps;
    unsigned total_B_SIZE = inst_k*inst_n*nwarps;
    unsigned total_C_SIZE = inst_m*inst_n*nwarps;


    op_AB *host_matA = (op_AB *)malloc(total_A_SIZE * sizeof(op_AB));
    op_AB *host_matB = (op_AB *)malloc(total_B_SIZE * sizeof(op_AB));

    op_CD *host_matC = (op_CD *)malloc(total_C_SIZE * sizeof(op_CD));
    op_CD *host_matD = (op_CD *)malloc(total_C_SIZE * sizeof(op_CD));
    std::random_device rd{};
    std::mt19937 gen{rd()};
    std::normal_distribution<> random_gen{MEAN,STDDEV};
    // initialize A, row-major
    float *host_matA_cpu = (float *)malloc(total_A_SIZE * sizeof(float));
    float *host_matB_cpu = (float *)malloc(total_B_SIZE * sizeof(float));
    for(int r = 0; r < inst_m; r ++){
        for(int c = 0; c < inst_k; c ++){
            //float rnd = (float)(r*inst_k+c);
            float rnd = (float)random_gen(gen);
            host_matA_cpu[r*inst_k+c] = rnd;
            host_matA[r*inst_k+c] = (op_AB)rnd;
        }
    }
    // std::cout<<"print MatA" <<std::endl;
    // print_mat(host_matA_cpu,inst_m,inst_k);

    // initialize B, row-major
    for(int r = 0; r < inst_k; r ++){
        for(int c = 0; c < inst_n; c ++){
            float rnd = (float)random_gen(gen);
            //float rnd = float(r*inst_n+c);   //(float)random_gen(gen);
            host_matB_cpu[r*inst_n+c] = rnd;
            host_matB[r*inst_n+c] = (op_AB)rnd;
        }
    }
    // std::cout<<"print MatB" <<std::endl;
    // print_mat(host_matB_cpu,inst_k,inst_n);

    // initialize C, row-major
    for(int r = 0; r < inst_m; r ++){
        for(int c = 0; c < inst_n; c ++){
            host_matC[r*inst_n+c] = 0.0 ;//  (op_CD)random_gen(gen);
        }
    }



    float *cpu_res_baseline = (op_CD *)malloc(total_C_SIZE * sizeof(op_CD));
    // host computation
    gemm_mnk_cpu(host_matA_cpu,host_matB_cpu,host_matC,cpu_res_baseline,inst_m,inst_n,inst_k);

    

    //device ptr
    op_AB *dev_matA;
    op_AB *dev_matB;
    op_CD *dev_matC;

    op_CD *dev_matD;
    // allocate device global memory
    // D = A*B + C
    cudaMalloc(&dev_matA, total_A_SIZE * sizeof(op_AB));
    cudaMalloc(&dev_matB, total_B_SIZE * sizeof(op_AB));
    cudaMalloc(&dev_matC, total_C_SIZE * sizeof(op_CD));
    cudaMalloc(&dev_matD, total_C_SIZE * sizeof(op_CD));
    // copy data from host to device
    gpuErrchk(cudaMemcpy(dev_matA, host_matA, total_A_SIZE * sizeof(op_AB), cudaMemcpyHostToDevice));

    gpuErrchk(cudaMemcpy(dev_matB, host_matB, total_B_SIZE * sizeof(op_AB), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(dev_matC, host_matC, total_C_SIZE * sizeof(op_CD), cudaMemcpyHostToDevice));

    gemm_m16n8k8_kernel<<<BLOCKS_NUM, nwarps*warp_size>>>(dev_matA,dev_matB,dev_matC,dev_matD);
    gpuErrchk(cudaPeekAtLastError());

    gpuErrchk(cudaMemcpy(host_matD, dev_matD, total_C_SIZE * sizeof(op_CD), cudaMemcpyDeviceToHost));

    //check errors
    double l1_norm = 0.0;
    double abs_err = 0.0;
    double l2_relative_err = 0.0;
    compute_diff_l1_norm(cpu_res_baseline,host_matD,inst_m,inst_n,abs_err,l1_norm);
    compute_diff_l2_norm(cpu_res_baseline,host_matD,inst_m,inst_n,l2_relative_err);
    // std::cout<<"print cpu_res_baseline" <<std::endl;

    // print_mat(cpu_res_baseline,inst_m,inst_n);

    // std::cout<<"print GPU res" <<std::endl;
    // print_mat(host_matD,inst_m,inst_n);

    // std::cout<<"element-wise abs_err = " << abs_err <<std::endl;
    // std::cout<<"element-wise l1 norm = " << l1_norm <<std::endl;

    // std::cout<<"err/FMA :"<<std::endl;
    // std::cout<<"abs_err = " << abs_err/inst_k <<std::endl;
    // std::cout<<"l1 norm = " << l1_norm/inst_k <<std::endl;
    
    std::vector<double> errors{abs_err,l1_norm,abs_err/inst_k,l1_norm/inst_k,l2_relative_err};
    
    return errors;
}




int main(){
    std::cout<<"***********************************"<<std::endl;
    std::cout << "mma.sync.aligned.m16n8k8.row.col.f32.bf16.bf16.f32 numeric errors w.r.t fp32 on cpu"  << std::endl;

    double avg_abs_err = 0.0;
    double avg_l1_norm = 0.0;
    double avg_abs_err_FMA = 0.0;
    double avg_l1_norm_FMA = 0.0;
    double l2_relative = 0.0;
    for(int i=0;i < ROUNDS; i ++){
        std::vector<double> errors = gemm_m16n8k8_bf16();
        avg_abs_err += errors[0];
        avg_l1_norm += errors[1];
        avg_abs_err_FMA += errors[2];
        avg_l1_norm_FMA += errors[3];
        l2_relative += errors[4];
    }

    // std::cout<<"element-wise avg_abs_err = " << avg_abs_err/ITERS <<std::endl;
    // std::cout<<"element-wise avg_l1_norm_err = " << avg_l1_norm/ITERS <<std::endl;

    std::cout<<"l2 relative error :"<< l2_relative/ROUNDS << std::endl;
    std::cout<<"l2 relative error per FMA :"<< l2_relative/(ROUNDS*inst_m*inst_k*inst_n) << std::endl;

    // std::cout<<"error/FMA :"<<std::endl;
    // std::cout<<"avg_abs_err_FMA = " << avg_abs_err_FMA/ITERS <<std::endl;
    // std::cout<<"avg_l1_norm_FMA = " << avg_l1_norm_FMA/ITERS <<std::endl;
}



