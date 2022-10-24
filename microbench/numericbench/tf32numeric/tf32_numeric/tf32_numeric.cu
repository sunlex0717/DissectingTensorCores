// one m16n8k8_tf32 = two m16n8k4 
// see the errors for a same m16n8k8
// chain matrix multiplication
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
#include <cstring>


typedef float op_AB; 
typedef float op_CD; 


#ifndef ITERS
#define ITERS  (1024 )
#endif

#define ROUNDS  (ITERS*10 )

const int inst_m = 16;
const int inst_n = 8;
const int inst_k = 8;

__forceinline__ __device__ unsigned lane_id_()
{
    unsigned ret; 
    asm volatile ("mov.u32 %0, %laneid;" : "=r"(ret));
    return ret;
}


__global__ void gemm_m16n8k4_kernel(op_AB* MatA,op_AB* MatB,op_CD* MatC, op_CD* MatD ){

    // constexpr const int inst_m = 16;
    // constexpr const int inst_n = 8;
    constexpr const int inst_k = 4;
    uint32_t lane_id =  lane_id_();
    // four threads per group, group id
    uint32_t group_id = lane_id >>2;
    uint32_t tid_in_group = lane_id % 4;
    // m16 n8 k4
    uint32_t frag_A[2]; // 16 * 16  / 32 = 8 * bf16
    uint32_t frag_B[1]; // 8 * 16  / 32
    op_CD frag_D[4]; // float , 16*8 /32 = 4*float
    // load operand fragA
    #pragma unroll
    for(int i =0; i < 2; i++){
        uint32_t row_a = 0;
        uint32_t col_a = 0;
        if( i==0 ){
            row_a = group_id;
        }else{
            row_a = group_id + 8;
        }
        col_a = tid_in_group;
        // row major
        // Cvt Float - TF32
        asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_A[i]) : "f"(MatA[inst_k*row_a + col_a]));
    }

    // for(int i =0; i < 2; i++){
    //     printf("laneId = %d, fragA[%d] = %f \n", lane_id, i, MatA[inst_k*row_a + col_a]));
    // }
    #pragma unroll
    for(int i =0; i < 1; i++){
        uint32_t row_b =  tid_in_group ; 
        uint32_t col_b = group_id;
        // row-major B
        asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[i]) : "f"(MatB[row_b*inst_n + col_b]));
        //frag_B[i] = (MatB[row_b*inst_n + col_b]);
    }

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

    uint32_t const *A = reinterpret_cast<uint32_t const *>(&frag_A[0]);
    uint32_t const *B = reinterpret_cast<uint32_t const *>(&frag_B[0]);//?
    float *C = reinterpret_cast<float *>(&frag_D[0]);
    float *D = C;  // D = A*B + D.

    asm volatile(
        "mma.sync.aligned.m16n8k4.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%7,%8,%9,%10};\n"
        : "=f"(D[0]), "=f"(D[1]), "=f"(D[2]), "=f"(D[3])
        : "r"(A[0]), "r"(A[1]), 
          "r"(B[0]), 
          "f"(C[0]), "f"(C[1]), "f"(C[2]), "f"(C[3])
    );
    
    __syncwarp();

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



__global__ void gemm_m16n8k8_kernel(op_AB* MatA,op_AB* MatB,op_CD* MatC, op_CD* MatD ){

    // constexpr const int inst_m = 16;
    // constexpr const int inst_n = 8;
    constexpr const int inst_k = 8;
    uint32_t lane_id =  lane_id_();
    // four threads per group, group id
    uint32_t group_id = lane_id >>2;
    uint32_t tid_in_group = lane_id % 4;
    // m16 n8 k8
    uint32_t frag_A[4]; // 16 * 16  / 32 = 8 * bf16
    uint32_t frag_B[2]; // 8 * 16  / 32
    op_CD frag_D[4]; // float , 16*8 /32 = 4*float
    // load operand fragA
    #pragma unroll
    for(int i =0; i < 4; i++){
        uint32_t row_a = 0;
        uint32_t col_a = 0;
        if( i==0 || i ==2 ){
            row_a = group_id;
        }else{
            row_a = group_id + 8;
        }
        if(i == 0 || i==1){// i ==0 || i ==2
            col_a = tid_in_group;
        }else{
            col_a = tid_in_group + 4;
        }
        asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_A[i]) : "f"(MatA[inst_k*row_a + col_a]));
    }

    #pragma unroll
    for(int i =0; i < 2; i++){
        uint32_t row_b =  0 ; //(i==0)?tid_in_group:(tid_in_group+4);//  (tid_in_group * 2) + (i);
        if(i == 0){
            row_b = tid_in_group;
        }else{
            row_b = tid_in_group + 4;
        }
        uint32_t col_b = group_id;
        // row-major B
        asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[i]) : "f"(MatB[row_b*inst_n + col_b]));
        //frag_B[i] = (MatB[row_b*inst_n + col_b]);
    }

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

    uint32_t const *A = reinterpret_cast<uint32_t const *>(&frag_A[0]);
    uint32_t const *B = reinterpret_cast<uint32_t const *>(&frag_B[0]);//?
    float *C = reinterpret_cast<float *>(&frag_D[0]);
    float *D = C;  // D = A*B + D.

    asm volatile(
        "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5, %6, %7}, {%8,%9}, {%10,%11,%12,%13};\n"
        : "=f"(D[0]), "=f"(D[1]), "=f"(D[2]), "=f"(D[3])
        : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), 
          "r"(B[0]), "r"(B[1]), 
          "f"(C[0]), "f"(C[1]), "f"(C[2]), "f"(C[3])
    );
    
    __syncwarp();

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


std::vector<double> numeric_tf32_bench(){

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
    std::normal_distribution<> random_gen{-1.0,1.0};
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
    // initialize C, row-major
    for(int r = 0; r < inst_m; r ++){
        for(int c = 0; c < inst_n; c ++){
            host_matC[r*inst_n+c] = 0.0 ;//  (op_CD)random_gen(gen);
        }
    }



    float *cpu_res_baseline = (op_CD *)malloc(total_C_SIZE * sizeof(op_CD));
    // host computation
    gemm_mnk_cpu(host_matA_cpu,host_matB_cpu,host_matC,cpu_res_baseline,inst_m,inst_n,inst_k);


    /***************************************** m16n8k8 *******************************************/
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

    /***************************************** m16n8k4 *******************************************/


    op_AB *host_matA_halfK = (op_AB *)malloc(total_A_SIZE * sizeof(op_AB)/2 );
    op_AB *host_matB_halfK = (op_AB *)malloc(total_B_SIZE * sizeof(op_AB)/2);

    // store temp result
    op_CD *host_matD_halfK = (op_CD *)malloc(total_C_SIZE * sizeof(op_CD));

    // final result
    op_CD *host_matD_final_halfK = (op_CD *)malloc(total_C_SIZE * sizeof(op_CD));

    for(int i = 0; i < inst_m*inst_n; i ++){
        host_matD_final_halfK[i] = 0.0f;
        host_matD_halfK[i] = 0.0f;
    }

    //std::memset(host_matD_final_halfK, 0.0, total_C_SIZE * sizeof(op_CD));
    op_AB *dev_matA_halfK;
    op_AB *dev_matB_halfK;
    op_CD *dev_matC_halfK;

    op_CD *dev_matD_halfK;

    cudaMalloc(&dev_matA_halfK, total_A_SIZE * sizeof(op_AB)/2 );
    cudaMalloc(&dev_matB_halfK, total_B_SIZE * sizeof(op_AB)/2 );
    cudaMalloc(&dev_matC_halfK, total_C_SIZE * sizeof(op_CD));
    cudaMalloc(&dev_matD_halfK, total_C_SIZE * sizeof(op_CD));

    for(int i =0; i <2; i++){
        // copy half K to
        // mat a 
        for(int row = 0; row < inst_m; row ++){
            for(int col = 0; col < inst_k/2; col ++){
                host_matA_halfK[col + row*inst_k/2] = host_matA_cpu[col + row*inst_k + (inst_k/2) * i];
            }
        }

        for(int row = 0; row < inst_k/2; row ++){
            for(int col = 0; col < inst_n; col ++){
                host_matB_halfK[col + row*inst_n] = host_matB_cpu[col + (row + i * inst_k/2)*inst_n ];
            }
        }
        // copy data from host to device
        gpuErrchk(cudaMemcpy(dev_matA_halfK, host_matA_halfK, total_A_SIZE * sizeof(op_AB) / 2, cudaMemcpyHostToDevice));
    
        gpuErrchk(cudaMemcpy(dev_matB_halfK, host_matB_halfK, total_B_SIZE * sizeof(op_AB) / 2, cudaMemcpyHostToDevice));

        gpuErrchk(cudaMemcpy(dev_matC_halfK, host_matC, total_C_SIZE * sizeof(op_CD), cudaMemcpyHostToDevice));

        gemm_m16n8k4_kernel<<<BLOCKS_NUM, nwarps*warp_size>>>(dev_matA_halfK,dev_matB_halfK,dev_matC_halfK,dev_matD_halfK);
        gpuErrchk(cudaPeekAtLastError());
    
        gpuErrchk(cudaMemcpy(host_matD_halfK, dev_matD_halfK, total_C_SIZE * sizeof(op_CD), cudaMemcpyDeviceToHost));
        
        cudaDeviceSynchronize();
        for(int n = 0; n < inst_m*inst_n; n ++){
            host_matD_final_halfK[n] += host_matD_halfK[n];
        }

    }


    double l2_relative_err = 0.0;
    compute_diff_l2_norm(cpu_res_baseline,host_matD,inst_m,inst_n,l2_relative_err);


    double l2_relative_err_halfK = 0.0;
    compute_diff_l2_norm(cpu_res_baseline,host_matD_final_halfK,inst_m,inst_n,l2_relative_err_halfK);

    std::vector<double> res{l2_relative_err,l2_relative_err_halfK};

    return res;

}

int main(){
    std::cout<<"***********************************"<<std::endl;
    std::cout << "mma tf32 numeric errors w.r.t fp32 on cpu "  << std::endl;


    double l2_relative_err = 0.0;
    double l2_relative_err_halfK = 0.0;
    
    for(int i=0;i < ROUNDS; i ++){
        std::vector<double> errors = numeric_tf32_bench();
        l2_relative_err += errors[0];
        l2_relative_err_halfK += errors[1];
    }

    // std::cout<<"element-wise error :"<<std::endl;
    // std::cout<<"element-wise avg_abs_err = " << avg_abs_err/ITERS <<std::endl;
    // std::cout<<"element-wise avg_l1_norm_err = " << avg_l1_norm/ITERS <<std::endl;

    std::cout<<"mma.m16n8k8.tf32 l2 relative error :"<< l2_relative_err/ROUNDS << std::endl;
    std::cout<<"mma.m16n8k4.tf32 l2 relative error :"<< l2_relative_err_halfK/ROUNDS << std::endl;

    std::cout<<"mma.m16n8k8.tf32 l2 relative error per FMA :"<< l2_relative_err/(ROUNDS*inst_m*inst_k*inst_n) << std::endl;
    std::cout<<"mma.m16n8k4.tf32 l2 relative error per FMA :"<< l2_relative_err_halfK/(ROUNDS*inst_m*inst_k*inst_n) << std::endl;

}