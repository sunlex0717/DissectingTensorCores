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


// typedef float op_AB; 
// typedef float op_CD; 

// typedef float init_type;
// #define Init_With_TF32 1

typedef float init_type;
#define Init_With_TF32 0

#ifndef MEAN
#define MEAN (0.0)
#endif

#ifndef STDDEV
#define STDDEV (1.0)
#endif

#ifndef ITERS
#define ITERS  (1024 )
#endif

#define ROUNDS  (1000 )


#define SEED 123456

struct random_generator{
    // int seed_;
    // float mean_;
    // float stddev_;
    std::mt19937 gen;
    std::normal_distribution<float> random_gen{MEAN,STDDEV};
    //std::uniform_real_distribution<> random_gen{-1.0,1.0};
    random_generator(int seed_){
        gen.seed(seed_);
    }

    float operator()(){
        return random_gen(gen);
    };
};

// const int inst_m = 16;
const int inst_n = 8;
// const int inst_k = 8;

__forceinline__ __device__ unsigned lane_id_()
{
    unsigned ret; 
    asm volatile ("mov.u32 %0, %laneid;" : "=r"(ret));
    return ret;
}





__global__ void gemm_m16n8k8_kernel(float* MatA,float* MatB,float* MatC, float* MatD ){
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
    float frag_D[4]; // float , 16*8 /32 = 4*float
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

template<typename typeDest,typename typeSrc >
void copy_mat(typeDest* dest, typeSrc* source, int num_eles){
    for(int i=0;i<num_eles;i++){
        dest[i] = typeDest(source[i]);
    }
}



void gpu_tf32_m16n8k8(float* matA_in, float * matB_in, float* matC_in, float* matD_out){
    const int inst_m = 16;
    const int inst_n = 8;
    const int inst_k = 8;

    int BLOCKS_NUM = 1;
    int nwarps = 1;
    int warp_size = 32;
    
    unsigned total_A_SIZE = inst_m*inst_k*nwarps;
    unsigned total_B_SIZE = inst_k*inst_n*nwarps;
    unsigned total_C_SIZE = inst_m*inst_n*nwarps;

    //*********** m16n8k8 device mem allocation***//
    float *dev_matA;
    float *dev_matB;
    float *dev_matC;

    float *dev_matD;
    // allocate device global memory
    // D = A*B + C
    cudaMalloc(&dev_matA, total_A_SIZE * sizeof(float));
    cudaMalloc(&dev_matB, total_B_SIZE * sizeof(float));
    cudaMalloc(&dev_matC, total_C_SIZE * sizeof(float));
    cudaMalloc(&dev_matD, total_C_SIZE * sizeof(float));

    /***************************************** m16n8k8 *******************************************/
    // copy data from host to device
    gpuErrchk(cudaMemcpy(dev_matA, matA_in, total_A_SIZE * sizeof(float), cudaMemcpyHostToDevice));

    gpuErrchk(cudaMemcpy(dev_matB, matB_in, total_B_SIZE * sizeof(float), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(dev_matC, matC_in, total_C_SIZE * sizeof(float), cudaMemcpyHostToDevice));

    gemm_m16n8k8_kernel<<<BLOCKS_NUM, nwarps*warp_size>>>(dev_matA,dev_matB,dev_matC,dev_matD);
    gpuErrchk(cudaPeekAtLastError());

    gpuErrchk(cudaMemcpy(matD_out, dev_matD, total_C_SIZE * sizeof(float), cudaMemcpyDeviceToHost));
    cudaDeviceSynchronize();

    cudaFree(dev_matA);
    cudaFree(dev_matB);
    cudaFree(dev_matC);
    cudaFree(dev_matD);
}




std::vector<double> tf32_addtion_innerProduct(random_generator & random_gen){
    const int inst_m = 16;
    const int inst_n = 8;
    const int inst_k = 8;

    // int BLOCKS_NUM = 1;
    int nwarps = 1;
    //int warp_size = 32;
    unsigned total_A_SIZE = inst_m*inst_k*nwarps;
    unsigned total_B_SIZE = inst_k*inst_n*nwarps;
    unsigned total_C_SIZE = inst_m*inst_n*nwarps;

    // int num_chain = n;
    float *host_matA = (float *)malloc(total_A_SIZE * sizeof(float));
    float *host_matB = (float *)malloc(total_B_SIZE * sizeof(float));

    float *host_matC = (float *)malloc(total_C_SIZE * sizeof(float));
    float *host_matD = (float *)malloc(total_C_SIZE * sizeof(float));


    // initialize A, row-major
    float *host_matA_cpu = (float *)malloc(total_A_SIZE * sizeof(float));
    float *host_matB_cpu = (float *)malloc(total_B_SIZE * sizeof(float));

    float *host_matC_cpu = (float *)malloc(total_C_SIZE * sizeof(float));


    for(int r = 0; r < inst_m; r ++){
        for(int c = 0; c < inst_k; c ++){
            //float rnd = (float)(r*inst_k+c);


            host_matA_cpu[r*inst_k+c] = 0.0;
            host_matA[r*inst_k+c] = (float)0.0;
        }
    }


    float rnd = (float)random_gen();

    // std::cout<<rnd<<std::endl;

    #if Init_With_TF32 == 1
    uint32_t* tmp = reinterpret_cast<uint32_t*>(&rnd);
    *tmp = (*tmp  & ~0x1fff);
    #endif
    host_matA_cpu[0] = rnd;
    host_matA[0] = rnd;


    rnd = (float)random_gen();
    #if Init_With_TF32 == 1
    uint32_t* tmp2 = reinterpret_cast<uint32_t*>(&rnd);
    *tmp2 = (*tmp2  & ~0x1fff);
    #endif

    host_matA_cpu[1] = rnd;
    host_matA[1] = rnd;


    for(int r = 0; r < inst_k; r ++){
        for(int c = 0; c < inst_n; c ++){
            float rnd = 0.0;
            //float rnd = float(r*inst_n+c);   //(float)random_gen(gen);
   
            host_matB_cpu[r*inst_n+c] = rnd;
            host_matB[r*inst_n+c] = (float)rnd;
        }
    }





    
    host_matB_cpu[0] = 1.0;
    host_matB[0] = 1.0;
    
    host_matB[inst_n] = 1.0;
    host_matB_cpu[inst_n] = 1.0;

    // initialize C, row-major
    for(int r = 0; r < inst_m; r ++){
        for(int c = 0; c < inst_n; c ++){
            float rnd = 0.0;  //(float)random_gen(gen);
            //float rnd = float(r*inst_n+c);   //(float)random_gen(gen);
            //host_matB_cpu[r*inst_n+c] = rnd;
            host_matC_cpu[r*inst_n+c] = (float)rnd;
            host_matC[r*inst_n+c] = float(rnd); //0.0 ;//  (op_CD)random_gen(gen);
        }
    }

    float *cpu_res_baseline = (float *)malloc(total_C_SIZE * sizeof(float));
    float *gpu_m16n8k8 = (float *)malloc(total_C_SIZE * sizeof(float));
    float *gpu_m16n8k4 = (float *)malloc(total_C_SIZE * sizeof(float));
    // host computation
    gemm_mnk_cpu(host_matA_cpu,host_matB_cpu,host_matC_cpu,cpu_res_baseline,inst_m,inst_n,inst_k);

    //gpu 
    // gpu tf32 m16b8k8
    gpu_tf32_m16n8k8(host_matA,host_matB,host_matC,gpu_m16n8k8); //(float* matA_in, float * matB_in, float* matC_in, float* matD_out)

    double l2_relative_err = 0.0;
    compute_diff_l2_norm(cpu_res_baseline,gpu_m16n8k8,inst_m,inst_n,l2_relative_err);


    double abs_err = 0.0;
    abs_err = std::abs(double(cpu_res_baseline[0]) - double(gpu_m16n8k8[0]));

    std::vector<double> res{l2_relative_err,abs_err};

    return res;
}




std::vector<double> tf32_addtion_accumulation(random_generator & random_gen){
    const int inst_m = 16;
    const int inst_n = 8;
    const int inst_k = 8;

    // int BLOCKS_NUM = 1;
    int nwarps = 1;
    //int warp_size = 32;
    unsigned total_A_SIZE = inst_m*inst_k*nwarps;
    unsigned total_B_SIZE = inst_k*inst_n*nwarps;
    unsigned total_C_SIZE = inst_m*inst_n*nwarps;

    // int num_chain = n;
    float *host_matA = (float *)malloc(total_A_SIZE * sizeof(float));
    float *host_matB = (float *)malloc(total_B_SIZE * sizeof(float));

    float *host_matC = (float *)malloc(total_C_SIZE * sizeof(float));
    float *host_matD = (float *)malloc(total_C_SIZE * sizeof(float));


    // initialize A, row-major
    float *host_matA_cpu = (float *)malloc(total_A_SIZE * sizeof(float));
    float *host_matB_cpu = (float *)malloc(total_B_SIZE * sizeof(float));

    float *host_matC_cpu = (float *)malloc(total_C_SIZE * sizeof(float));


    for(int r = 0; r < inst_m; r ++){
        for(int c = 0; c < inst_k; c ++){
            //float rnd = (float)(r*inst_k+c);


            host_matA_cpu[r*inst_k+c] = 0.0;
            host_matA[r*inst_k+c] = (float)0.0;
        }
    }

    for(int r = 0; r < inst_k; r ++){
        for(int c = 0; c < inst_n; c ++){
            //float rnd = float(r*inst_n+c);   //(float)random_gen(gen);
            host_matB_cpu[r*inst_n+c] = 0.0;
            host_matB[r*inst_n+c] = (float)0.0;
        }
    }

    float rnd = (float)random_gen();
    #if Init_With_TF32 == 1
    uint32_t* tmp = reinterpret_cast<uint32_t*>(&rnd);
    *tmp = (*tmp  & ~0x1fff);
    #endif
    host_matA_cpu[0] = rnd;
    host_matA[0] = rnd;


    host_matB_cpu[0] = 1.0;
    host_matB[0] = 1.0;

    // initialize C, row-major
    for(int r = 0; r < inst_m; r ++){
        for(int c = 0; c < inst_n; c ++){
            
            //float rnd = float(r*inst_n+c);   //(float)random_gen(gen);
            //host_matB_cpu[r*inst_n+c] = rnd;
            host_matC_cpu[r*inst_n+c] = 0.0;
            host_matC[r*inst_n+c] = 0.0; //0.0 ;//  (op_CD)random_gen(gen);
        }
    }



    rnd = (float )random_gen();

    // std::cout << "random val fp32 " << rnd << std::endl;

    #if Init_With_TF32 == 1
    uint32_t* tmpc = reinterpret_cast<uint32_t*>(&rnd);
    *tmpc = (*tmpc  & ~0x1fff);
    #endif

    host_matC_cpu[0] = rnd;
    host_matC[0] = rnd;

    
    // std::cout << "convert to tf32 " << rnd << std::endl;



    float *cpu_res_baseline = (float *)malloc(total_C_SIZE * sizeof(float));
    float *gpu_m16n8k8 = (float *)malloc(total_C_SIZE * sizeof(float));
    float *gpu_m16n8k4 = (float *)malloc(total_C_SIZE * sizeof(float));
    // host computation
    gemm_mnk_cpu(host_matA_cpu,host_matB_cpu,host_matC_cpu,cpu_res_baseline,inst_m,inst_n,inst_k);

    //gpu 
    // gpu tf32 m16b8k8
    gpu_tf32_m16n8k8(host_matA,host_matB,host_matC,gpu_m16n8k8); //(float* matA_in, float * matB_in, float* matC_in, float* matD_out)

    double l2_relative_err = 0.0;
    compute_diff_l2_norm(cpu_res_baseline,gpu_m16n8k8,inst_m,inst_n,l2_relative_err);


    double abs_err = 0.0;
    abs_err = std::abs(double(cpu_res_baseline[0]) - double(gpu_m16n8k8[0]));

    std::vector<double> res{l2_relative_err,abs_err};

    return res;
}


int main(){
    

    // std::cout<<"***********************************"<<std::endl;
    // if(Init_With_TF32 == 1){
    //     std::cout<<"Initialization with tf32"<<std::endl;
    // }else{
    //     std::cout<<"Initialization with fp32"<<std::endl;
    // }
    // double l2_relative_err = 0.0;
    // double abs_err = 0.0;
    // for(int i=0;i < ROUNDS; i ++){
    //     std::vector<double> errors = tf32_mul();
    //     l2_relative_err += errors[0];
    //     abs_err+= errors[1];
    // }

    // std::cout<<"mma tf32 with accum fp32 | multiplication | abs numeric errors w.r.t fp32 on CPU = "<< abs_err/ROUNDS << std::endl;
    

    std::cout<<"***********************************"<<std::endl;
    if(Init_With_TF32 == 1){
        std::cout<<"Initialization with tf32"<<std::endl;
    }else{
        std::cout<<"Initialization with fp32"<<std::endl;
    }

    double l2_relative_err = 0.0;
    double abs_err= 0.0;
    random_generator random_gen(SEED);
    for(int i=0;i < ROUNDS; i ++){
        std::vector<double> errors = tf32_addtion_innerProduct(random_gen);
        l2_relative_err += errors[0];
        abs_err += errors[1];
        // l2_relative_err_halfK += errors[1];
    }
    // std::cout<<"mma tf32 with accum fp32 | addition of Inner Product | l2 numeric errors w.r.t fp32 = "<< l2_relative_err/ROUNDS << std::endl;
    std::cout<<"mma tf32 with accum fp32 | addition of Inner Product | abs numeric errors w.r.t fp32 = "<< abs_err/ROUNDS << std::endl;
    //std::cout<<"mma.m16n8k8.bf16 l2 relative error per FMA :"<< l2_relative_err/(ROUNDS*inst_m*inst_k*inst_n) << std::endl;

    std::cout<<"***********************************"<<std::endl;
    l2_relative_err = 0.0;
    abs_err= 0.0;
    random_generator random_gen2(SEED);
    for(int i=0;i < ROUNDS; i ++){
        std::vector<double> errors = tf32_addtion_accumulation(random_gen2);

        // if(errors[2] == 1){ //l2 is inf
        //     i --;
        //     continue;
        // }
        l2_relative_err += errors[0];
        abs_err += errors[1];
        // l2_relative_err_halfK += errors[1];
    }
    // std::cout<<"mma tf32 with accum fp32 | addition of accumulation | l2 numeric errors w.r.t fp32 ="<< l2_relative_err/ROUNDS << std::endl;
    std::cout<<"mma tf32 with accum fp32 | addition of accumulation | abs numeric errors w.r.t fp32 = "<< abs_err/ROUNDS << std::endl;

}



/************ chain matmul pseudocode  **************/







/************ chain matmul pseudocode **************/