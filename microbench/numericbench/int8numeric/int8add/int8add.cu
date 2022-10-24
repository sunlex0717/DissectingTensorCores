#include <algorithm>
#include <cuda.h>
#include <iostream>
#include <mma.h>
#include <stdio.h>
#include <stdlib.h>
#include <cstdlib>
#include <cuda_fp16.h>
#include <random>
#include <cmath>
#include "../../../hw_def/hw_def.h"
// #include "../../cpu_base.h"
#include "../../cpu_int_base.h"
// typedef float op_AB; 
// typedef float op_CD; 


// typedef half init_type;
// #define Init_With_FP16 1

typedef float init_type;
#define Init_With_FP16 0

// #ifndef MEAN
// #define MEAN (0.0)
// #endif

// #ifndef STDDEV
// #define STDDEV (1.0)
// #endif



#ifndef ITERS
#define ITERS  (1024 )
#endif

#define ROUNDS  (ITERS )

// #ifndef CHAINs
// #define CHAINs  (50)
// #endif


#define SEED 123456

const static int random_int_lower = INT8_MIN;
const static int random_int_upper = INT8_MAX;

struct random_generator{
    // int seed_;
    // float mean_;
    // float stddev_;
    std::mt19937 gen;
    //std::normal_distribution<float> random_gen{MEAN,STDDEV};
    std::uniform_int_distribution<> random_gen{random_int_lower,random_int_upper};
    //std::uniform_real_distribution<> random_gen{-1.0,1.0};
    random_generator(int seed_){
        gen.seed(seed_);
    }

    int operator()(){
        return random_gen(gen);
    };
};



const int inst_n = 8;
// const int inst_k = 8;

__forceinline__ __device__ unsigned lane_id_()
{
    unsigned ret; 
    asm volatile ("mov.u32 %0, %laneid;" : "=r"(ret));
    return ret;
}

__global__ void gemm_m16n8k16_s8_int32accum_kernel(int8_t* MatA,int8_t* MatB,int32_t* MatC, int32_t* MatD ){
    constexpr const int inst_k = 8;

    uint32_t lane_id =  lane_id_();
    uint32_t group_id = lane_id >>2;
    uint32_t tid_in_group = lane_id % 4;
    // m16 n8 k16
    int8_t frag_A[8]; // 16 * 16  / 32 = 8 * bf16
    int8_t frag_B[4]; // 8 * 16  / 32
    int32_t frag_D[4]; // float , 16*8 /32 = 4*float
    // load operand fragA
    #pragma unroll
    for(int i =0; i < 8; i++){
        uint32_t row_a = 0;
        if( i<4 ){
            row_a = group_id;
        }else{
            row_a = group_id + 8;
        }

        uint32_t col_a = (tid_in_group * 4) + (i & 0x3);
        // row major
        frag_A[i] = MatA[inst_k*row_a + col_a];
        
    }
    // load operand fragB, MatB has to be col-major
    #pragma unroll
    for(int i =0; i < 4; i++){
        uint32_t row_b =  (tid_in_group * 4) + (i);

        uint32_t col_b = group_id;
        // row-major B
        frag_B[i] = MatB[row_b*inst_n + col_b];
    }

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

    uint32_t const *A = reinterpret_cast<uint32_t const *>(&frag_A[0]);
    uint32_t const *B = reinterpret_cast<uint32_t const *>(&frag_B[0]);//?
    int32_t *C = reinterpret_cast<int32_t *>(&frag_D[0]);
    int32_t *D = C;  // D = A*B + D.

    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 "
        "{%0,%1,%2,%3}, {%4,%5}, {%6}, {%7,%8,%9,%10};\n"
        : "=r"(D[0]), "=r"(D[1]), "=r"(D[2]), "=r"(D[3])
        : "r"(A[0]), "r"(A[1]), // "r"(A[2]), "r"(A[3]), 
          "r"(B[0]), //"r"(B[1]),
          "r"(C[0]), "r"(C[1]), "r"(C[2]), "r"(C[3])
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





void gpu_int8_int32accum_m16n8k16(int8_t* matA_in, int8_t * matB_in, int32_t* matC_in, int32_t* matD_out){

    const int inst_m = 16;
    const int inst_n = 8;
    const int inst_k = 16;

    int BLOCKS_NUM = 1;
    int nwarps = 1;
    int warp_size = 32;
    
    unsigned total_A_SIZE = inst_m*inst_k*nwarps;
    unsigned total_B_SIZE = inst_k*inst_n*nwarps;
    unsigned total_C_SIZE = inst_m*inst_n*nwarps;
    int8_t *dev_matA;
    int8_t *dev_matB;
    int32_t *dev_matC;

    int32_t *dev_matD;
    // allocate device global memory
    // D = A*B + C
    cudaMalloc(&dev_matA, total_A_SIZE * sizeof(int8_t));
    cudaMalloc(&dev_matB, total_B_SIZE * sizeof(int8_t));
    cudaMalloc(&dev_matC, total_C_SIZE * sizeof(int32_t));
    cudaMalloc(&dev_matD, total_C_SIZE * sizeof(int32_t));

    /***************************************** m16n8k16 *******************************************/
    // copy data from host to device
    gpuErrchk(cudaMemcpy(dev_matA, matA_in, total_A_SIZE * sizeof(int8_t), cudaMemcpyHostToDevice));

    gpuErrchk(cudaMemcpy(dev_matB, matB_in, total_B_SIZE * sizeof(int8_t), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(dev_matC, matC_in, total_C_SIZE * sizeof(int32_t), cudaMemcpyHostToDevice));

    gemm_m16n8k16_s8_int32accum_kernel<<<BLOCKS_NUM, nwarps*warp_size>>>(dev_matA,dev_matB,dev_matC,dev_matD);
    gpuErrchk(cudaPeekAtLastError());

    gpuErrchk(cudaMemcpy(matD_out, dev_matD, total_C_SIZE * sizeof(int32_t), cudaMemcpyDeviceToHost));
    cudaDeviceSynchronize();

    cudaFree(dev_matA);
    cudaFree(dev_matB);
    cudaFree(dev_matC);
    cudaFree(dev_matD);
}




template<typename typeDest,typename typeSrc >
void copy_mat(typeDest* dest, typeSrc* source, int num_eles){
    for(int i=0;i<num_eles;i++){
        dest[i] = typeDest(source[i]);
    }
}


std::vector<double> int8_addtion_innerProduct( random_generator &random_gen){

    const int inst_m = 16;
    const int inst_n = 8;
    const int inst_k = 16;
    // int BLOCKS_NUM = 1;
    int nwarps = 1;
    //int warp_size = 32;
    unsigned total_A_SIZE = inst_m*inst_k*nwarps;
    unsigned total_B_SIZE = inst_k*inst_n*nwarps;
    unsigned total_C_SIZE = inst_m*inst_n*nwarps;

    int8_t *host_matA = (int8_t *)malloc(total_A_SIZE * sizeof(int8_t));
    int8_t *host_matB = (int8_t *)malloc(total_B_SIZE * sizeof(int8_t));

    int32_t *host_matC = (int32_t *)malloc(total_C_SIZE * sizeof(int32_t));
    int32_t *host_matD = (int32_t *)malloc(total_C_SIZE * sizeof(int32_t));

    // unsigned seed = 123456; //std::chrono::system_clock::now().time_since_epoch().count();
    // std::mt19937 gen(seed); //
    // //std::default_random_engine gen (seed); 
    // //std::normal_distribution<float> random_gen{0,1.0};
    // std::uniform_real_distribution<> random_gen{-1.0,1.0};
    // initialize A, row-major
    int8_t *host_matA_cpu = (int8_t *)malloc(total_A_SIZE * sizeof(int8_t));
    int8_t *host_matB_cpu = (int8_t *)malloc(total_B_SIZE * sizeof(int8_t));

    int32_t *host_matC_cpu = (int32_t *)malloc(total_C_SIZE * sizeof(int32_t));

    // initialize MatA
    // set the the first element as no zero, others are zeros
    for(int r = 0; r < inst_m; r ++){
        for(int c = 0; c < inst_k; c ++){
            //float rnd = (float)(r*inst_k+c);
            // float rnd = (init_type )random_gen(gen);
            host_matA_cpu[r*inst_k+c] = 0;
            host_matA[r*inst_k+c] = 0;
        }
    }
    int8_t rnd = (int8_t )random_gen();
    host_matA_cpu[0] = rnd;
    host_matA[0] = rnd;

    rnd = (int8_t )random_gen();
    host_matA_cpu[1] = rnd;
    host_matA[1] = rnd;
    // initialize MatB
    for(int r = 0; r < inst_k; r ++){
        for(int c = 0; c < inst_n; c ++){
            //float rnd = (init_type )random_gen(gen);
            //float rnd = float(r*inst_n+c);   //(float)random_gen(gen);
            host_matB_cpu[r*inst_n+c] = 0;
            host_matB[r*inst_n+c] = 0;
        }
    }
    // rnd = (init_type )random_gen(gen);
    host_matB_cpu[0] = 1;
    host_matB[0] = 1;

    host_matB_cpu[inst_n] = 1;
    host_matB[inst_n] =1;
    // initialize MatC as 0.0
    for(int r = 0; r < inst_m; r ++){
        for(int c = 0; c < inst_n; c ++){
            // float rnd = 0.0;  //(float)random_gen(gen);
            //float rnd = float(r*inst_n+c);   //(float)random_gen(gen);
            //host_matB_cpu[r*inst_n+c] = rnd;
            host_matC_cpu[r*inst_n+c] = 0;
            host_matC[r*inst_n+c] = 0; //0.0 ;//  (op_CD)random_gen(gen);
        }
    }



    int32_t *cpu_res_baseline = (int32_t *)malloc(total_C_SIZE * sizeof(int32_t));
    int32_t *gpu_m16n8k16_int32accum = (int32_t *)malloc(total_C_SIZE * sizeof(int32_t));
    


    //float *gpu_m16n8k4 = (op_CD *)malloc(total_C_SIZE * sizeof(op_CD));
    // host computation
    gemm_mnk_cpu(host_matA_cpu,host_matB_cpu,host_matC_cpu,cpu_res_baseline,inst_m,inst_n,inst_k);

    gpu_int8_int32accum_m16n8k16(host_matA,host_matB,host_matC,gpu_m16n8k16_int32accum);

    

    double l2_relative_err = 0.0;
    double abs_err = 0.0;
    compute_diff_l2_norm(&cpu_res_baseline[0],&gpu_m16n8k16_int32accum[0],1,1,l2_relative_err);



    double abs_err_fp16accum = 0.0;
    double abs_err_fp16accum_cvt_half = 0.0;
    abs_err = std::abs(double(cpu_res_baseline[0]) - double(gpu_m16n8k16_int32accum[0]));
    abs_err_fp16accum = std::abs(double( (cpu_res_baseline[0]) ) - double(gpu_m16n8k16_int32accum[0]));
    abs_err_fp16accum_cvt_half = std::abs(double( half(cpu_res_baseline[0]) ) - double(gpu_m16n8k16_int32accum[0]));
    // double l2_relative_err_halfK = 0.0;
    // compute_diff_l2_norm(cpu_res_baseline,host_matD_final_halfK,inst_m,inst_n,l2_relative_err_halfK);
    std::vector<double> res{l2_relative_err,abs_err,abs_err_fp16accum,abs_err_fp16accum_cvt_half};
    return res;
}



std::vector<double> int8_addtion_accumulation(random_generator &random_gen){

    const int inst_m = 16;
    const int inst_n = 8;
    const int inst_k = 16;
    // int BLOCKS_NUM = 1;
    int nwarps = 1;
    //int warp_size = 32;
    unsigned total_A_SIZE = inst_m*inst_k*nwarps;
    unsigned total_B_SIZE = inst_k*inst_n*nwarps;
    unsigned total_C_SIZE = inst_m*inst_n*nwarps;

    int8_t *host_matA = (int8_t *)malloc(total_A_SIZE * sizeof(int8_t));
    int8_t *host_matB = (int8_t *)malloc(total_B_SIZE * sizeof(int8_t));

    int32_t *host_matC = (int32_t *)malloc(total_C_SIZE * sizeof(int32_t));
    int32_t *host_matD = (int32_t *)malloc(total_C_SIZE * sizeof(int32_t));

    // unsigned seed = 123456; //std::chrono::system_clock::now().time_since_epoch().count();
    // std::mt19937 gen(seed); //
    // //std::default_random_engine gen (seed); 
    // //std::normal_distribution<float> random_gen{0,1.0};
    // std::uniform_real_distribution<> random_gen{-1.0,1.0};
    // initialize A, row-major
    int8_t *host_matA_cpu = (int8_t *)malloc(total_A_SIZE * sizeof(int8_t));
    int8_t *host_matB_cpu = (int8_t *)malloc(total_B_SIZE * sizeof(int8_t));

    int32_t *host_matC_cpu = (int32_t *)malloc(total_C_SIZE * sizeof(int32_t));

    // initialize MatA
    // set the the first element as no zero, others are zeros
    for(int r = 0; r < inst_m; r ++){
        for(int c = 0; c < inst_k; c ++){
            //float rnd = (float)(r*inst_k+c);
            // float rnd = (init_type )random_gen();
            host_matA_cpu[r*inst_k+c] = 0;
            host_matA[r*inst_k+c] = 0;
        }
    }
    int8_t rnd = (int8_t )random_gen();
    //std::cout<<"rand val" << rnd <<std::endl;
    host_matA_cpu[0] = rnd;
    host_matA[0] = rnd;


    // initialize MatB
    for(int r = 0; r < inst_k; r ++){
        for(int c = 0; c < inst_n; c ++){
            //float rnd = (init_type )random_gen(gen);
            //float rnd = float(r*inst_n+c);   //(float)random_gen(gen);
            host_matB_cpu[r*inst_n+c] = 0;
            host_matB[r*inst_n+c] = 0;
        }
    }
    // rnd = (init_type )random_gen(gen);
    host_matB_cpu[0] = 1;
    host_matB[0] = 1;
    // initialize MatC as 0.0
    for(int r = 0; r < inst_m; r ++){
        for(int c = 0; c < inst_n; c ++){
            // float rnd = 0.0;  //(float)random_gen(gen);
            //float rnd = float(r*inst_n+c);   //(float)random_gen(gen);
            //host_matB_cpu[r*inst_n+c] = rnd;
            host_matC_cpu[r*inst_n+c] =0;
            host_matC[r*inst_n+c] = 0; //0.0 ;//  (op_CD)random_gen(gen);
        }
    }


    rnd = (int8_t )random_gen();
    host_matC_cpu[0] = rnd;
    host_matC[0] = rnd;



    int32_t *cpu_res_baseline = (int32_t *)malloc(total_C_SIZE * sizeof(int32_t));
    int32_t *gpu_m16n8k16_int32accum = (int32_t *)malloc(total_C_SIZE * sizeof(int32_t));
    


    //float *gpu_m16n8k4 = (op_CD *)malloc(total_C_SIZE * sizeof(op_CD));
    // host computation
    gemm_mnk_cpu(host_matA_cpu,host_matB_cpu,host_matC_cpu,cpu_res_baseline,inst_m,inst_n,inst_k);

    gpu_int8_int32accum_m16n8k16(host_matA,host_matB,host_matC,gpu_m16n8k16_int32accum);

    

    double l2_relative_err = 0.0;
    double abs_err = 0.0;
    compute_diff_l2_norm(&cpu_res_baseline[0],&gpu_m16n8k16_int32accum[0],1,1,l2_relative_err);



    double abs_err_fp16accum = 0.0;
    double abs_err_fp16accum_cvt_half = 0.0;
    abs_err = std::abs(double(cpu_res_baseline[0]) - double(gpu_m16n8k16_int32accum[0]));
    abs_err_fp16accum = std::abs(double( (cpu_res_baseline[0]) ) - double(gpu_m16n8k16_int32accum[0]));
    abs_err_fp16accum_cvt_half = std::abs(double( half(cpu_res_baseline[0]) ) - double(gpu_m16n8k16_int32accum[0]));
    // double l2_relative_err_halfK = 0.0;
    // compute_diff_l2_norm(cpu_res_baseline,host_matD_final_halfK,inst_m,inst_n,l2_relative_err_halfK);

    // int is_inf = 0;
    // l2_is_inf = std::isinf(l2_relative_err);
    std::vector<double> res{l2_relative_err,abs_err,abs_err_fp16accum,abs_err_fp16accum_cvt_half,};
    return res;
}




int main(){
    // const int inst_m = 16;
    // const int inst_n = 8;
    // const int inst_k = 8;
    std::cout<<"***********************************"<<std::endl;
    if(Init_With_FP16 == 1){
        std::cout<<"Initialization with fp16"<<std::endl;
    }else{
        std::cout<<"Initialization with fp32"<<std::endl;
    }
    //std::cout << "mma bf16 multiplication numeric errors w.r.t fp32"  << std::endl;

    double l2_relative_err = 0.0;
    double abs_err = 0.0;
    double abs_err_fp16accum = 0.0;
    double abs_err_fp16accum_cvt_half = 0.0;
    random_generator random_gen(SEED);
    for(int i=0;i < ROUNDS; i ++){
        std::vector<double> errors = int8_addtion_innerProduct(random_gen);
        l2_relative_err += errors[0];
        abs_err += errors[1];
        abs_err_fp16accum += errors[2];
        abs_err_fp16accum_cvt_half += errors[3];
        // l2_relative_err_halfK += errors[1];
    }
    //std::cout<<"mma fp16 multiplication l2 numeric errors w.r.t fp32 = "<< l2_relative_err/ROUNDS << std::endl;
    std::cout<<"mma fp16 with accum fp32 | addtion of inner product |  abs numeric errors w.r.t fp32 on CPU = "<< abs_err/ROUNDS << std::endl;
    std::cout<<"mma fp16 with accum fp16 | addtion of inner product |  abs numeric errors w.r.t fp32 on CPU = "<< abs_err_fp16accum/ROUNDS << std::endl;
    std::cout<<"mma fp16 with accum fp16 | addtion of inner product |  abs numeric errors w.r.t fp32 on CPU coverted to half = "<< double(abs_err_fp16accum_cvt_half/ROUNDS) << std::endl;
    //std::cout<<"mma.m16n8k8.bf16 l2 relative error per FMA :"<< l2_relative_err/(ROUNDS*inst_m*inst_k*inst_n) << std::endl;

    l2_relative_err = 0.0;
    abs_err = 0.0;
    abs_err_fp16accum = 0.0;
    abs_err_fp16accum_cvt_half = 0.0;
    random_generator random_gen2(SEED);
    for(int i=0;i < ROUNDS; i ++){
        std::vector<double> errors = int8_addtion_accumulation(random_gen2);
        l2_relative_err += errors[0];
        abs_err += errors[1];
        abs_err_fp16accum += errors[2];
        abs_err_fp16accum_cvt_half += errors[3];
        // l2_relative_err_halfK += errors[1];
    }
    std::cout<<"***********************************"<<std::endl;
    //std::cout<<"mma fp16 multiplication l2 numeric errors w.r.t fp32 = "<< l2_relative_err/ROUNDS << std::endl;
    std::cout<<"mma fp16 with accum fp32 | addtion of accumulation |  abs numeric errors w.r.t fp32 on CPU = "<< abs_err/ROUNDS << std::endl;
    std::cout<<"mma fp16 with accum fp16 | addtion of accumulation |  abs numeric errors w.r.t fp32 on CPU = "<< abs_err_fp16accum/ROUNDS << std::endl;
    std::cout<<"mma fp16 with accum fp16 | addtion of accumulation |  abs numeric errors w.r.t fp32 on CPU coverted to half = "<< double(abs_err_fp16accum_cvt_half/ROUNDS) << std::endl;
    
}


// // Define initialization data type
// typedef init_type fp16; // fp16 or fp32
// // Define Matrix C/D data type
// typedef C_type fp16; // fp16 or fp32
// // Initialization
// initialize_random<init_type>(A_tmp);
// initialize_random<init_type>(B_tmp);
// initialize_random<init_type>(C_tmp);
// // Copy to CPU
// Copy_CPU<fp32>(A_CPU,A_tmp);
// Copy_CPU<fp32>(B_CPU,B_tmp);
// Copy_CPU<fp32>(C_CPU,C_tmp);
// // Copy to GPU
// Copy_GPU<fp16>(A_TC,A_tmp);
// Copy_GPU<fp16>(B_TC,B_tmp);
// Copy_GPU<C_type>(C_TC,C_tmp);
// // CPU computation
// D_CPU_fp32 = A_CPU*B_CPU+C_CPU;
// // Tensor Cores computation
// D_TC = A_TC*B_TC + C_TC;
// // Convert FP32 CPU resilt to FP16
// D_CPU_fp16 = fp16(D_CPU_fp32);
// // Compute errors
// abs_err_TC_CPUfp32 = abs(D_CPU_fp32 - D_TC);
// abs_err_TC_CPUfp16 = abs(D_CPU_fp16 - D_TC);