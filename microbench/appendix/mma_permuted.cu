//permuted data layout to reduce bank conflicts
//Reference: https://www.nvidia.com/en-us/on-demand/session/gtcsj20-s21745/
// TODO: fixed tile size and analyze how to apply the premuted shared memory layout introduced in the slide


#include <algorithm>
#include <cuda.h>
#include <iostream>
#include <mma.h>
#include <stdio.h>
#include <stdlib.h>
#include <cstdlib>
#include "../../hw_def/hw_def.h"
#include <vector>
#include <experimental/random>
#include <cooperative_groups/memcpy_async.h>
#include <cuda/pipeline>


#define checkKernelErrors(expr)                             \
  do {                                                      \
    expr;                                                   \
                                                            \
    cudaError_t __err = cudaGetLastError();                 \
    if (__err != cudaSuccess) {                             \
      printf("Line %d: '%s' failed: %s\n", __LINE__, #expr, \
             cudaGetErrorString(__err));                    \
      abort();                                              \
    }                                                       \
  } while (0)


const static bool CPU_DEBUG = false;
const static unsigned int Profiling_ROUNDS = 1000;
//#define DEBUG_KERNEL

const static unsigned int M_GLOBAL = 2048;
const static unsigned int N_GLOBAL = 2048;
const static unsigned int K_GLOBAL = 2048;


const static unsigned int BLOCK_Tile_M = 64;
const static unsigned int BLOCK_Tile_N = 64;
const static unsigned int BLOCK_Tile_K = 64;

const static unsigned int NUM_BlockTile_M = M_GLOBAL/BLOCK_Tile_M;
const static unsigned int NUM_BlockTile_N = N_GLOBAL/BLOCK_Tile_N;

const static unsigned int WARP_Tile_M = 32;
const static unsigned int WARP_Tile_N = 32;

//const static unsigned int WARP_M_ratio = WARP_Tile_M/(WARP_Tile_M +WARP_Tile_N );
const static unsigned int SKEW_BF16 = 0;

//m16n8k16
const static unsigned int MMA_Tile_M = 16;
const static unsigned int MMA_Tile_N = 8;
const static unsigned int MMA_Tile_K = 16;
//16 8 16


const static unsigned int NUM_warp_m_per_block = BLOCK_Tile_M/WARP_Tile_M;
const static unsigned int NUM_warp_n_per_block = BLOCK_Tile_N/WARP_Tile_N;
const static unsigned int NUM_warp_block = NUM_warp_m_per_block * NUM_warp_n_per_block;


const static unsigned int Block_TileK_Bytes = BLOCK_Tile_K *  sizeof(__nv_bfloat16);

const static unsigned int Warp_TileK_Copy_Bytes = 32 * sizeof(int4);

const static unsigned int Block_TileK_Copy_Lines_Per_Warp = Warp_TileK_Copy_Bytes/Block_TileK_Bytes;

const static unsigned int Block_TileK_Copy_Line_LANEs = (32/Block_TileK_Copy_Lines_Per_Warp); // every Block_TileK_Copy_Line_LANEs threads copy one Block_TileK

using namespace nvcuda;
using namespace cooperative_groups;
// using namespace std;

inline __device__ void check_fragementA(__nv_bfloat16* fragementA,int laneid,__nv_bfloat16* fragementA_global_vis){
    unsigned int groupId = laneid >> 2;
    unsigned int threadID_in_group = laneid % 4;
    
    for(int i=0;i<8;i++){
        int row =0;
        int col = 0;
        if( (i>=0 && i<2) || (i>=4 && i<6) ){
            row = groupId;
        }else{
            row = groupId + 8;
        }

        col = i <4? ((threadID_in_group * 2) + (i & 0x1)) : ((threadID_in_group * 2) + (i & 0x1) + 8);
        fragementA_global_vis[row *16 + col] = fragementA[i];
    }
    __syncthreads();
    if(laneid==0){
        for(int i=0;i<16;i++){
            for(int j=0;j<16;j++){
                printf("%-6.2f ",float(fragementA_global_vis[i *16 + j]));
            }
            printf("\n");
        }
    }
    

}

// borrow from cutlass

inline __device__ size_t permut_idx(const unsigned int contiguous,const unsigned int strided,const unsigned int stride_size){

    size_t tc = contiguous / 8;
    size_t ts = strided / 4;

    size_t c = contiguous % 8;
    size_t s = strided % 4;

    size_t k_index = (c / 2);

    size_t bank = (((c & 1) * 4) | (s ^ k_index));

    size_t offset = tc * 8 + bank + (ts * 4 + k_index) * stride_size;

    return offset;
}


__global__ void 
mma_permuted(const __nv_bfloat16* __restrict__   matA, const __nv_bfloat16* __restrict__  matB, 
            float* matD, long long int* d_gpu_clock /*, __nv_bfloat16* fragementA_global_vis*/)
{
    auto this_grid = cooperative_groups::this_grid();
    auto this_block = cooperative_groups::this_thread_block();
    auto this_tile = tiled_partition<32>(this_block);
    

    //const size_t shmem_idx_b_off = BLOCK_Tile_M*BLOCK_Tile_K;

    const unsigned int warpId = threadIdx.x / 32;
    const unsigned int laneId = threadIdx.x % 32;

    extern __shared__ __nv_bfloat16 buffer[][BLOCK_Tile_K + SKEW_BF16];
    // __nv_bfloat16* matA_tile = (__nv_bfloat16*)&buffer[0][0]; // size of matA tile = BLOCK_Tile_M * BLOCK_Tile_K
    // __nv_bfloat16* matB_tile = (__nv_bfloat16*)&matA_tile[BLOCK_Tile_M*BLOCK_Tile_K];

    const unsigned int warpId_m = warpId / (NUM_warp_n_per_block);
    const unsigned int warpId_n = warpId % (NUM_warp_n_per_block);

    long long int start_t =  clock64();

    float fragementD[(WARP_Tile_M/16)] [(WARP_Tile_N/8)][4] = {};


    for(unsigned int block_pos = blockIdx.x;; block_pos += gridDim.x){

        const unsigned int block_tile_m = block_pos/NUM_BlockTile_N;
        const unsigned int block_tile_n = block_pos % NUM_BlockTile_N;

        if(block_tile_m >= NUM_BlockTile_M){ // boundary check
            break;
        }




        // copy A 

        // const __nv_bfloat16 *warp_ptr = (warpId < (WARPS_PER_BLOCK/2)) ? (&A[block_tile_i * M * K_GLOBAL] + M * K_GLOBAL * (warpId % (WARPS_PER_BLOCK/2)) * 2) :
        //                                       (&B[block_tile_j * N * K_GLOBAL] + N * K_GLOBAL * (warpId % (WARPS_PER_BLOCK/2)) * 2);

  
        // const __nv_bfloat16 *block_ptr_matA = &matA[block_tile_m * BLOCK_Tile_M * K_GLOBAL];
        // const __nv_bfloat16 *block_ptr_matB = &matB[block_tile_n * BLOCK_Tile_N * K_GLOBAL];
        
        //const __nv_bfloat16 *warp_ptr_matA_offset = block_ptr_matA + 
        for(int blk_tile_k =0; blk_tile_k < (K_GLOBAL/BLOCK_Tile_K);blk_tile_k++){


            // streaming matA to MatTileA size Block_Tile_M * Block_Tile_K
            // earch warp can copy Block_TileK_Copy_Lines_Per_Warp * BLOCK_Tile_M, we need BLOCK_Tile_M lines in total for MatA
            // 
            // we have NUM_warp_block, so each warp needs to do BLOCK_Tile_M/ (NUM_warp_block *Block_TileK_Copy_Lines_Per_Warp ) iteartions
            /******************************* shared mem store************************************/
            const unsigned int num_iters_ATile = BLOCK_Tile_M/ (NUM_warp_block *Block_TileK_Copy_Lines_Per_Warp );
            const unsigned int num_lanes_workload_per_warp = BLOCK_Tile_M/NUM_warp_block;

            unsigned int shmem_idx_warp = (num_lanes_workload_per_warp) * warpId;
            unsigned int shmem_idx_lane = shmem_idx_warp + (laneId / Block_TileK_Copy_Line_LANEs);
            unsigned int global_idx_lane_m = shmem_idx_lane + block_tile_m * BLOCK_Tile_M;
            unsigned int global_idx_lane_k = laneId%Block_TileK_Copy_Line_LANEs;   //laneId%Block_TileK_Copy_Line_LANEs + blk_tile_k*BLOCK_Tile_K;
            #pragma unroll
            for(int i=0;i<num_iters_ATile;i++){
                unsigned int contiguous_glb = global_idx_lane_k;
                unsigned int strided_glb = (global_idx_lane_m + i*Block_TileK_Copy_Lines_Per_Warp)*K_GLOBAL + blk_tile_k*BLOCK_Tile_K;
                unsigned int contiguous_shmem = (laneId%Block_TileK_Copy_Line_LANEs);
                unsigned int strided_shmem = shmem_idx_lane + i*Block_TileK_Copy_Lines_Per_Warp;
                size_t offset_permuted = permut_idx(contiguous_shmem,strided_shmem,8);
                //*((int4*)&buffer[0][0] + strided_shmem*8+ contiguous_shmem) =  *((int4*)&matA[strided_glb] +  contiguous_glb);
                *((int4*)&buffer[0][0] + offset_permuted) =  *((int4*)&matA[strided_glb] +  contiguous_glb);
            }

            // copy B, the difference is N dimension, K dimension should be same as MatA
            const unsigned int num_iters_BTile = BLOCK_Tile_N/ (NUM_warp_block *Block_TileK_Copy_Lines_Per_Warp );
            const unsigned int num_lanes_workload_per_warp_b = BLOCK_Tile_N/NUM_warp_block;
            unsigned int shmem_idx_warp_b = (num_lanes_workload_per_warp_b) * warpId;
            unsigned int shmem_idx_lane_n = shmem_idx_warp_b + (laneId / Block_TileK_Copy_Line_LANEs);
            unsigned int global_idx_lane_n = shmem_idx_lane_n + block_tile_n * BLOCK_Tile_N;
            #pragma unroll
            for(int j=0;j<num_iters_BTile;j++){
                unsigned int contiguous_glb = global_idx_lane_k;
                unsigned int strided_glb = (global_idx_lane_n + j*Block_TileK_Copy_Lines_Per_Warp)*K_GLOBAL + blk_tile_k*BLOCK_Tile_K;
                unsigned int contiguous_shmem = (laneId%Block_TileK_Copy_Line_LANEs);
                unsigned int strided_shmem = shmem_idx_lane + j*Block_TileK_Copy_Lines_Per_Warp + BLOCK_Tile_M ;
                size_t offset_permuted = permut_idx(contiguous_shmem,strided_shmem,8);
                //*((int4*)&buffer[0][0] + strided_shmem*8 + contiguous_shmem) =  *((int4*)&matB[strided_glb] +  contiguous_glb);
                *((int4*)&buffer[0][0] + offset_permuted) =  *((int4*)&matB[strided_glb] +  contiguous_glb);
            }
            /******************************* shared mem store************************************/
            __syncthreads();

            // if(block_pos == 1&& warpId==0 && laneId==0 && blk_tile_k == 0){
            //     printf("MatA in shared mem at blk_tile_k %d\n",blk_tile_k);
            //     for(int i=0;i<BLOCK_Tile_M;i++){
            //         for(int j=0;j<BLOCK_Tile_K;j++){
            //             printf("%-6.2f ",float(buffer[i][j]));
            //         }
            //         printf("\n");
            //     }
            // }

            // if(block_pos == 1 &&warpId==0 && laneId==0 && blk_tile_k==1){
            //     printf("MatB in shared mem at blk_tile_k %d\n",blk_tile_k);
            //     for(int i=0;i<BLOCK_Tile_M;i++){
            //         for(int j=0;j<BLOCK_Tile_K;j++){
            //             printf("%-6.2f ",float(buffer[i+BLOCK_Tile_M][j]));
            //         }
            //         printf("\n");
            //     }
            // }
            //break;


            for(int k_mma_step =0; k_mma_step<(BLOCK_Tile_K/MMA_Tile_K);k_mma_step++){


                __nv_bfloat16 fragementA[WARP_Tile_M/MMA_Tile_M][8] = {}; 
                __nv_bfloat16 fragementB[WARP_Tile_N/MMA_Tile_N][4] = {}; 
                #pragma unroll
                for(int mma_m =0; mma_m<WARP_Tile_M/MMA_Tile_M;mma_m++){

                    unsigned int *A = reinterpret_cast<unsigned int *>(&fragementA[mma_m][0]);
                    unsigned int contiguous_shmem_ptr = ((laneId/16) * 8 + k_mma_step*MMA_Tile_K) / 8;
                    unsigned int strided_shmem_ptr = warpId_m*WARP_Tile_M + mma_m *MMA_Tile_M+ laneId%16;
                    size_t offset_permuted = permut_idx(contiguous_shmem_ptr,strided_shmem_ptr,8);
                    // unsigned  tile_matA_shared_ptr= static_cast<unsigned>(__cvta_generic_to_shared(&buffer[warpId_m*WARP_Tile_M + mma_m *MMA_Tile_M+ laneId%16 ][(laneId/16) * 8 + k_mma_step*MMA_Tile_K]));
                    unsigned  tile_matA_shared_ptr= static_cast<unsigned>(__cvta_generic_to_shared((int4*)&buffer[0][0] + offset_permuted));

                    asm volatile("ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];" : "=r"(A[0]), "=r"(A[1]),"=r"(A[2]), "=r"(A[3]) : "r"(tile_matA_shared_ptr)); 

                    // if(warpId==0 &&  k_mma_step ==0 && mma_m==1){
                    //     if(laneId==0){
                    //         printf("fragementA at k_mma_step = %d, mma_m = %d, warp = %d\n",k_mma_step,mma_m,warpId);
                    //     }
                    //     check_fragementA(fragementA[mma_m],laneId,fragementA_global_vis);
                    // }
                    #pragma unroll
                    for(int mma_n = 0; mma_n<WARP_Tile_N/MMA_Tile_N;mma_n++){
                        unsigned int *B = reinterpret_cast<unsigned int *>(&fragementB[mma_n][0]);
                        unsigned int contiguous_shmem_ptr = ((laneId/8) * 8 + k_mma_step*MMA_Tile_K) / 8;
                        unsigned int strided_shmem_ptr = warpId_n*WARP_Tile_N + mma_n*MMA_Tile_N+ laneId%8 + BLOCK_Tile_M;
                        size_t offset_permuted = permut_idx(contiguous_shmem_ptr,strided_shmem_ptr,8);
                        // unsigned  tile_matB_shared_ptr= static_cast<unsigned>(__cvta_generic_to_shared(&buffer[warpId_n*WARP_Tile_N + mma_n*MMA_Tile_N+ laneId%8 + BLOCK_Tile_M ][(laneId/8) * 8 + k_mma_step*MMA_Tile_K]));

                        unsigned  tile_matB_shared_ptr= static_cast<unsigned>(__cvta_generic_to_shared((int4*)&buffer[0][0] + offset_permuted));

                        asm volatile ("ldmatrix.sync.aligned.x2.m8n8.shared.b16 {%0, %1}, [%2];" : "=r"(B[0]), "=r"(B[1]) : "r"(tile_matB_shared_ptr)); 


                        // if(warpId==1 &&  k_mma_step ==0 && mma_m==0 && mma_n==0){
                        //     if(laneId==0){
                        //         printf("fragementB at k_mma_step = %d, mma_n = %d, warp = %d\n",k_mma_step,mma_n,warpId);
                        //     }
                        //     for(int i=0;i<4;i++){
                        //         printf("laneid = %d, fragementB[%d] = %-6.2f\n",laneId,i,float(fragementB[mma_n][i]));
                        //     }
                        // }

                        //float *C = reinterpret_cast<float *>(fragementD[mma_m][mma_n]);
                        float *D = reinterpret_cast<float *>(fragementD[mma_m][mma_n]);

                        asm volatile(
                        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, "
                        "{%8,%9}, {%10,%11,%12,%13};\n"
                        : "=f"(D[0]), "=f"(D[1]), "=f"(D[2]), "=f"(D[3])
                        : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]),
                            "r"(B[0]),"r"(B[1]),
                            "f"(D[0]), "f"(D[1]), "f"(D[2]), "f"(D[3]));
                        
                        // if(warpId==0 &&  k_mma_step == 1 && mma_m==0 && mma_n==0){
                        //     if(laneId==0){
                        //         printf("fragementD at k_mma_step = %d, mma_m = %d, mma_n = %d, warp = %d\n",k_mma_step,mma_m,mma_n,warpId);
                        //     }
                        //     for(int i=0;i<4;i++){
                        //         printf("laneid = %d, fragementD[%d] = %-6.2f\n",laneId,i,float(fragementD[mma_m][mma_n][i]));
                        //     }
                        // }

                    }
                }
            }
            __syncthreads();
        }

        // #ifdef DEBUG_KERNEL
        // if(warpId==0 ){
        //     if(laneId==0){
        //         printf("fragementD at mma_m = %d, mma_n = %d, warp = %d\n",0,0,warpId);
        //     }
        //     for(int i=0;i<4;i++){
        //         printf("laneid = %d, fragementD[%d] = %-6.2f\n",laneId,i,float(fragementD[0][0][i]));
        //     }
        // }
        // #endif
        this_block.sync();
        long long int end_t =  clock64();



        // if(warpId==0 && laneId==0){
        //     printf("num cycles of mma baseline %lld at blockID = %d\n", end_t-start_t,block_pos);
        // }

        if(laneId==0){
            d_gpu_clock[block_pos*NUM_warp_block + warpId]=end_t-start_t;
        }

        // note below implementation is used for verifying GPU resluts, streaming the result back to global mem can be optimized
        // this lazy implemention gives poor performance since we did not consider memory coalesces
        int group_id = laneId>>2;
        int threadID_in_group = laneId % 4;
        for(int i=0;i<WARP_Tile_M/16;i++){
            for(int j=0;j<WARP_Tile_N/8;j++){
                for(int k=0;k<4;k++){
                    int row = k<2? group_id : (group_id + 8);
                    int col = (threadID_in_group * 2) + (k & 0x1);
                    row = row + i*16;
                    col = col + j*8;
                    row = row + warpId_m*WARP_Tile_M;
                    col = col + warpId_n*WARP_Tile_N;
                    row = BLOCK_Tile_M*block_tile_m + row;
                    col = BLOCK_Tile_N*block_tile_n + col;
                    // if(warpId==0 && laneId ==0){
                    //     printf("store fragementD[%d][%d][%d] = %-6.2f to matD[%d][%d]\n",i,j,k,fragementD[i][j][k],row,col);
                    // }
                    matD[row * N_GLOBAL + col] = fragementD[i][j][k];
                }

            }
        }
        __syncthreads();
    }

    //
}









__host__ void init_host_matrices(__nv_bfloat16 *a, __nv_bfloat16 *b)
{
    // row major
    for (int i = 0; i < M_GLOBAL; i++) {
        for (int j = 0; j < K_GLOBAL; j++) {
            //a[i*K_GLOBAL+j] = (__nv_bfloat16)(float)((1+i+j) % 4); //rand()
            a[i*K_GLOBAL+j] = (__nv_bfloat16)(float)(rand() % 4);
        }
    }
    // col major
    for (int i = 0; i < N_GLOBAL; i++) {
        for (int j = 0; j < K_GLOBAL; j++) {
            //b[i*K_GLOBAL+j] = (__nv_bfloat16)(float)((1+i+j) % 4); 
            b[i*K_GLOBAL+j] = (__nv_bfloat16)(float)((rand()) % 4); 
        }
    }
}





__host__ void compute_cpu(__nv_bfloat16* matrixA, __nv_bfloat16* matrixB, int gemmM, int gemmN, int gemmK, float* MatrixD){

    for(int row =0; row < gemmM;row ++){
        for(int col=0; col < gemmN; col++){
            float tmp = 0.0;
            for(int k=0;k<gemmK; k++){
                tmp += float(matrixA[row * gemmK + k]) * float(matrixB[col*gemmK + k]);
                //tmp += float(matrixA[row * gemmK/2 + k].y) * float(matrixB[col*gemmK/2 + k].y);
            }
            MatrixD[row*gemmN + col] = tmp;
        }
    }
    


}



__host__ void printMatrix(float* matrix, int rows, int cols){

    for(int i=0;i<rows;i++){
        for(int j=0;j<cols;j++){
            printf("%-6.2f ",float(matrix[i*cols +j]));
        }
        printf("\n");
    }
}

__host__ void printMatrix(__nv_bfloat16* matrix, int rows, int cols){

    for(int i=0;i<rows;i++){
        for(int j=0;j<cols;j++){
            printf("%-6.2f ",float(matrix[i*cols +j]));
        }
        printf("\n");
    }
}


__host__ bool compare_two_matrix(float* matrixA,float* matrixB, int rows, int cols){

    for(int i=0;i<rows;i++){
        for(int j=0;j<cols;j++){
            if(matrixA[i*cols +j]!=matrixB[i*cols+j]){
                printf("matrixA[%d][%d] (%-6.2f) != matrixB[%d][%d] (%-6.2f)\n",i,j,matrixA[i*cols +j],i,j,matrixB[i*cols +j]);
                return false;
            }
        }
        //printf("\n");
    }
    printf("Two input matrices are same\n");
    return true;
}




int main(){


    __nv_bfloat16* matA_cpu = new __nv_bfloat16[M_GLOBAL*K_GLOBAL];

    __nv_bfloat16* matB_cpu = new __nv_bfloat16[N_GLOBAL*K_GLOBAL];

    float* cpu_result = new float[M_GLOBAL * N_GLOBAL];

    init_host_matrices(matA_cpu,matB_cpu);

    // std::cout<<"print MatA"<<std::endl;
    // printMatrix(matA_cpu,M_GLOBAL,K_GLOBAL);
    
    // std::cout<<"print MatB"<<std::endl;
    // printMatrix(matB_cpu,N_GLOBAL,K_GLOBAL);

   



    int num_blocks = (M_GLOBAL/BLOCK_Tile_M) * (N_GLOBAL/BLOCK_Tile_N);
    int num_threads_per_block = NUM_warp_block*32;
    int size_shmem_per_block_bytes = (BLOCK_Tile_M*(BLOCK_Tile_K+SKEW_BF16) + BLOCK_Tile_N*(BLOCK_Tile_K+SKEW_BF16)) * sizeof(__nv_bfloat16);

    __nv_bfloat16* d_matA = nullptr;

    __nv_bfloat16* d_matB = nullptr;
    float* d_matD = nullptr;
    float* h_matD = new float[M_GLOBAL * N_GLOBAL];

    long long int* d_gpu_clock =  nullptr;
    long long int* h_gpu_clock =  new long long int[num_blocks*NUM_warp_block];

    __nv_bfloat16* d_fragementA_vis = nullptr;


    gpuErrchk(cudaMalloc(&d_gpu_clock, num_blocks*NUM_warp_block * sizeof(long long int)));
    gpuErrchk(cudaMalloc(&d_matA, M_GLOBAL*K_GLOBAL * sizeof(__nv_bfloat16)));
    gpuErrchk(cudaMalloc(&d_matB, N_GLOBAL*K_GLOBAL * sizeof(__nv_bfloat16)));
    gpuErrchk(cudaMalloc(&d_matD, M_GLOBAL * N_GLOBAL * sizeof(float)));


    gpuErrchk(cudaMalloc(&d_fragementA_vis, MMA_Tile_M * MMA_Tile_K * sizeof(__nv_bfloat16)));

    gpuErrchk(cudaMemcpy(d_matA, matA_cpu,M_GLOBAL*K_GLOBAL * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_matB, matB_cpu,N_GLOBAL*K_GLOBAL * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));



    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);



    long long int baseline_cycles_total = 0;
    int NUM_PROFILES = Profiling_ROUNDS;
    for(int iter=0; iter<NUM_PROFILES; ++iter){
            // float milliseconds = 0;
            gpuErrchk(cudaMemset ( d_gpu_clock, 0, num_blocks * sizeof(long long int) ));
            cudaEventRecord(start);
            mma_permuted<<<num_blocks,num_threads_per_block,size_shmem_per_block_bytes>>>(d_matA,d_matB,d_matD,d_gpu_clock /*,d_fragementA_vis*/); 

            gpuErrchk(cudaMemcpy(h_gpu_clock, d_gpu_clock, num_blocks*NUM_warp_block  * sizeof(long long int), cudaMemcpyDeviceToHost));
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            cudaDeviceSynchronize();
            // cudaEventElapsedTime(&milliseconds,start,stop);
            baseline_cycles_total += *std::max_element(h_gpu_clock, h_gpu_clock + num_blocks*NUM_warp_block);
    }
  
    long long int baseline_cycles = (baseline_cycles_total)/(NUM_PROFILES);
    printf("num of cycles mma permuted: %lld\n", baseline_cycles);  

    //mma_baseline<<<num_blocks,num_threads_per_block,size_shmem_per_block_bytes>>>(d_matA,d_matB,d_matD /*,d_fragementA_vis*/); 

    gpuErrchk(cudaPeekAtLastError());
    //checkKernelErrors ( (mma_pipeline1<<<1,32>>>(nullptr,nullptr,nullptr)  ));
    
    
    
    gpuErrchk(cudaMemcpy(h_matD, d_matD, M_GLOBAL * N_GLOBAL  * sizeof(float), cudaMemcpyDeviceToHost));


    cudaDeviceSynchronize();

    if(CPU_DEBUG == true){
        printf("check GPU result against CPU result\n");
        compute_cpu(matA_cpu,matB_cpu,M_GLOBAL,N_GLOBAL,K_GLOBAL,cpu_result);
        bool check = compare_two_matrix(cpu_result,h_matD,M_GLOBAL,N_GLOBAL);
        // if(check == false){
        //     std::cout<<"print GPU result"<<std::endl;
        //     printMatrix(h_matD,M_GLOBAL,N_GLOBAL);
        //     std::cout<<"print CPU reference"<<std::endl;
        //     printMatrix(cpu_result,M_GLOBAL,N_GLOBAL);
        // }
    }



}