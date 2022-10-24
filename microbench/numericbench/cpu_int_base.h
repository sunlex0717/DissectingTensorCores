#include<iostream>
#include<vector>
#include<cmath>
#include<cstdlib>
#pragma once
// check the difference of two matrix
void compute_diff_l1_norm(int* cpu_base, int* gpu_res, int rows, int cols,double& abs_err, double&  l1_norm){

    // l1 norm : |gpu_res[i] - cpu_res[i]|/|gpu_res[i|
    // double l1_norm = 0.0; 
    // double abs = 0.0;
    for(int row =0; row< rows; row ++){
        for(int col =0; col < cols; col ++){
            int gid = col + row*cols;
            abs_err += std::abs(gpu_res[gid] - cpu_base[gid]);
            l1_norm += abs_err/std::abs(gpu_res[gid]);
        }
    }
    // l1_norm = l1_norm/(rows*cols);
    // abs_err = abs_err/(rows*cols);
    //return l1_norm/(rows*cols);
};



void compute_diff_l2_norm(int* cpu_base, int* gpu_res, int rows, int cols, double&  l2_norm){

    // l1 norm : |gpu_res[i] - cpu_res[i]|/|gpu_res[i|
    // double l1_norm = 0.0; 
    // double abs = 0.0;
    double tensor_diff_norm = 0.0;
    double tensor_gpu_norm = 0.0;
    for(int row =0; row< rows; row ++){
        for(int col =0; col < cols; col ++){
            int gid = col + row*cols;
            tensor_diff_norm += std::pow((double(gpu_res[gid]) - double(cpu_base[gid])) ,2 ) ;
            tensor_gpu_norm += std::pow(double(gpu_res[gid]),2 );
        }
    }
    l2_norm = std::sqrt(tensor_diff_norm)/std::sqrt(tensor_gpu_norm);
    //l1_norm = l1_norm/(rows*cols);
    // abs_err = abs_err/(rows*cols);
    //return l1_norm/(rows*cols);
};


template<typename datatypeIN, typename datatypeOut>
void gemm_mnk_cpu(datatypeIN* MatA,datatypeIN* MatB,datatypeOut* MatC, datatypeOut* MatD, int M, int N, int K){

    // Matd = MatA * MatB + MatC
    for(int row=0; row < M; row++){
        for(int col=0; col < N; col++){
            int gid =  col + row*N;
            datatypeOut tmp = 0;
            for(int inner=0; inner < K; inner++)
            {
                tmp += MatA[inner + row*K] * MatB[col + inner*N];
            }
            MatD[gid] = tmp+ MatC[gid];
        }
    }

};

void print_mat(int* Mat, int rows, int cols){
    
    for(int row = 0; row < rows; row++){
        for(int col =0;col < cols; col++){
            printf("%8d ", Mat[col + row*cols]);
        }
        std::cout<<std::endl;
    }
    std::cout<<std::endl;

}

// check if the matrix has inf or not
bool Mat_has_inf(int* Mat, int num_eles){

    for(int i=0;i<num_eles;i++){
        if(std::isinf(Mat[i])){
            //std::cout<<Mat[i]<<std::endl;
            return true;

        } 
    }
    return false;
}


bool Mat_has_nan(int* Mat, int num_eles){

    for(int i=0;i<num_eles;i++){
        if(std::isnan(Mat[i])) return true;
    }
    return false;
}
// template <typename typeIn, typename typeOut,int M, int N, int K >
// struct gemmCPU
// {
//     /* data */
// };


