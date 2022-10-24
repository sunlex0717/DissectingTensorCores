CC = nvcc
FLAG =-gencode=arch=compute_86,code=\"sm_86,compute_86\"  -lcudart

all: pipeline_mma \
	baseline_mma \
	permuted_mma 
	

baseline_mma: ./mma_baseline.cu
	$(CC) $(FLAG)  -o $@.out $^


pipeline_mma: ./mma_pipeline.cu
	$(CC) $(FLAG)  -o $@.out $^


permuted_mma: ./mma_permuted.cu
	$(CC) $(FLAG)  -o $@.out $^


clean:
	rm -rf *.out