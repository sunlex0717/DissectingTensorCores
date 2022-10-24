BASE_DIR := $(shell pwd)
BIN_DIR := $(BASE_DIR)/../../../bin/

GENCODE_SM70 ?= -gencode=arch=compute_70,code=\"sm_70,compute_70\" # V100
GENCODE_SM75 ?= -gencode=arch=compute_75,code=\"sm_75,compute_75\" # Turing
GENCODE_SM80 ?= -gencode=arch=compute_80,code=\"sm_80,compute_80\" # A100
GENCODE_SM86 ?= -gencode=arch=compute_86,code=\"sm_86,compute_86\" # RTX30


TargetSM ?= 80
GENCODE_SM = -gencode=arch=compute_${TargetSM},code=\"sm_${TargetSM},compute_${TargetSM}\"
CUOPTS = $(GENCODE_ARCH) $(GENCODE_SM) 

CC := nvcc

CUDA_PATH ?= /use/local/cuda-10.2/
INCLUDE := $(CUDA_PATH)/samples/common/inc/
LIB := 
ILP ?= 1
ITERS ?= 999
MEAN ?= 0.0
STDDEV ?= 1.0
release:
	$(CC) $(NVCC_FLGAS) --define-macro ILPconfig=$(ILP),ITERS=$(ITERS),MEAN=$(MEAN),STDDEV=$(STDDEV) $(CUOPTS) $(SRC) -o $(EXE) -I $(INCLUDE) -L $(LIB) -lcudart
	cp $(EXE) $(BIN_DIR)

# clean:
# 	rm -f *.o; rm -f $(EXE)

clean:
	rm -f *.app *.txt

run:
	./$(EXE)

profile:
	nv-nsight-cu-cli --metrics l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum.per_second,smsp__sass_average_data_bytes_per_wavefront_mem_shared.pct,l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum,smsp__inst_executed_pipe_lsu.avg.pct_of_peak_sustained_active ./$(EXE)

profile_bank:
	nv-nsight-cu-cli --metrics l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,sm__sass_l1tex_data_pipe_lsu_wavefronts_mem_shared_op_ld.sum,sm__sass_l1tex_data_pipe_lsu_wavefronts_mem_shared_op_ldsm.sum,smsp__sass_average_data_bytes_per_wavefront_mem_shared ./$(EXE)

profile_lsu_mio:
	nv-nsight-cu-cli --metrics smsp__average_warp_latency_issue_stalled_lg_throttle.ratio,smsp__average_warp_latency_issue_stalled_mio_throttle.ratio,smsp__average_warp_latency_issue_stalled_short_scoreboard.ratio ./$(EXE)


# smsp__average_warps_issue_stalled_mio_throttle_per_issue_active.ratio 
# sm__inst_executed_pipe_lsu.sum
# smsp__average_inst_executed_pipe_lsu_per_warp.ratio

profile_smem:
	nv-nsight-cu-cli --metrics l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum,sm__sass_data_bytes_mem_shared_op_ld.sum,sm__inst_executed_pipe_lsu.sum,sm__sass_l1tex_pipe_lsu_wavefronts_mem_shared.sum,sm__sass_l1tex_data_pipe_lsu_wavefronts_mem_shared_op_ld.sum ./$(EXE)

events:
	nvprof  --events elapsed_cycles_sm ./$(EXE)

profileall:
	nvprof --concurrent-kernels off --print-gpu-trace -u us --metrics all --demangling off --csv --log-file data.csv ./$(EXE) 

nvsight:
	nv-nsight-cu-cli --metrics gpc__cycles_elapsed.avg,sm__cycles_elapsed.sum,smsp__inst_executed.sum,sm__warps_active.avg.pct_of_peak_sustained_active,l1tex__t_sectors_pipe_lsu_mem_global_op_ld_lookup_hit.sum,l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,l1tex__t_sectors_pipe_lsu_mem_global_op_st_lookup_hit.sum,l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum,lts__t_sectors_srcunit_tex_op_read.sum,lts__t_sectors_srcunit_tex_op_write.sum,lts__t_sectors_srcunit_tex_op_read_lookup_hit.sum,lts__t_sectors_srcunit_tex_op_write_lookup_hit.sum,lts__t_sector_op_read_hit_rate.pct,lts__t_sector_op_write_hit_rate.pct,lts__t_sectors_srcunit_tex_op_read.sum.per_second,dram__sectors_read.sum,dram__sectors_write.sum,dram__bytes_read.sum  --csv --page raw ./$(EXE) | tee nsight.csv

ptx:
	cuobjdump -ptx ./$(EXE)  tee ptx.txt

sass:
	cuobjdump -sass ./$(EXE)  tee sass.txt
