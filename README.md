# Dissecting Tensor Cores via Microbenchmarks: Latency, Throughput and numeric Behaviors

Code repo for Dissecting Tensor Cores via Microbenchmarks: Latency, Throughput and numeric Behaviors.


TODO: Provide better instructions and explanations

# Setup 

## Add CUDA path to env. e.g

```
export CUDA_PATH=/usr/local/cuda-11.0
export PATH=$CUDA_PATH/bin:$PATH
export CUDACXX=$CUDA_PATH/bin/nvcc
```

## Config compiler target Arch/SM 

```export TargetSM=80 ``` // for A100

```export TargetSM=70 ``` // for V100

```export TargetSM=75 ``` // for Turing


## Run script

```
cd/microbench
sh run_all.sh
```

You are expected to get xxx-ILPx.log files.

Note, there will be static_assert errors messages when running the scripts, because some codes have static_assert() for larger ILPs. This kind of error messages can be ignored.


-------------------------------------

## References
Some codes are borrowed from [Accel-Sim](https://github.com/accel-sim/accel-sim-framework)



