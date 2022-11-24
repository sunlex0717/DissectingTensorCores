# Dissecting Tensor Cores via Microbenchmarks: Latency, Throughput and Numeric Behaviors

Code repo for Dissecting Tensor Cores via Microbenchmarks: Latency, Throughput and Numeric Behaviors.

* [preprint](https://arxiv.org/abs/2206.02874)
* [IEEE TPDS](https://ieeexplore.ieee.org/document/9931992)

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

## citations
```
@ARTICLE{9931992,
  author={Sun, Wei and Li, Ang and Geng, Tong and Stuijk, Sander and Corporaal, Henk},
  journal={IEEE Transactions on Parallel and Distributed Systems}, 
  title={Dissecting Tensor Cores via Microbenchmarks: Latency, Throughput and Numeric Behaviors}, 
  year={2023},
  volume={34},
  number={1},
  pages={246-261},
  doi={10.1109/TPDS.2022.3217824}}
```
