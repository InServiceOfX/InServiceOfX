#!/bin/bash

# Consider looking at these for references:
# https://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/

get_gpu_model()
{
	nvidia-smi --query-gpu=gpu_name --format=csv,noheader | head -n 1
}

# See https://developer.nvidia.com/cuda-gpus for "Compute capability" data.
get_compute_capability()
{
	local gpu_model="$(get_gpu_model)"
	case "$gpu_model" in
		"NVIDIA GeForce GTX 980 Ti"*) echo "5.2" ;;
		"NVIDIA GeForce GTX 1050"*) echo "6.1" ;;
		"NVIDIA GeForce RTX 3070"*) echo "8.6" ;;
		"NVIDIA H100 80GB HBM3"*) echo "9.0" ;;
		*) echo "Unknown" ;;
	esac
}

get_compute_capability_as_cuda_architecture()
{
	local gpu_model="$(get_gpu_model)"
	case "$gpu_model" in
		"NVIDIA GeForce GTX 980 Ti"*) echo "52" ;;
		"NVIDIA GeForce GTX 1050"*) echo "61" ;;
		"NVIDIA GeForce RTX 3070"*) echo "86" ;;
		"NVIDIA H100 80GB HBM3"*) echo "90" ;;
		*) echo "Unknown" ;;
	esac
}

print_results()
{
	echo $(get_gpu_model)
	echo $(get_compute_capability)
	echo $(get_compute_capability_as_cuda_architecture)
}

print_results