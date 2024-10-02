#!/usr/bin/python3
import subprocess

def get_idle_gpus():
  # Run the nvidia-smi command to get GPU information in CSV format
  gpu_info_result = subprocess.run(
    ['nvidia-smi', '--query-gpu=index,pci.bus_id', '--format=csv,noheader'],
    stdout=subprocess.PIPE, universal_newlines=True
  )
	
  # Run the nvidia-smi command to get processes information in CSV format
  process_info_result = subprocess.run(
    ['nvidia-smi', '--query-compute-apps=gpu_bus_id', '--format=csv,noheader'],
    stdout=subprocess.PIPE, universal_newlines=True
  )
  # Parse GPU information
  bus2gpu = {line.strip().split(",")[1].strip(): line.strip().split(",")[0].strip() for line in gpu_info_result.stdout.splitlines()}
  # Parse process information
  busy_gpus = set(line.strip() for line in process_info_result.stdout.splitlines())
  # Determine idle GPUs
  idle_gpus = [gpu for bus, gpu in bus2gpu.items() if bus not in busy_gpus]
  return idle_gpus

if __name__ == "__main__":
  idle_gpus = get_idle_gpus()
  print(",".join(idle_gpus))