import torch
import subprocess

print("=" * 60)
print("DIAGNOSI GPU NVIDIA 970")
print("=" * 60)

# 1. Check PyTorch CUDA support
print("\n1. PyTorch:")
print(f"   CUDA disponibile: {torch.cuda.is_available()}")
print(f"   CUDA versione compilata: {torch.version.cuda}")
print(f"   cuDNN versione: {torch.backends.cudnn.version()}")

# 2. Check driver NVIDIA
print("\n2. Driver NVIDIA:")
try:
    driver_output = subprocess.check_output(["nvidia-smi"], text=True)
    lines = driver_output.split("\n")
    for line in lines[:10]:
        print(f"   {line}")
except FileNotFoundError:
    print("   ❌ NVIDIA driver NON trovato!")
    print("      Scarica da: https://www.nvidia.com/Download/driverDetails.aspx")

# 3. Check CUDA toolkit
print("\n3. CUDA Toolkit:")
try:
    cuda_output = subprocess.check_output(["nvcc", "--version"], text=True)
    print(f"   {cuda_output}")
except FileNotFoundError:
    print("   ❌ CUDA Toolkit NON trovato!")
    print("      Scarica da: https://developer.nvidia.com/cuda-toolkit")

print("\n" + "=" * 60)
