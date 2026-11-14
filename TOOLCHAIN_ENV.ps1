# Load the reproducible project-local Triton/CUDA compiler paths in a
# PowerShell session before running the fused-kernel benchmarks.
$env:TRITON_CACHE_DIR = 'D:\raw\.triton_cache'
$env:CUDA_PATH = 'C:\Users\NVT\AppData\Roaming\Python\Python313\site-packages\triton\backends\nvidia'
$env:CC = 'C:\Users\NVT\AppData\Roaming\Python\Python313\site-packages\triton\runtime\tcc\tcc.exe'
$env:CAMO_NVCC_ROOT = 'D:\raw\cuda_nvcc_12.4.131\cuda_nvcc-windows-x86_64-12.4.131-archive'
$env:PATH = "$env:CAMO_NVCC_ROOT\bin;$env:PATH"
Write-Host "Triton cache: $env:TRITON_CACHE_DIR"
Write-Host "nvcc: $env:CAMO_NVCC_ROOT\bin\nvcc.exe"
