import torchtest

# 当前安装的 PyTorch 库的版本
print(torchtest.__version__)
# 检查 CUDA 是否可用，即你的系统有 NVIDIA 的 GPU
print(torchtest.cuda.is_available())