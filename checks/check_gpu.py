"""项目2：验证 PyTorch 能在 GPU 上完成一次矩阵乘法。

这是环境检查，不是模型性能测试。
"""

import sys

import torch
import torchvision


print("Python:", sys.version.split()[0])
print("Python path:", sys.executable)
print("PyTorch:", torch.__version__)
print("Torchvision:", torchvision.__version__)
print("PyTorch CUDA runtime:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())

if not torch.cuda.is_available():
    raise SystemExit("FAIL: PyTorch cannot access CUDA. Please share this output.")

print("GPU:", torch.cuda.get_device_name(0))

# 创建一个 64 行、64 列、所有元素均为 1 的矩阵，直接放到 GPU 上。
x = torch.ones((64, 64), dtype=torch.float32, device="cuda:0")

# @ 表示矩阵乘法；输出的每个元素应当是 64 个 1 相加，即 64。
y = x @ x
torch.cuda.synchronize()  # 等待 GPU 完成计算；发生计算错误时在这里暴露。

print("Result device:", y.device)
print("Result[0, 0]:", y[0, 0].item())
if not torch.all(y == 64).item():
    raise SystemExit("FAIL: GPU matrix multiplication returned an unexpected result.")

print("PASS: GPU matrix multiplication is correct.")
