import torch
print(f"PyTorch 版本: {torch.__version__}")
print(f"CUDA 可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA 版本: {torch.version.cuda}")
    print(f"GPU 数量: {torch.cuda.device_count()}")
    print(f"当前设备: {torch.cuda.current_device()}")
    print(f"GPU 型号: {torch.cuda.get_device_name(0)}")
    print(f"BF16 支持: {torch.cuda.is_bf16_supported()}")
else:
    print("❌ 无 CUDA GPU 检测到，可能装了 CPU 版 PyTorch")
print(f"当前设备类型: {next(torch.zeros(1).device for _ in [0])}")
