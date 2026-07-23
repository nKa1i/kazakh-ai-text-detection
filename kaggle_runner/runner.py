import torch
import sys

print("=" * 60)
print("KAGGLE REMOTE GPU TEST SUCCESSFUL")
print(f"Python Version: {sys.version}")
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU Device Count: {torch.cuda.device_count()}")
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
print("=" * 60)
