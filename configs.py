import torch

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 由于这里使用的模型很小，使用CUDA只会让训练变慢，所以这里强制使用CPU
device = torch.device("cpu")
