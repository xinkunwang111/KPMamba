import torch
import clip
import os

# 指定保存的路径
model_path = "./clip_model.pth"
device = "cuda" if torch.cuda.is_available() else "cpu"

# 确保目录存在
os.makedirs(os.path.dirname(model_path), exist_ok=True)

# 下载并加载 CLIP 模型
model, _ = clip.load("ViT-B/16", device=device)

# 保存权重文件
torch.save(model.state_dict(), model_path)
