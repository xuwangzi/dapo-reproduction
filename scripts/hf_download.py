"""
此文件用于下载Hugging Face的datasets和models到本地

功能说明:
- 设置Hugging Face镜像环境变量，使用国内镜像加速下载
- 下载预训练模型和分词器到本地指定目录
- 下载数据集到本地缓存目录
- 支持自定义模型名称和保存路径

作者: xwz
创建时间: 2025-07-08
"""

import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

# 设置镜像环境变量
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# 下载模型和分词器到本地
save_dir = "/root/group-shared/models/base_models"
model_name = "Qwen/Qwen3-0.6B"

model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

model.save_pretrained(f"{save_dir}/{model_name}")
tokenizer.save_pretrained(f"{save_dir}/{model_name}")

# 下载数据集到本地缓存
save_dir = "/root/group-shared/datasets/base_datasets"
dataset_name = 'open-r1/OpenR1-Math-220k'

dataset = load_dataset(dataset_name)
dataset.save_to_disk(f"{save_dir}/{dataset_name}")