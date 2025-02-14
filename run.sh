#!/bin/bash

# 检查CUDA是否可用
if ! command -v nvidia-smi &> /dev/null; then
    echo "错误: 未检测到NVIDIA GPU或CUDA环境"
    exit 1
fi

# 检查配置文件
if [ ! -f "config.yaml" ]; then
    echo "错误: 未找到config.yaml配置文件"
    exit 1
fi

# 检查Python环境
if ! command -v python3 &> /dev/null; then
    echo "错误: 未找到Python3"
    exit 1
fi

# 检查必要的Python包
echo "检查并安装必要的Python包..."
pip install -r requirements.txt

# 设置CUDA设备
export CUDA_VISIBLE_DEVICES=0

# 运行训练脚本
echo "开始训练..."
python3 trainGRPO.py

# 检查训练是否成功完成
if [ $? -eq 0 ]; then
    echo "训练成功完成!"
else
    echo "训练过程中出现错误!"
    exit 1
fi 