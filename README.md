# GRPO中文数学推理训练

本项目使用GRPO(Guided Reward Policy Optimization)方法来训练中文数学推理模型。
基于unsloth原因本项目目前只支持单卡，如需多卡和分布式请参考我们另外的项目
https://github.com/826568389/GRPO-R1

## 如需要打印RL过程模型的输出，请在配置文件中设置log级别为debug
## 最新更新

### 2025-02-14 更新
1. 代码优化
   - 禁用vLLM以解决序列化问题
   - 移除不支持的use_cuda_graph参数
   - 优化日志输出，删除重复信息
   - 添加更详细的训练状态记录
   - 重构数据集加载逻辑，支持多种数据源

2. 数据集加载优化
   - 支持本地数据集和Hugging Face数据集
   - 添加数据集格式配置（json, csv, parquet等）
   - 增加数据集验证和错误处理
   - 优化数据处理流程

3. wandb配置优化
   - 改为离线模式，仅在训练结束时上传数据
   - 添加训练进度回调，每10步记录一次
   - 优化日志存储和上传机制
   - 减少不必要的网络同步

4. 内存管理
   - 添加GPU内存使用监控
   - 优化内存使用配置
   - 降低gpu_memory_utilization为0.2
   - 添加内存使用日志

5. 错误处理
   - 添加更完善的错误处理机制
   - 输出完整的错误栈信息
   - 添加数据集路径检查
   - 增加数据验证

6. 配置文件更新
   - 添加测试配置部分
   - 修正数值类型问题
   - 优化配置项结构
   - 新增数据集类型配置

## 项目结构

```
.
├── trainGRPO.py    # 主训练脚本
├── config.yaml     # 配置文件
├── run.sh         # 运行脚本
└── README.md      # 项目文档
```

## 使用方法

1. 修改配置文件
```yaml
# 在config.yaml中设置相关参数
model:
  name: "/path/to/model"  # 设置模型路径
  max_seq_length: 2048
  lora_rank: 64
  load_in_4bit: true
  fast_inference: false
  gpu_memory_utilization: 0.2

# 数据集配置 - 支持两种方式
data:
  # 方式1: 本地数据集
  dataset_type: "local"
  dataset_name: "/path/to/dataset"
  dataset_format: "json"  # 支持json, csv, parquet等
  
  # 方式2: Hugging Face数据集
  # dataset_type: "huggingface"
  # dataset_name: "gsm8k"
  # dataset_config: "main"
```

2. 运行训练
```bash
bash run.sh
```

## 注意事项
- 确保有足够的GPU内存
- 建议先使用小数据集测试
- 检查数据集路径和格式是否正确
- 留意日志输出中的警告信息
- 确保数据集包含必要的列（question和answer）

## 依赖
- Python 3.8+
- PyTorch
- transformers
- unsloth
- wandb (可选)

如果没有依赖，直接：
```bash
pip install -r requirements.txt
```

当然，即使你有conda，创建一个新的环境后也可以直接选择pip install -r requirements.txt 进行安装。

对你有帮助的话可以点一下Star 谢谢
作者：阿生(2662007798@qq.com)、团子(826568389@qq.com)

欢迎大家来共同学习，有问题多交流。

# qwen2.5-3b-Think-GRPO
基于unsloth的GRPO训练
https://github.com/unslothai/unsloth

在24G显存200步复现"aha" 时刻

![fbf3d09c887a6a496779c63f2b82a94](https://github.com/user-attachments/assets/f8517316-249b-4d46-82eb-2e5eafe1e091)

如果你有conda
```
conda create --name unsloth_env \
    python=3.11 \
    pytorch-cuda=12.1 \
    pytorch cudatoolkit xformers -c pytorch -c nvidia -c xformers \
    -y
conda activate unsloth_env

pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
pip install --no-deps trl peft accelerate bitsandbytes
```
进行安装

如果没有  直接 
```
pip install -r requirements.txt
```
即可  

当然 即使你有conda  创建一个新的环境  你也可以直接选择pip install -r requirements.txt 进行安装。

对你有帮助的话可以点一下Star 谢谢
作者：阿生(2662007798@qq.com)、团子(826568389@qq.com)

欢迎大家来共同学习，有问题多交流。

