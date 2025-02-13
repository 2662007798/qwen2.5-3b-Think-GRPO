# qwen2.5-3b-Think-GRPO
基于unsloth的GRPO训练
https://github.com/unslothai/unsloth

200步复现"aha" 时刻

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
pip -r requirements.txt
```
即可  

当然 即使你有conda  创建一个新的环境  你也可以直接选择pip -r requirements.txt 进行安装。

对你有帮助的话可以点一下Star 谢谢

欢迎大家来共同学习，有问题多交流。

