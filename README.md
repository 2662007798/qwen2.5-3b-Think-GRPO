# qwen2.5-3b-Think-GRPO

如果你有conda

conda create --name unsloth_env \
    python=3.11 \
    pytorch-cuda=12.1 \
    pytorch cudatoolkit xformers -c pytorch -c nvidia -c xformers \
    -y
conda activate unsloth_env

pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
pip install --no-deps trl peft accelerate bitsandbytes

进行安装

如果没有  直接 pip -r requirements.txt 即可  

当然 即使你有conda  创建一个新的环境  你也可以直接选择pip -r requirements.txt 进行安装。

欢迎大家来共同学习，有问题多交流。
