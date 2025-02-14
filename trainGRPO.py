# -*- coding: utf-8 -*-
from unsloth import FastLanguageModel, PatchFastRL, is_bfloat16_supported
from trl import GRPOConfig, GRPOTrainer
PatchFastRL("GRPO", FastLanguageModel)

import re
from datasets import load_dataset, Dataset
import os
import yaml
import wandb
import resource
import tempfile
import shutil
from datetime import datetime
from contextlib import contextmanager
from transformers.trainer_callback import TrainerCallback
import torch
import logging

# 删除未使用的变量和注释
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# 加载配置
logger.info("加载配置文件...")
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

# 创建日志目录
os.makedirs(config['logging']['save_path'], exist_ok=True)
file_handler = logging.FileHandler(
    os.path.join(config['logging']['save_path'], 
                f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
)
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(file_handler)
logger.setLevel(getattr(logging, config['logging']['level']))

# 在加载数据集前添加数据集路径信息
logger.info("数据集配置:")
logger.info(f"数据集路径: {config['data']['dataset_name']}")
logger.info(f"数据集配置: {config['data']['dataset_config']}")
logger.info(f"使用数据集分割: {config['data']['split']}")

# 删除重复的模型配置打印
logger.info("配置信息:")
logger.info("模型配置:")
logger.info(f"- 路径: {config['model']['name']}")
logger.info(f"- 序列长度: {config['model']['max_seq_length']}")
logger.info(f"- LoRA秩: {config['model']['lora_rank']}")

logger.info("训练配置:")
logger.info(f"- 学习率: {config['training']['learning_rate']}")
logger.info(f"- 批次大小: {config['training']['batch_size']}")
logger.info(f"- 梯度累积步数: {config['training']['gradient_accumulation_steps']}")
logger.info(f"- 最大训练步数: {config['training']['max_steps']}")
logger.info(f"- 保存步数: {config['training']['save_steps']}")
logger.info(f"- 权重衰减: {config['training']['weight_decay']}")
logger.info(f"- 预热比例: {config['training']['warmup_ratio']}")
logger.info(f"- 最大梯度范数: {config['training']['max_grad_norm']}")

logger.info("数据集配置:")
logger.info(f"- 路径: {config['data']['dataset_name']}")
logger.info(f"- 配置: {config['data']['dataset_config']}")
logger.info(f"- 分割: {config['data']['split']}")

# 在模型加载之前添加日志
logger.info("开始加载模型...")
logger.info(f"模型路径: {config['model']['name']}")
logger.info(f"模型配置: max_seq_length={config['model']['max_seq_length']}, lora_rank={config['model']['lora_rank']}")

# 然后再使用配置初始化模型
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = config['model']['name'],
    max_seq_length = config['model']['max_seq_length'],
    load_in_4bit = config['model']['load_in_4bit'],
    fast_inference = False,  # 禁用 vLLM
    max_lora_rank = config['model']['lora_rank'],
    gpu_memory_utilization = config['model']['gpu_memory_utilization'],
)
logger.info("基础模型加载完成")

logger.info("配置LoRA参数...")
model = FastLanguageModel.get_peft_model(
    model,
    r = config['model']['lora_rank'],
    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_alpha = config['model']['lora_rank'],
    use_gradient_checkpointing = "unsloth",
    random_state = config['training']['seed'],
)
logger.info("LoRA配置完成")


def set_ulimit():
    try:
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        resource.setrlimit(resource.RLIMIT_NOFILE, (hard, hard))
    except Exception as e:
        print(f"设置文件句柄限制失败: {e}")

@contextmanager
def manage_temp_dir():
    original_temp = tempfile.gettempdir()
    try:
        temp_dir = tempfile.mkdtemp(prefix='train_temp_')
        tempfile.tempdir = temp_dir
        yield
    finally:
        tempfile.tempdir = original_temp
        try:
            shutil.rmtree(temp_dir, ignore_errors=True)
        except Exception as e:
            print(f"清理临时目录失败: {e}")

# 系统提示词和数据集准备
def extract_xml_answer(text: str) -> str:
    answer = text.split("<answer>")[-1]
    answer = answer.split("</answer>")[0]
    return answer.strip()

def extract_hash_answer(text: str) -> str | None:
    if "####" not in text:
        return None
    return text.split("####")[1].strip()

def get_gsm8k_questions(config, split = None) -> Dataset:
    logger.info(f"开始加载数据集: {config['data']['dataset_name']}")
    logger.info(f"数据集配置: {config['data']['dataset_config']}")
    split = split or config['data']['split']
    logger.info(f"使用数据集分割: {split}")
    
    try:
        # 首先检查数据集路径是否存在
        if not os.path.exists(config['data']['dataset_name']):
            raise ValueError(f"数据集路径不存在: {config['data']['dataset_name']}")
            
        # 直接加载数据集，不尝试获取配置信息
        logger.info("加载数据集文件...")
        data = load_dataset(
            config['data']['dataset_name'], 
            config['data']['dataset_config']
        )[split]
        logger.info(f"原始数据集大小: {len(data)}")
        
        # 检查数据集的列
        logger.info(f"数据集列: {data.column_names}")
        
        logger.info("处理数据集...")
        data = data.map(lambda x: {
            'prompt': [
                {'role': 'system', 'content': config['data']['system_prompt']},
                {'role': 'user', 'content': x['question']}
            ],
            'answer': extract_hash_answer(x['answer'])
        })
        logger.info(f"处理后数据集大小: {len(data)}")
        logger.info("数据集处理完成")
        return data
        
    except Exception as e:
        logger.error(f"加载数据集时发生错误: {str(e)}", exc_info=True)
        raise

dataset = get_gsm8k_questions(config)
def think_quality_reward_len_func(prompts, completions, answer, **kwargs) -> list[float]:
    responses = [completion[0]['content'] for completion in completions]
    
    def evaluate_think_length(response):
        think_match = re.search(r'<Think>(.*?)</Think>', response, re.DOTALL)
        if not think_match:
            logger.warning("Think标签未找到")
            return 0.0
        
        think_content = think_match.group(1).strip()
        content_length = len(think_content)
        
        # 提高各档位的分数
        if content_length < 300:
            return 0
        elif content_length < 400:
            return 1.0
        elif content_length < 600:
            return 1.5        # 从0.5提高到1.0
        elif content_length < 800:
            return 2.0        # 从1.0提高到2.0
        elif content_length < 1000:
            return 3.0 
        elif content_length < 1600:
            return 4.0 
        else:
            return 5.0        # 从2.0提高到4.0
    
    rewards = [evaluate_think_length(r) for r in responses]
    
    # 打印当前Think长度和得分（用于调试）
    think_content = re.search(r'<Think>(.*?)</Think>', responses[0], re.DOTALL)
    if think_content:
        content_length = len(think_content.group(1).strip())
        logger.info(f"Think长度: {content_length}")
        logger.info(f"Think长度得分: {rewards[0]:.2f}")
    
    return rewards


def answer_quality_reward_len_func(prompts, completions, answer, **kwargs) -> list[float]:
    responses = [completion[0]['content'] for completion in completions]
    
    def evaluate_answer_length(response):
        answer_match = re.search(r'<answer>(.*?)</answer>', response, re.DOTALL)
        if not answer_match:
            logger.warning("answer标签未找到")
            return 0.0
        
        answer_content = answer_match.group(1).strip()
        content_length = len(answer_content)
        
        # 提高各档位的分数
        if content_length < 300:
            return 0
        elif content_length < 400:
            return 1.0
        elif content_length < 600:
            return 1.5        # 从0.5提高到1.0
        elif content_length < 800:
            return 2.0        # 从1.0提高到2.0
        elif content_length < 1000:
            return 3.0 
        elif content_length < 1600:
            return 4.0 
        else:
            return 5.0        # 从2.0提高到4.0
    
    rewards = [evaluate_answer_length(r) for r in responses]
    
    # 打印当前答案长度和得分（用于调试）
    answer_content = re.search(r'<answer>(.*?)</answer>', responses[0], re.DOTALL)
    if answer_content:
        content_length = len(answer_content.group(1).strip())
        print(f"\n答案长度: {content_length}")
        print(f"答案长度得分: {rewards[0]:.2f}")
    
    return rewards
# Reward functions
def correctness_reward_func(prompts, completions, answer, **kwargs) -> list[float]:
    responses = [completion[0]['content'] for completion in completions]
    q = prompts[0][-1]['content']
    extracted_responses = [extract_xml_answer(r) for r in responses]
    
    logger.debug("评估答案:")
    logger.debug(f"问题: {q}")
    logger.debug(f"期望答案: {answer[0]}")
    logger.debug(f"模型输出: {responses[0]}")
    
    # 更精确的答案匹配
    def check_answer(response, expected):
        if str(expected).isdigit():
            pattern = r'(?:^|[^\d])' + str(expected) + r'(?:[^\d]|$)'
            return bool(re.search(pattern, response))
        return response.strip() == str(expected)
    
    # 计算答案奖励
    rewards = [2.0 if check_answer(r, answer[0]) else 0.0 for r in extracted_responses]
    
    logger.debug(f"答案评分: {rewards[0]:.2f}")
    
    return rewards

def int_reward_func(completions, **kwargs) -> list[float]:
    responses = [completion[0]['content'] for completion in completions]
    extracted_responses = [extract_xml_answer(r) for r in responses]
    return [0.5 if r.isdigit() else 0.0 for r in extracted_responses]

def strict_format_reward_func(completions, **kwargs) -> list[float]:
    """Reward function that checks if the completion has a specific format."""
    pattern = r"^<Think>\n.*?\n</Think>\n<answer>\n.*?\n</answer>\n$"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r) for r in responses]
    return [0.5 if match else 0.0 for match in matches]

def soft_format_reward_func(completions, **kwargs) -> list[float]:
    """Reward function that checks if the completion has a specific format."""
    pattern = r"<Think>.*?</Think>\s*<answer>.*?</answer>"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r) for r in responses]
    return [0.5 if match else 0.0 for match in matches]
def evaluate_answer_content(text: str) -> float:
    # 提取<answer>标签中的内容
    answer_match = re.search(r'<answer>(.*?)</answer>', text, re.DOTALL)
    if not answer_match:
        return 0.0
    
    answer_content = answer_match.group(1).strip()
    score = 0.0
    
    # 评估答案的完整性和质量
    if len(answer_content.split('\n')) >= 2:  # 至少包含两个段落
        score += 0.2
    if '但是' in answer_content or '然而' in answer_content:  # 包含条件或限制
        score += 0.2
    if '建议' in answer_content or '注意' in answer_content:  # 包含建议或注意事项
        score += 0.2
    if '例如' in answer_content or '比如' in answer_content:  # 包含具体例子
        score += 0.2
    if any(word in answer_content for word in ['可能', '或许', '也许']):  # 考虑其他可能性
        score += 0.2
        
    return score

def evaluate_think_content(text: str) -> float:
    # 提取<Think>标签中的内容
    think_match = re.search(r'<Think>(.*?)</Think>', text, re.DOTALL)
    if not think_match:
        return 0.0
    
    think_content = think_match.group(1).strip()
    score = 0.0
    
    # 评估推理的完整性和质量
    if len(think_content.split('\n')) >= 3:  # 至少有3个推理步骤
        score += 0.3
    if '因为' in think_content or '所以' in think_content:  # 包含因果关系
        score += 0.3
    if '首先' in think_content or '然后' in think_content or '最后' in think_content:  # 有清晰的推理步骤
        score += 0.2
    if '可能' in think_content and '但是' in think_content:  # 考虑多种可能性
        score += 0.2
        
    return score
def think_quality_reward_func(completions, **kwargs) -> list[float]:
    """评估推理过程的质量"""
    responses = [completion[0]['content'] for completion in completions]
    return [evaluate_think_content(r) for r in responses]

def answer_quality_reward_func(completions, **kwargs) -> list[float]:
    """评估答案的完整性和质量"""
    responses = [completion[0]['content'] for completion in completions]
    return [evaluate_answer_content(r) for r in responses]
    

def count_xml(text) -> float:
    count = 0.0
    if text.count("<Think>\n") == 1:
        count += 0.125
    if text.count("\n</Think>\n") == 1:
        count += 0.125
    if text.count("\n<answer>\n") == 1:
        count += 0.125
        count -= len(text.split("\n</answer>\n")[-1])*0.001
    if text.count("\n</answer>") == 1:
        count += 0.125
        count -= (len(text.split("\n</answer>")[-1]) - 1)*0.001
    return count

def xmlcount_reward_func(completions, **kwargs) -> list[float]:
    contents = [completion[0]["content"] for completion in completions]
    return [count_xml(c) for c in contents]

"""<a name="Train"></a>
### Train the model

Now set up GRPO Trainer and all configurations!
"""

# wandb回调
class WandbCallback(TrainerCallback):
    def __init__(self):
        self.logs = []  # 用于存储训练过程中的日志
        
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            # 只存储日志，不立即同步
            log_dict = {}
            if "reward" in logs:
                log_dict.update({
                    "total_reward": logs["reward"],
                    "reward_std": logs.get("reward_std", 0),
                })
            log_dict.update(logs)
            self.logs.append(log_dict)
    
    def upload_logs(self):
        """训练结束后上传所有日志"""
        logger.info("开始上传训练日志到wandb...")
        for log in self.logs:
            wandb.log(log)
        logger.info("wandb日志上传完成")

class TrainingProgressCallback(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % 10 == 0:  # 每10步记录一次
            logger.info(f"训练进度: {state.global_step}/{args.max_steps} 步")
            
# 训练配置
training_args = GRPOConfig(
    use_vllm = False,  # 禁用 vLLM
    learning_rate = float(config['training']['learning_rate']),  # 确保转换为float类型
    adam_beta1 = 0.9,
    adam_beta2 = 0.99,
    weight_decay = float(config['training']['weight_decay']),
    warmup_ratio = float(config['training']['warmup_ratio']),
    lr_scheduler_type = "cosine",
    optim = "adamw_8bit",
    logging_steps = config['training']['logging_steps'],
    bf16 = is_bfloat16_supported(),
    fp16 = not is_bfloat16_supported(),
    per_device_train_batch_size = config['training']['batch_size'],
    gradient_accumulation_steps = config['training']['gradient_accumulation_steps'],
    num_generations = 2,
    max_prompt_length = 256,
    max_completion_length = config['model']['max_seq_length'],
    max_steps = config['training']['max_steps'],
    save_steps = config['training']['save_steps'],
    max_grad_norm = float(config['training']['max_grad_norm']),
    report_to = "wandb" if config['wandb']['enabled'] else "none",
    output_dir = config['output']['dir'],
    resume_from_checkpoint = True,  # 支持断点续训
)

# 在训练器初始化时添加更多参数信息
logger.info("训练配置:")
logger.info(f"学习率: {config['training']['learning_rate']}")
logger.info(f"批次大小: {config['training']['batch_size']}")
logger.info(f"梯度累积步数: {config['training']['gradient_accumulation_steps']}")
logger.info(f"最大训练步数: {config['training']['max_steps']}")
logger.info(f"保存步数: {config['training']['save_steps']}")
logger.info(f"权重衰减: {config['training']['weight_decay']}")
logger.info(f"预热比例: {config['training']['warmup_ratio']}")
logger.info(f"最大梯度范数: {config['training']['max_grad_norm']}")

def log_gpu_memory():
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            memory_allocated = torch.cuda.memory_allocated(i) / 1024**2
            memory_reserved = torch.cuda.memory_reserved(i) / 1024**2
            logger.info(f"GPU {i} 内存使用: {memory_allocated:.1f}MB (已分配) / {memory_reserved:.1f}MB (已预留)")

def main():
    try:
        # 创建输出目录
        os.makedirs(config['output']['dir'], exist_ok=True)
        logger.info(f"创建输出目录: {config['output']['dir']}")
        
        # 创建日志目录
        os.makedirs(config['logging']['save_path'], exist_ok=True)
        logger.info(f"创建日志目录: {config['logging']['save_path']}")
        
        set_ulimit()
        logger.info("系统文件句柄限制已设置")
        
        wandb_callback = None
        if config['wandb']['enabled']:
            logger.info("初始化wandb...")
            wandb.init(
                project=config['wandb']['project'],
                name=f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                config=config,
                mode="offline"  # 设置为离线模式
            )
            wandb_callback = WandbCallback()
            logger.info("wandb初始化完成(离线模式)")

        logger.info("开始加载数据集...")
        dataset = get_gsm8k_questions(config)
        
        logger.info("初始化训练器...")
        logger.info(f"训练参数: learning_rate={config['training']['learning_rate']}, "
                   f"batch_size={config['training']['batch_size']}, "
                   f"max_steps={config['training']['max_steps']}")
        trainer = GRPOTrainer(
            model = model,
            processing_class = tokenizer,
            reward_funcs = [
                xmlcount_reward_func,
                soft_format_reward_func,
                strict_format_reward_func,
                think_quality_reward_func,
                answer_quality_reward_func,
                correctness_reward_func,
                think_quality_reward_len_func,
                answer_quality_reward_len_func
            ],
            args = training_args,
            train_dataset = dataset,
        )

        if wandb_callback:
            trainer.add_callback(wandb_callback)
        trainer.add_callback(TrainingProgressCallback())
        logger.info("训练器初始化完成")

        with manage_temp_dir():
            try:
                logger.info("开始训练过程...")
                trainer.train()
                logger.info("训练过程完成")
                
                # 训练完成后上传日志
                if wandb_callback:
                    wandb_callback.upload_logs()
                    wandb.finish()
                    logger.info("wandb同步完成")
            finally:
                if config['wandb']['enabled']:
                    wandb.finish()

        # 保存模型
        if config['output']['save_model']:
            logger.info(f"保存模型到: {os.path.join(config['output']['dir'], config['output']['model_name'])}")
            model.save_lora(
                os.path.join(config['output']['dir'], 
                config['output']['model_name'])
            )
            logger.info("模型保存完成")
            
        logger.info("所有训练流程完成!")
        
    except Exception as e:
        logger.error(f"训练过程发生错误: {e}", exc_info=True)  # 添加exc_info=True以输出完整错误栈
        if config['wandb']['enabled']:
            wandb.finish()
        raise

    # 在关键点调用
    log_gpu_memory()

if __name__ == "__main__":
    main()
    
    # 以下测试代码应该条件执行
    if config.get('test', {}).get('enabled', False):
        logger.info("开始测试模型...")
        text = tokenizer.apply_chat_template([
            {"role" : "user", "content" : "草莓里有几个r？"},
        ], tokenize = False, add_generation_prompt = True)
        # ... 测试代码 ...

