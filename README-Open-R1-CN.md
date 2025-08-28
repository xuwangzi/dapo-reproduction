# Open R1

*完全开源的DeepSeek-R1复现项目。这个仓库正在开发中，让我们一起构建它！*

**目录**  
1. [概述](#概述)  
2. [攻击计划](#攻击计划)  
3. [安装](#安装)  
4. [训练模型](#训练模型)  
   - [SFT](#sft)  
   - [GRPO](#grpo)  
5. [评估模型](#评估模型)  
6. [复现Deepseek的评估结果](#复现deepseek的评估结果)  
7. [数据生成](#数据生成)  
   - [从小型蒸馏R1模型生成数据](#从小型蒸馏r1模型生成数据)  
   - [从DeepSeek-R1生成数据](#从deepseek-r1生成数据)  
8. [贡献](#贡献)

## 概述

这个仓库的目标是构建R1流水线中缺失的部分，使每个人都能复现并在此基础上进行构建。该项目在设计上保持简单，主要包括：

- `src/open_r1`: 包含训练模型以及生成合成数据的脚本：
    - `grpo.py`: 在给定数据集上使用GRPO训练模型。
    - `sft.py`: 在数据集上对模型执行简单的SFT。
    - `generate.py`: 使用[Distilabel](https://github.com/argilla-io/distilabel)从模型生成合成数据。
- `Makefile`: 包含利用上述脚本进行R1流水线中每个步骤的易运行命令。

### 攻击计划

我们将使用DeepSeek-R1[技术报告](https://github.com/deepseek-ai/DeepSeek-R1)作为指南，该报告大致可以分解为三个主要步骤：

* 步骤1: 通过从DeepSeek-R1蒸馏高质量语料库来复制R1-Distill模型。
* 步骤2: 复制DeepSeek用于创建R1-Zero的纯RL流水线。这可能涉及为数学、推理和代码策划新的大规模数据集。
* 步骤3: 展示我们可以通过多阶段训练从基础模型到RL调优。

<center>
    <img src="assets/plan-of-attack.png" width="500">
</center>

## 新闻 🗞️

* **🧑‍🍳 [2025/05/26] (步骤1完成!)** 我们发布了[**Mixture-of-Thoughts**](https://huggingface.co/datasets/open-r1/Mixture-of-Thoughts)——一个经过精心策划的推理数据集，包含从R1蒸馏出的35万个验证轨迹。该数据集涵盖数学、编程和科学任务，旨在教导语言模型逐步推理。我们还提供了训练[OpenR1-Distill-7B](https://huggingface.co/open-r1/OpenR1-Distill-7B)的配方，它复制了[deepseek-ai/DeepSeek-R1-Distill-Qwen-7B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B)的推理能力，标志着Open R1项目步骤1的完成。
* **⚡️ [2025/03/11] [(更新 #3)](https://huggingface.co/blog/open-r1/update-3):** 我们发布了[**CodeForces-CoTs**](https://huggingface.co/datasets/open-r1/codeforces-cots)数据集，包含1万个竞技编程问题和10万个从R1蒸馏的解决方案。我们还发布了IOI24：一个由国际奥林匹克_非常_困难问题组成的新基准。在CodeForces-CoTs上训练的7B Qwen模型在IOI24上可以超越Claude 3.7 Sonnet，而32B模型可以超越R1本身。
* **∞ [2025/02/10] [(更新 #2)](https://huggingface.co/blog/open-r1/update-2):** 我们发布了[**OpenR1-Math-220k**](https://huggingface.co/datasets/open-r1/OpenR1-Math-220k)数据集，包含从R1在新版本NuminaMath上蒸馏的22万个轨迹。在此数据集上训练的模型与DeepSeek的蒸馏模型性能相匹配。
* **🔥 [2025/02/02] [(更新 #1)](https://huggingface.co/blog/open-r1/update-1):** 我们实现了[训练](https://github.com/huggingface/open-r1?tab=readme-ov-file#training-models)、[推理](https://github.com/huggingface/open-r1?tab=readme-ov-file#data-generation)和[评估](https://github.com/huggingface/open-r1?tab=readme-ov-file#reproducing-deepseeks-evaluation-results)流水线的第一部分。让我们开始吧！

## 安装

> [!CAUTION]
> 库依赖CUDA 12.4。如果您看到与分段错误相关的错误，请使用`nvcc --version`仔细检查您系统正在运行的版本。

要运行此项目中的代码，首先，使用例如`uv`创建一个Python虚拟环境。
要安装`uv`，请遵循[UV安装指南](https://docs.astral.sh/uv/getting-started/installation/)。

> [!NOTE]
> 作为快捷方式，运行`make install`来设置开发库（详细说明如下）。之后，如果一切设置正确，您可以尝试Open-R1模型。

```shell
uv venv openr1 --python 3.11 && source openr1/bin/activate && uv pip install --upgrade pip
```

> [!TIP]
> 对于Hugging Face集群用户，在您的`.bashrc`中添加`export UV_LINK_MODE=copy`以抑制来自`uv`的缓存警告

接下来，安装vLLM和FlashAttention：

```shell
uv pip install vllm==0.8.5.post1
uv pip install setuptools && uv pip install flash-attn --no-build-isolation
```

这也将安装PyTorch `v2.6.0`，使用这个版本是**非常重要的**，因为vLLM二进制文件是为它编译的。然后您可以通过`pip install -e .[LIST OF MODES]`为您的特定用例安装其余依赖项。对于大多数贡献者，我们推荐：

```shell
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e ".[dev]"
```

接下来，登录您的Hugging Face和Weights and Biases账户，如下所示：

```shell
huggingface-cli login
wandb login
```

最后，检查您的系统是否安装了Git LFS，以便您可以加载和推送模型/数据集到Hugging Face Hub：

```shell
git-lfs --version
```

如果未安装，运行：

```shell
sudo apt-get install git-lfs
```

## 训练模型

> [!NOTE]
> 下面的训练命令是为8 x H100s (80GB)节点配置的。对于不同的硬件和拓扑，您可能需要调整批量大小和梯度累积步数。

我们支持使用DDP或DeepSpeed (ZeRO-2和ZeRO-3)训练模型。例如，要在从DeepSeek-R1蒸馏的数据集上执行SFT，其中包含推理轨迹如[open-r1/Mixture-of-Thoughts](https://huggingface.co/datasets/open-r1/Mixture-of-Thoughts)，运行：

```shell
# 通过命令行训练
accelerate launch --config_file=recipes/accelerate_configs/zero3.yaml src/open_r1/sft.py \
    --model_name_or_path open-r1/Qwen2.5-Math-7B-RoPE-300k \
    --dataset_name open-r1/Mixture-of-Thoughts \
    --dataset_config all \
    --eos_token '<|im_end|>' \
    --learning_rate 4.0e-5 \
    --num_train_epochs 5 \
    --max_seq_length 32768 \
    --per_device_train_batch_size 2 \
    --gradient_checkpointing \
    --bf16 \
    --use_liger_kernel \
    --output_dir data/OpenR1-Distill-7B

# 通过YAML配置训练
accelerate launch --config_file recipes/accelerate_configs/zero3.yaml src/open_r1/sft.py \
    --config recipes/OpenR1-Distill-7B/sft/config_distill.yaml
```

目前，支持以下任务：

* 监督微调 `sft`
* 群体相对策略优化 `grpo`

> [!TIP]
> 如果您增加/减少GPU数量，我们建议也相应地增加每设备批量大小或梯度累积步数，以保持全局批量大小不变。

默认情况下，这些脚本会将每个模型推送到您的Hugging Face Hub用户名，即`{username}/{model_name}-{task}`。您可以通过将参数附加到命令中来覆盖每个YAML配置中的参数，如下所示：

```shell
# 将基础模型更改为较小的变体
accelerate launch --config_file recipes/accelerate_configs/zero3.yaml src/open_r1/sft.py \
    --config recipes/OpenR1-Distill-7B/sft/config_distill.yaml \
    --model_name_or_path Qwen/Qwen3-0.6B-Base \
    --hub_model_id OpenR1-Distill-0.6B \
    --output_dir data/OpenR1-Distill-0.6B
```

如果您还希望覆盖Weights and Biases默认设置，可以如下操作：

```shell
accelerate launch --config_file recipes/accelerate_configs/zero3.yaml src/open_r1/sft.py \
    --config recipes/OpenR1-Distill-7B/sft/config_distill.yaml
    --wandb_entity huggingface --wandb_project open-r1 --run_name Qwen2.5-1.5B-GRPO
```

**🚨 警告 🚨**

大多数基础模型如`meta-llama/Llama-3.2-1B`没有聊天模板，所以我们在训练期间设置ChatML作为默认值。但是，对于Qwen基础模型如`Qwen/Qwen2.5-1.5B`，聊天模板在分词器中是预定义的，所以必须相应地设置EOS令牌，例如

```diff
# 为Qwen基础模型将EOS令牌与聊天模板对齐
accelerate launch --config_file=recipes/accelerate_configs/zero3.yaml src/open_r1/sft.py \
    --model_name_or_path Qwen/Qwen2.5-1.5B \
+   --eos_token '<|im_end|>'
    --dataset_name open-r1/Mixture-of-Thoughts \
    --dataset_config all \
    --learning_rate 4.0e-5 \
    --num_train_epochs 1 \
    --max_seq_length 32768 \
    --per_device_train_batch_size 16 \
    --gradient_checkpointing \
    --bf16 \
    --use_liger_kernel \
    --output_dir data/Qwen2.5-1.5B-Open-R1-Distill
```

如果您希望使用自定义聊天模板（例如Llama或Gemma），则必须提供聊天模板和相关的EOS令牌：

```diff
# 将EOS令牌与自定义聊天模板对齐
accelerate launch --config_file=recipes/accelerate_configs/zero3.yaml src/open_r1/sft.py \
    --model_name_or_path meta-llama/Llama-3.2-1B \
+   --chat_template "$(cat llama_chat_template.jinja)" \
+   --eos_token '<|eot_id|>' \
    --dataset_name open-r1/Mixture-of-Thoughts \
    --dataset_config all \
    --learning_rate 4.0e-5 \
    --num_train_epochs 1 \
    --max_seq_length 32768 \
    --per_device_train_batch_size 16 \
    --gradient_checkpointing \
    --bf16 \
    --use_liger_kernel \
    --output_dir data/Llama-3.2-1B-Open-R1-Distill
```

### SFT蒸馏

我们提供了一个配方来复现[deepseek-ai/DeepSeek-R1-Distill-Qwen-7B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B)的推理能力，从相同的基础模型开始。要做到这一点，运行：

```shell
ACCELERATE_LOG_LEVEL=info accelerate launch --config_file recipes/accelerate_configs/zero3.yaml \
    src/open_r1/sft.py \
    --config recipes/OpenR1-Distill-7B/sft/config_distill.yaml
```

结果将是一个类似[open-r1/OpenR1-Distill-7B](https://huggingface.co/open-r1/OpenR1-Distill-7B)的模型，具有以下下游性能：

| 模型                       | AIME 2024 | MATH-500 | GPQA Diamond | LiveCodeBench v5 |
|-----------------------------|-----------|----------|--------------|------------------|
| OpenR1-Distill-7B           | 52.7      | 89.0     | 52.8         | 39.4             |
| DeepSeek-R1-Distill-Qwen-7B | 51.3      | 93.5     | 52.4         | 37.4             |

您可以调整YAML配置以在不同的基础模型或数据集上训练。

### GRPO

我们使用TRL的[vLLM后端](https://huggingface.co/docs/trl/speeding_up_training?vllm+examples=GRPO#vllm-for-fast-generation-in-online-methods)来扩展跨多个节点的大型模型训练。对于跨8个GPU的小型模型的单节点训练，使用`vllm_mode="colocate"`在与训练脚本相同的进程中运行vLLM：

```shell
ACCELERATE_LOG_LEVEL=info \
    accelerate launch --config_file recipes/accelerate_configs/zero3.yaml \
    src/open_r1/grpo.py --config recipes/DeepSeek-R1-Distill-Qwen-1.5B/grpo/config_demo.yaml \
    --vllm_mode colocate
```

> [!WARNING]
> 蒸馏DeepSeek模型中使用的聊天模板省略了`<think>`和`</think>`标签内推理块的内容。它还用`<think>`预填充助手响应，这会干扰格式奖励函数。为了处理这个问题，重要的是要覆盖聊天模板，如在例如[recipes/DeepSeek-R1-Distill-Qwen-1.5B/grpo/config_demo.yaml](./recipes/DeepSeek-R1-Distill-Qwen-1.5B/grpo/config_demo.yaml)中所做的。

对于N+1个节点的多节点训练，其中1个节点运行vLLM服务器，N个节点运行训练，我们提供了一个示例Slurm脚本。例如，要在1+1个节点上使用数据并行运行上述示例，运行：

```shell
sbatch --nodes=2 slurm/train.slurm --model Qwen2.5-1.5B-Instruct --task grpo --config demo --accelerator zero2 --dp 8 --tp 1
```

有关更多详细信息，请参阅[在Slurm集群上启动作业](#在slurm集群上启动作业)部分。

#### GRPO数据集过滤

我们提供支持通过生成和计算可验证任务的通过率来过滤数据集，请参阅此[README](scripts/pass_rate_filtering/README.md)

#### 👨‍💻 使用代码解释器训练

我们提供了一个`code`奖励函数，用于在训练期间执行策略生成的代码。目前，此奖励函数针对代码竞赛如[Codeforces](https://codeforces.com)，其中解决方案针对一组测试用例执行，总体成功率作为最终奖励返回。为了确保安全执行，我们支持多个沙箱提供商：

1. [E2B](https://e2b.dev) - 快速、基于云的沙箱，专注于Python执行
2. [Morph](https://cloud.morph.so/web/) - 基于云的沙箱，具有更广泛的语言支持 - Python/JS/C++/Rust

要使用代码奖励函数，首先安装必要的依赖项：

```shell
uv pip install -e '.[code]'
```

##### E2B提供商

要使用E2B沙箱，创建一个`.env`文件并添加您的E2B API令牌：

```
E2B_API_KEY="e2b_xxx"
```

##### Morph提供商

要使用Morph，首先安装morphcloud包：

```shell
pip install morphcloud
```

然后将您的Morph API令牌添加到`.env`文件：

```
MORPH_API_KEY="YOUR_MORPH_API_KEY"
```

要指定使用哪个提供商，在您的配置中添加`provider_type`参数：

```yaml
# 对于E2B
provider_type: e2b

# 对于Morph
provider_type: morph
```

##### 数据集要求

确保您的数据集包含具有以下模式的`verification_info`列（采用PrimeIntellect优秀的可验证问题[数据集](https://huggingface.co/collections/PrimeIntellect/synthetic-1-67a2c399cfdd6c9f7fae0c37)）：

```python
{
    "language": "python",  # Morph支持更多语言，包括C++、Java等。
    "test_cases": [
        {
            "input": "4\n4\n0001\n1000\n0011\n0111\n3\n010\n101\n0\n2\n00000\n00001\n4\n01\n001\n0001\n00001\n",
            "output": "1\n3 \n-1\n0\n\n2\n1 2 \n",
            "type": "stdin_stdout",
        }
    ],
}
```

例如，要在Python问题上训练小型模型，启动vLLM服务器：

```shell
CUDA_VISIBLE_DEVICES=0 trl vllm-serve --model Qwen/Qwen2.5-1.5B-Instruct
```

然后运行训练：

```shell
CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 ACCELERATE_LOG_LEVEL=info \
    accelerate launch --config_file recipes/accelerate_configs/zero2.yaml --num_processes=7 \
    src/open_r1/grpo.py --config recipes/Qwen2.5-1.5B-Instruct/grpo/config_demo_code.yaml
```

##### 使用路由器服务

当在沙箱服务上执行太多脚本时，可能会受到速率限制。对于两个提供商，我们提供可以在CPU节点上启动的路由器脚本：

对于E2B：
```shell
sbatch slurm/e2b_router.slurm
```

对于Morph：
```shell
sbatch slurm/morph_router.slurm
```

然后在您的训练YAML配置中添加路由器URL：
```yaml
# 对于E2B
e2b_router_url: 1.2.3.4:8000

# 对于Morph
morph_router_url: 1.2.3.4:8000
```

端口应与启动路由器时使用的端口匹配。
所有训练作业可以共享相同的路由器IP，这将确保并行执行得到正确管理。

#### 竞技编程问题：IOI和CodeForces

我们为执行来自[IOI](https://hf.co/datasets/open-r1/ioi)和[CodeForces](https://huggingface.co/datasets/open-r1/codeforces)的问题分别提供了`ioi_code_reward`和`cf_code_reward`奖励函数。您可以使用[piston](https://github.com/engineer-man/piston)或Morph（目前仅IOI）作为您的执行提供商。

##### Piston

要使用Piston：
1. 让piston工作器运行，请参阅[slurm/piston/README.md](./slurm/piston/README.md)
2. 将您的环境变量`PISTON_ENDPOINTS`设置为`slurm`或piston工作器端点列表

对于IOI：

3. 在您的配置中，使用`ioi_provider: "piston"`

对于CodeForces：

3. 下载生成的（困难）测试用例：
```
# 更改PATH_TO_SAVE_TESTCASES。根据您机器的容量增加--max-workers
huggingface-cli download open-r1/codeforces --repo-type=dataset --include='generated_tests/*.parquet' --max-workers=8 --local-dir PATH_TO_SAVE_TESTCASES 
```
4. 将路径保存在.env中：
```
CF_TESTS_FOLDER=PATH_TO_SAVE_TESTCASES
```

##### Morph

Morph是一个基于云的解决方案，为运行代码提供沙箱环境。要使用它：
1. 安装Morph客户端：`pip install morphcloud`
2. 将您的Morph API密钥添加到`.env`文件：`MORPH_API_KEY="your_key_here"`
3. 在您的配置中，使用`ioi_provider: "morph"`

##### 示例配方
对于IOI：

请参阅[示例配方](./recipes/Qwen2.5-1.5B-Instruct/grpo/config_demo_code_ioi.yaml)了解如何使用IOI奖励函数：

```shell
ACCELERATE_LOG_LEVEL=info accelerate launch --config_file recipes/accelerate_configs/zero2.yaml \
    --num_processes=7 src/open_r1/grpo.py \
    --config recipes/Qwen2.5-1.5B-Instruct/grpo/config_demo_code_ioi.yaml
```

对于CodeForces：

```shell
sbatch --job-name=cf-grpo --nodes=2 slurm/train.slurm --model Qwen2.5-Coder-7B-Instruct --task grpo --config codeforces --accelerator zero3 --dp 8 --tp 1
```

### 在Slurm集群上启动作业

如果您有访问Slurm集群的权限，我们提供了一个`slurm/train.slurm`脚本，它将自动为您排队训练作业。以下是如何使用它：

```shell
sbatch --job-name=open_r1 --nodes=1 slurm/train.slurm --model {model_name} --task {task} --config {config_suffix} --accelerator {accelerator}
```

这里`{model_name}`和`{task}`如上所定义，而`{config_suffix}`指的是特定配置，`{accelerator}`指的是`recipes/accelerate_configs`中🤗 Accelerate配置的选择。如果您希望覆盖默认配置参数，可以通过附加空格分隔的字符串来提供它们，如`'--arg1=value1 --arg2=value2'`。以下是在8个GPU的1个节点上运行SFT的具体示例：

```shell
sbatch --job-name=open_r1 --nodes=1 slurm/train.slurm --model OpenR1-Distill-7B --task sft --config distill --accelerator zero3
```

您可以通过增加`--nodes`标志来扩展节点数量。

对于GRPO，我们使用1个节点用于vLLM服务器，N个节点用于训练。例如，要在1+1个节点上使用混合数据和张量并行运行GRPO，运行：

```shell
sbatch --job-name=open_r1 --nodes=2 slurm/train.slurm --model Qwen2.5-1.5B-Instruct --task grpo --config demo --accelerator zero2 --dp 4 --tp 2
```

> [!NOTE]
> `slurm/train.slurm`中的配置针对Hugging Face计算集群进行了优化，可能需要调整以适应您自己的计算节点。

### 自定义数据集混合

要将多个数据集组合为单个训练混合，您可以在YAML配置文件中指定`dataset_mixture`参数。以下是执行此操作的模板：

```yaml
dataset_mixture:
  datasets:                     # 要包含在混合中的数据集列表
    - id: dataset_1             # Hub数据集ID
      config: config_name_1     # 数据集配置的名称
      split: split_1            # 要从数据集使用的分割
      columns:                  # 要保留的列
        - column_1              
        - column_2    
      weight: 0.25              # 要使用的数据集分数
    - id: dataset_2
      config: config_name_2
      split: split_2
      columns:                  
        - column_1              
        - column_2   
      weight: 0.5
  seed: 42                      # 用于混洗组合数据集的种子
  test_split_size: 0.1          # 用于测试分割的混合分数
```

## 评估模型

我们使用`lighteval`来评估模型。对于适合单个GPU的模型，运行：

```shell
export VLLM_WORKER_MULTIPROC_METHOD=spawn # vLLM所需
MODEL=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
MODEL_ARGS="model_name=$MODEL,dtype=bfloat16,max_model_length=32768,gpu_memory_utilization=0.8,generation_parameters={max_new_tokens:32768,temperature:0.6,top_p:0.95}"
OUTPUT_DIR=data/evals/$MODEL

# AIME 2024
TASK=aime24
lighteval vllm $MODEL_ARGS "lighteval|$TASK|0|0" \
    --use-chat-template \
    --output-dir $OUTPUT_DIR

# MATH-500
TASK=math_500
lighteval vllm $MODEL_ARGS "lighteval|$TASK|0|0" \
    --use-chat-template \
    --output-dir $OUTPUT_DIR

# GPQA Diamond
TASK=gpqa:diamond
lighteval vllm $MODEL_ARGS "lighteval|$TASK|0|0" \
    --use-chat-template \
    --output-dir $OUTPUT_DIR

# LiveCodeBench
lighteval vllm $MODEL_ARGS "extended|lcb:codegeneration|0|0" \
    --use-chat-template \
    --output-dir $OUTPUT_DIR 
```

要在多个GPU上增加吞吐量，使用_数据并行_如下：

```shell
NUM_GPUS=8
MODEL=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
MODEL_ARGS="model_name=$MODEL,dtype=bfloat16,data_parallel_size=$NUM_GPUS,max_model_length=32768,gpu_memory_utilization=0.8,generation_parameters={max_new_tokens:32768,temperature:0.6,top_p:0.95}"
TASK=aime24
OUTPUT_DIR=data/evals/$MODEL

lighteval vllm $MODEL_ARGS "lighteval|$TASK|0|0" \
    --use-chat-template \
    --output-dir $OUTPUT_DIR
```

对于需要跨GPU分片的大型模型，使用_张量并行_并运行：

```shell
NUM_GPUS=8
MODEL=deepseek-ai/DeepSeek-R1-Distill-Qwen-32B
MODEL_ARGS="model_name=$MODEL,dtype=bfloat16,tensor_parallel_size=$NUM_GPUS,max_model_length=32768,gpu_memory_utilization=0.8,generation_parameters={max_new_tokens:32768,temperature:0.6,top_p:0.95}"
TASK=aime24
OUTPUT_DIR=data/evals/$MODEL

export VLLM_WORKER_MULTIPROC_METHOD=spawn
lighteval vllm $MODEL_ARGS "lighteval|$TASK|0|0" \
    --use-chat-template \
    --output-dir $OUTPUT_DIR
```

您也可以使用`make evaluate`启动评估，指定模型、任务和可选的并行技术和GPU数量。

要在单个GPU上评估：

```shell
make evaluate MODEL=deepseek-ai/DeepSeek-R1-Distill-Qwen-32B TASK=aime24
```

要使用数据并行：

```shell
make evaluate MODEL=deepseek-ai/DeepSeek-R1-Distill-Qwen-32B TASK=aime24 PARALLEL=data NUM_GPUS=8
```

要使用张量并行：

```shell
make evaluate MODEL=deepseek-ai/DeepSeek-R1-Distill-Qwen-32B TASK=aime24 PARALLEL=tensor NUM_GPUS=8
```

## 复现Deepseek的评估结果

DeepSeek-R1论文使用每个查询4-64个响应的采样来估计`pass@1`准确率，但没有指定每个基准的具体响应数量。在下表中，我们使用以下每个查询的响应数量来估计`pass@1`准确率：

|   基准   | 每个查询的响应数量 |
|:-------------:|:-----------------------------:|
|   AIME 2024   |              64               |
|   MATH-500    |               4               |
| GPQA Diamond  |               8               |
| LiveCodeBench |              16               |

请注意，对于像AIME24这样的基准，重要的是要采样许多响应，因为只有30个问题，这可能在重复运行中引入高方差。每个提示采样多少响应的选择可能解释了我们的评估结果与DeepSeek报告的结果之间的小差异。

### AIME 2024

我们能够在约1-3个标准差内复现Deepseek在AIME 2024基准上的报告结果：

| 模型                         | AIME 2024 (🤗 LightEval) | AIME 2024 (DeepSeek报告) |
|:------------------------------|:------------------------:|:-----------------------------:|
| DeepSeek-R1-Distill-Qwen-1.5B |           30.7           |             28.9              |
| DeepSeek-R1-Distill-Qwen-7B   |           50.8           |             55.5              |
| DeepSeek-R1-Distill-Qwen-14B  |           65.9           |             69.7              |
| DeepSeek-R1-Distill-Qwen-32B  |           69.7           |             72.6              |
| DeepSeek-R1-Distill-Llama-8B  |           43.9           |             41.7              |
| DeepSeek-R1-Distill-Llama-70B |           63.0           |             70.0              |

要复现这些结果，使用以下命令：

```shell
NUM_GPUS=1 # 对于32B和70B模型设置为8
MODEL=deepseek-ai/{model_name}
MODEL_ARGS="model_name=$MODEL,dtype=bfloat16,max_model_length=32768,gpu_memory_utilization=0.8,data_parallel_size=$NUM_GPUS,generation_parameters={max_new_tokens:32768,temperature:0.6,top_p:0.95}"
OUTPUT_DIR=data/evals/$MODEL

lighteval vllm $MODEL_ARGS "lighteval|aime24|0|0" \
    --use-chat-template \
    --output-dir $OUTPUT_DIR
```

或者，您可以如下启动Slurm作业：

```shell
python scripts/run_benchmarks.py --model-id {model_id}  --benchmarks aime24
```

### MATH-500

我们能够在约1-3个标准差内复现Deepseek在MATH-500基准上的报告结果：

| 模型                         | MATH-500 (🤗 LightEval) | MATH-500 (DeepSeek报告) |
|:------------------------------|:-----------------------:|:----------------------------:|
| DeepSeek-R1-Distill-Qwen-1.5B |          83.1           |             83.9             |
| DeepSeek-R1-Distill-Qwen-7B   |          94.5           |             92.8             |
| DeepSeek-R1-Distill-Qwen-14B  |          94.1           |             93.9             |
| DeepSeek-R1-Distill-Qwen-32B  |          95.6           |             94.3             |
| DeepSeek-R1-Distill-Llama-8B  |          88.6           |             89.1             |
| DeepSeek-R1-Distill-Llama-70B |          95.1           |             94.5             |

要复现这些结果，使用以下命令：

```shell
export VLLM_WORKER_MULTIPROC_METHOD=spawn
NUM_GPUS=1 # 对于32B和70B模型设置为8
MODEL=deepseek-ai/{model_name}
MODEL_ARGS="model_name=$MODEL,dtype=bfloat16,max_model_length=32768,gpu_memory_utilization=0.8,data_parallel_size=$NUM_GPUS,generation_parameters={max_new_tokens:32768,temperature:0.6,top_p:0.95}"
OUTPUT_DIR=data/evals/$MODEL

lighteval vllm $MODEL_ARGS "lighteval|math_500|0|0" \
    --use-chat-template \
    --output-dir $OUTPUT_DIR
```

或者，您可以如下启动Slurm作业：

```shell
python scripts/run_benchmarks.py --model-id {model_id}  --benchmarks math_500
```

### GPQA Diamond

我们能够在约1-3个标准差内复现Deepseek在GPQA Diamond基准上的报告结果：

| 模型                         | GPQA Diamond (🤗 LightEval) | GPQA Diamond (DeepSeek报告) |
|:------------------------------|:---------------------------:|:--------------------------------:|
| DeepSeek-R1-Distill-Qwen-1.5B |            35.8             |               33.8               |
| DeepSeek-R1-Distill-Qwen-7B   |            50.5             |               49.1               |
| DeepSeek-R1-Distill-Qwen-14B  |            61.5             |               59.1               |
| DeepSeek-R1-Distill-Qwen-32B  |            63.1             |               62.1               |
| DeepSeek-R1-Distill-Llama-8B  |            46.7             |               49.0               |
| DeepSeek-R1-Distill-Llama-70B |            67.4             |               65.2               |

要复现这些结果，使用以下命令：

```shell
export VLLM_WORKER_MULTIPROC_METHOD=spawn
NUM_GPUS=1 # 对于32B和70B模型设置为8
MODEL=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
MODEL_ARGS="model_name=$MODEL,dtype=bfloat16,max_model_length=32768,gpu_memory_utilization=0.8,generation_parameters={max_new_tokens:32768,temperature:0.6,top_p:0.95}"
OUTPUT_DIR=data/evals/$MODEL

lighteval vllm $MODEL_ARGS "lighteval|gpqa:diamond|0|0" \
    --use-chat-template \
    --output-dir $OUTPUT_DIR
```

```shell
python scripts/run_benchmarks.py --model-id {model_id}  --benchmarks gpqa
```

### LiveCodeBench

我们能够在约1-3个标准差内复现Deepseek在LiveCodeBench代码生成基准上的报告结果：

| 模型                         | LiveCodeBench (🤗 LightEval) | LiveCodeBench (DeepSeek报告) |
|:------------------------------|:----------------------------:|:---------------------------------:|
| DeepSeek-R1-Distill-Qwen-1.5B |             16.1             |               16.9                |
| DeepSeek-R1-Distill-Qwen-7B   |             37.4             |               37.6                |
| DeepSeek-R1-Distill-Qwen-14B  |             51.3             |               53.1                |
| DeepSeek-R1-Distill-Qwen-32B  |             56.0             |             57.2                |
| DeepSeek-R1-Distill-Llama-8B  |             37.4             |               39.6                |
| DeepSeek-R1-Distill-Llama-70B |             55.9             |               57.5                |

要复现这些结果，使用以下命令：

```shell
NUM_GPUS=1 # 对于32B和70B模型设置为8，或对于较小模型使用data_parallel_size=8以提高速度
MODEL=deepseek-ai/{model_name}
MODEL_ARGS="model_name=$MODEL,dtype=bfloat16,max_model_length=32768,gpu_memory_utilization=0.8,data_parallel_size=$NUM_GPUS,generation_parameters={max_new_tokens:32768,temperature:0.6,top_p:0.95}"
OUTPUT_DIR=data/evals/$MODEL

lighteval vllm $MODEL_ARGS "extended|lcb:codegeneration|0|0" \
    --use-chat-template \
    --output-dir $OUTPUT_DIR
```

```shell
python scripts/run_benchmarks.py --model-id {model_id}  --benchmarks lcb
```

## 数据生成

### 从小型蒸馏R1模型生成数据

以下示例可以在1xH100上运行。
首先安装以下依赖项：

```shell
uv pip install "distilabel[vllm]>=1.5.2"
```

现在将以下代码片段保存到名为`pipeline.py`的文件中，并使用`python pipeline.py`运行它。它将为10个示例中的每一个生成4个输出（将存储库的用户名更改为您的组织/用户名）：

```python
from datasets import load_dataset
from distilabel.models import vLLM
from distilabel.pipeline import Pipeline
from distilabel.steps.tasks import TextGeneration


prompt_template = """\
您将得到一个问题。请逐步推理，并将您的最终答案放在\boxed{}中：
{{ instruction }}"""

dataset = load_dataset("AI-MO/NuminaMath-TIR", split="train").select(range(10))

model_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"  # 与另一个小型蒸馏r1交换

with Pipeline(
    name="distill-qwen-7b-r1",
    description="从蒸馏r1模型生成数据的流水线",
) as pipeline:

    llm = vLLM(
        model=model_id,
        tokenizer=model_id,
        extra_kwargs={
            "tensor_parallel_size": 1,
            "max_model_len": 8192,
        },
        generation_kwargs={
            "temperature": 0.6,
            "max_new_tokens": 8192,
        },
    )
    prompt_column = "problem"
    text_generation = TextGeneration(
        llm=llm, 
        template=prompt_template,
        num_generations=4,
        input_mappings={"instruction": prompt_column} if prompt_column is not None else {}
    )


if __name__ == "__main__":
    distiset = pipeline.run(dataset=dataset)
    distiset.push_to_hub(repo_id="username/numina-deepseek-r1-qwen-7b")
```

查看[HuggingFaceH4/numina-deepseek-r1-qwen-7b](https://huggingface.co/datasets/HuggingFaceH4/numina-deepseek-r1-qwen-7b)的示例数据集。

### 从DeepSeek-R1生成数据

要运行更大的DeepSeek-R1，我们使用了2个节点，每个节点有8×H100 GPU，使用此存储库中的slurm文件`slurm/generate.slurm`。首先，安装依赖项：

（现在我们需要安装[修复R1 cuda图捕获](https://github.com/vllm-project/vllm/commits/221d388cc5a836fa189305785ed7e887cea8b510/csrc/moe/moe_align_sum_kernels.cu)的vllm dev wheel）
```shell
pip install https://wheels.vllm.ai/221d388cc5a836fa189305785ed7e887cea8b510/vllm-1.0.0.dev-cp38-abi3-manylinux1_x86_64.whl --extra-index-url https://download.pytorch.org/whl/cu121

uv pip install "distilabel[vllm,ray,openai]>=1.5.2"
```

然后运行以下命令：

```shell
sbatch slurm/generate.slurm \
    --hf-dataset AI-MO/NuminaMath-TIR \
    --temperature 0.6 \
    --prompt-column problem \
    --model deepseek-ai/DeepSeek-R1 \
    --hf-output-dataset username/r1-dataset
```

> [!NOTE]  
> 当作业运行时，您可以设置通过集群登录节点的SSH隧道，通过运行`ssh -L 8265:ray_ip_head_node:8265 <login_node>`从您的计算机访问Ray仪表板，然后浏览`http://localhost:8265`

### 数据去污染

遵循[s1: Simple test-time scaling](https://huggingface.co/papers/2501.19393)，可以使用位于[scripts/decontaminate.py](./scripts/decontaminate.py)的脚本对数据进行去污染，该脚本使用8-gram对数据集进行去污染并去重。示例运行：

```shell
python scripts/decontaminate.py \
    --dataset "open-r1/verifiable-coding-problems-python" \
    --problem_column problem \
    --cleanup
```

它将针对基准数据集进行去污染，并之后删除污染的样本。如果没有提供参数`--new_dataset_name`，将重用相同的数据集，添加`_decontaminated`。它针对提示运行，对于此数据集是列`problem`，但可以提供不同的列。

脚本的参数：

```shell
usage: decontaminate.py [-h] --dataset DATASET [--split SPLIT] [--ngram_size NGRAM_SIZE] [--problem_column PROBLEM_COLUMN] [--cleanup] [--new_dataset_name NEW_DATASET_NAME]

options:
  -h, --help            显示此帮助消息并退出
  --dataset DATASET     要检查污染的数据集名称。
  --split SPLIT         要检查污染的分割，默认为`train`。
  --ngram_size NGRAM_SIZE
                        要构建的n-gram大小，默认为8。
  --problem_column PROBLEM_COLUMN
                        包含问题（提示）的列的名称。
  --cleanup           是否在推送数据集之前删除污染的行。
  --new_dataset_name NEW_DATASET_NAME
                        数据集的新名称。如果未提供，将重用名称并在名称中添加`_decontaminated`。
```

## 贡献

欢迎贡献。请参考https://github.com/huggingface/open-r1/issues/23。

## 致谢

该项目是通过开源AI社区中许多团体和个人的集体努力构建的。我们特别感谢vLLM和SGLang团队创建了高性能工具来扩展GRPO的推出。我们还感谢[OpenThoughts](https://www.open-thoughts.ai)、[Prime Intellect](https://www.primeintellect.ai)和[General Reasoning](https://gr.inc)团队创建和分享高质量的推理数据集。

## 引用

如果您发现此项目在您自己的工作中有用，请考虑如下引用：

```
@misc{openr1,
    title = {Open R1: 完全开源的DeepSeek-R1复现},
    url = {https://github.com/huggingface/open-r1},
    author = {{Hugging Face}},
    month = {January},
    year = {2025}
}
```
