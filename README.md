# verl+

在 verl 的基础上，添加了对 Process Reward Model、LLM-as-a-Judge 和输出可视化的支持。

## Table of Contents

- [Installation](#installation)
- [Dataset Preparation](#dataset-preparation)
- [Usage](#usage)
- [Evaluation](#evaluation)
- [Configuration](#configuration)

## Installation

### Prerequisites

- Python >= 3.11.0
- PyTorch >= 2.0.0
- CUDA >= 12.4

### Environment Setup

For conda users:
```bash
conda create -n verl_plus python=3.11
conda activate verl_plus
```

### Install from source

```bash
git clone https://github.com/BiNLP/verl
cd verl
pip install -e .
```

### Install dependencies

```bash
pip install vllm==0.8.5
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
wget https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp311-cp311-linux_x86_64.whl
pip install flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp311-cp311-linux_x86_64.whl

pip install -r requirements.txt
```


## Dataset Preparation

### GSM8K
```
python3 examples/data_preprocess/gsm8k.py --local_dir data/gsm8k
```

### LogiQA
```
python3 examples/data_preprocess/logiqa.py --local_dir data/logiqa
```

## Usage


```bash
ray start --head --port 6379
sh bash_scripts/*.sh

```



## Evaluation
### Method 1:
一般使用我自己的评估脚本 [LogiEval](https://github.com/BiNLP/LogiEval)
### Method 2:
使用 lighteval：
```
sh bash_scripts/eval.sh
```

## Configurations

### Vllm online inference sevice
```
sh bash_scripts/VllmBackend/start_vllm_server.sh
```

### Process Reward Model
```
PRM_PATH="/path/to/prm/model"
reward_model.worker_type = 'prm'
```
### PRM + LLM-as-a-Judge
```
reward_model.worker_type = 'judge'
```
### PRM + Async LLM-as-a-Judge
```
reward_model.worker_type = 'async_judge'
```

