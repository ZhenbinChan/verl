# verl + Tree Search
- 2025.06: 新增对于 batch 的可视化 (输入、输出、Reward Score)
- 2025.08: 新增对于数据集 LogiQA 等的支持
- 2025.08: 新增 Evaluation 方法
- 2025.09: 新增对于 PRM 的支持
- 2025.10: 新增 LLM-as-Judege 作为 Process Reward Model
- 2024.02: 新增对于 Tree Search 的支持（复现 TreeRL，但是是 step-level）
- 2024.04: 将 TreeRL 的官方代码整合到训练阶段；整合 FOL 翻译和验证的代码；

## Table of Contents

- [Installation](#installation)
- [Dataset Preparation](#dataset-preparation)
- [Evaluation](#evaluation)
- [LLM-as-Judge & PRM](#LLM-as-Judge-&-PRM)
- [Tree Sampling for RL Training](#Tree-Sampling-for-RL-Training)

# Installation

## Prerequisites
- Python >= 3.11.0 （3.10 may raise some bug in deepseed)
- PyTorch >= 2.0.0 (2.6.0 is Recommended)
- CUDA >= 12.4
- ray==2.48.0
- vllm==0.8.5.post1

## Environment Setup

For conda users:
```bash
conda create -n verl_plus python=3.11
conda activate verl_plus
```

## Install from source

```bash
git clone https://github.com/BiNLP/verl
cd verl
pip install -e .
```

## Install dependencies

```bash
pip install vllm==0.8.5
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
wget https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp311-cp311-linux_x86_64.whl
pip install flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp311-cp311-linux_x86_64.whl

pip install -r requirements.txt
```


# Dataset Preparation

### GSM8K
```
python3 examples/data_preprocess/gsm8k.py --local_dir data/gsm8k
```

### LogiQA
```
python3 examples/data_preprocess/logiqa.py --local_dir data/logiqa
```
### LogiQA for Tree Sampling
```
python3 examples/data_preprocess/logiqa_tree.py --local_dir data/logiqa_tree
```
### LogiQA for TreeRL
```
python3 examples/data_preprocess/logiqa_action.py --local_dir data/logiqa_action
```


# Evaluation
### Method 1:
[LogiEval](https://github.com/BiNLP/LogiEval) is our developed evaluation framework that makes customization more convenient.
### Method 2 (Recommended):
使用 lighteval：
```
sh bash_scripts/eval.sh
```


# LLM-as-Judge & PRM
If you need to use LLM-as-Judge as a Process Reward Model (Generative PRM), you need to start an LLM service first and then call it via the vllm API.

### Vllm online inference sevice
```
sh bash_scripts/VllmBackend/start_vllm_server.sh
```


# Reward Manager
Reward Manager is an interface that uses a reward function. It is recommended to specify a reward function regardless of whether a reward model is used, because when too much validation is performed, the reward model is not supported, and the reward function will still be called for evaluation.
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


# Tree Sampling for RL Training

## TreeRL for Training

```sh bash_scripts/EntropyChain_LogiQA_smoke.sh```



## Step-level TreeRL

We use LogiQA as the training set, please pre-processing the dataset before start training.
The startup script is located at ```bash_script/Step_TreeRL_LogiQA_GAE```, and the key parameters are as following:
```
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=tree_gae \ # TreeRL 的 Value 并不使用 Critic Model，而是直接用 Backpropagation 之后的结点的 Value 作为 Value
    ...
    trainer.tree_sampling=True \
    reward_model.reward_manager='tree' \ # 这个 RewardManager 虽然调用了 Reward Function，但是只是为了记录，我们直接使用 TreeRL 的 Reward
    trainer.branch_level='step' \ # 'token' 的话就会每个 token 作为一个结点，对于长的推理序列，树会非常大！！！！
    trainer.step_reward_type='treerl' \ # 根据论文里面的公式，使用当前结点和 root 结点及其双亲的 Value 来计算 Reward；如果是 'fol'，则会将每一步翻译成 FOL 并使用 Z3 返回结果（Fragile）！！！！
    trainer.tree_rounds=1 \ # 分支轮数
    trainer.tree_top_k=1 \ # 每次分支出多少条，目前是只在最高熵的 step 进行分支

```
## FOL as Process Reward

```
trainer.step_reward_type='treerl' 
```
- ver/utils/nl2fol.py: Translating natural langugage to First-order-logic formulation。
- verl/tuils/fol_to_python_converter.py: Translating the FOL formulation into python code with Z3 verifier, and return the result (satify/unsatify).




# TODO
1. 将 MCTS sampling 和 FOL as PRM 整合到训练流程