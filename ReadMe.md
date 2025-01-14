# SPAM
This repo contains the pre-release version of SPAM optimizer, proposed by [SPAM: Spike-Aware Adam with Momentum Reset for Stable LLM Training](https://arxiv.org/pdf/2501.06842).


We introduce  SPAM, a novel optimizer with momentum reset and spike-aware clipping that outperforms existing methods like Adam and Adafactor on various tasks, including LLM training, Quantization-Aware LLM training, Reinforcement Learning, and Time Series Forecasting.

<div align="center">
  <img src="https://github.com/user-attachments/assets/365b571d-1004-4fff-8878-9af1374da057" alt="Image 2" style="width: 900px; margin: 0 auto;">
</div>

## Abstract

Large Language Models (LLMs) have achieved remarkable success, yet recent findings reveal that their deeper layers often contribute minimally and can be pruned without affecting overall performance. While some view this as an opportunity for model compression, we identify it as a training shortfall rooted in the widespread use of Pre-Layer Normalization (Pre-LN). We demonstrate that Pre-LN, commonly employed in models like GPT and LLaMA, leads to diminished gradient norms in its deeper layers, reducing their effectiveness. In contrast, Post-Layer Normalization (Post-LN) preserves larger gradient norms in deeper layers but suffers from vanishing gradients in earlier layers. To address this, we introduce Mix-LN, a novel normalization technique that combines the strengths of Pre-LN and Post-LN within the same model. Mix-LN applies Post-LN to the earlier layers and Pre-LN to the deeper layers, ensuring more uniform gradient norms across layers. This allows all parts of the network—both shallow and deep layers—to contribute effectively to training. Extensive experiments with various model sizes demonstrate that Mix-LN consistently outperforms both Pre-LN and Post-LN, promoting more balanced, healthier gradient norms throughout the network, and enhancing the overall quality of LLM pre-training. Furthermore, we demonstrate that models pre-trained with Mix-LN learn better compared to those using Pre-LN or Post-LN during supervised fine-tuning, highlighting the critical importance of high-quality deep layers. By effectively addressing the inefficiencies of deep layers in current LLMs, Mix-LN unlocks their potential, enhancing model capacity without increasing model size.

### TODO

- [x] Release LLM training codes.
- [ ] Release QAT (A4W4 and A8W8) LLM training codes.
- [ ] Release Reinforcement Learning training codes.
- [ ] Release Time Series Forescasting training codes.

## Quick Start

### Setup
Our repository is built on top of [GaLore](https://github.com/jiaweizzhao/GaLore). You can configure the environment using the following command lines:
conda create -n spam python=3.11 -y
conda activate spam
pip3 install torch torchvision torchaudio<br>
pip install transformers==4.31.0<br>
pip install tqdm wandb<br>
## Usage

```python
from galore_torch import SPAM
# define param groups as spam_params and non_spam_params
param_groups = [{'params': non_spam_params}, 
                {'params': spam_params, 'density': 1.0, 'update_proj_gap': 500}]
optimizer = SPAM(param_groups, lr=0.001)
```

### example: Training LLaMA-130M 

torchrun --standalone --nproc_per_node 2 torchrun_main.py \
    --model_config configs/llama_130m.json \
    --lr 8e-4 \
    --density 1.0 \
    --update_gap 500 \
    --batch_size 128  \
    --total_batch_size 512 \
    --num_training_steps 20000 \
    --warmup_steps 1000 \
    --weight_decay 0 \
    --dtype bfloat16 \
    --eval_every 1000 \
    --threshold 5000 \
    --save_dir $save_dir \
    --optimizer SPAM \
    --warmup_epoch 150 


### example: Training LLaMA-350M 

torchrun --standalone --nproc_per_node 2 torchrun_main.py \
    --model_config configs/llama_350m.json \
    --lr 4e-4 \
    --density 1.0 \
    --update_gap 500 \
    --batch_size 128  \
    --total_batch_size 512 \
    --num_training_steps 60000 \
    --warmup_steps 1000 \
    --weight_decay 0 \
    --dtype bfloat16 \
    --eval_every 1000 \
    --threshold 5000 \
    --save_dir $save_dir \
    --optimizer SPAM \
    --warmup_epoch 150 

## Acknowledgement
This repository is build upon the  [GaLore](https://github.com/jiaweizzhao/GaLore) repository. Thanks for the great work!