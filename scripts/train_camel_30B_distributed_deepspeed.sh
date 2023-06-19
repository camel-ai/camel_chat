#!/bin/bash -l
#SBATCH --job-name=deepspeed-camel-distributed
#SBATCH --output=slurm/%x.%3a.%A.out
#SBATCH --error=slurm/%x.%3a.%A.err
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:4
#SBATCH --nodes=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-gpu=6
#SBATCH --mem=384G
#SBATCH --reservation=A100

NUM_NODES=4
GPU_PER_NODE=4
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
PORT=$(python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')

# With DeepSpeed
srun torchrun --nnodes=$NUM_NODES --nproc_per_node=$GPU_PER_NODE \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$PORT  \
    fastchat/train/train_mem.py \
    --model_name_or_path ./camel30B-deepspeed-camel229K_sharegpt107K_alpaca52K_len2048/checkpoint-800  \
    --data_path ./datasets/camel229K_sharegpt107K_alpaca52K_len2048.json \
    --bf16 True \
    --output_dir ./camel30B-deepspeed-camel229K_sharegpt107K_alpaca52K_len2048 \
    --num_train_epochs 3 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 32 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "steps" \
    --eval_steps 200 \
    --save_strategy "steps" \
    --save_steps 200 \
    --save_total_limit 6 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.04 \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --lazy_preprocess True \
    --resume_from_checkpoint True \
    --deepspeed ./ds_config.json