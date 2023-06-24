#!/bin/bash -l
#SBATCH --job-name=camel-qlora
#SBATCH --output=slurm/%x.%3a.%A.out
#SBATCH --error=slurm/%x.%3a.%A.err
#SBATCH --time=24:00:00
#SBATCH --mem=256G
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-gpu 6
#SBATCH --reservation=A100

# All QLoRA models were trained with a total batch size of 256
# per_device_train_batch_size x NUM_GPUS x gradient_accumulation_steps = 256
# Tune the 3 parameters according to your hardware. Resuming from PEFT checkpoint
# is not supported by huggingface yet, but you can hack the trainer.train()
# in transformers. Replace 

# 7B model with flash attention and a batch size of 1 takes 9.5GB of VRAM on 1 GPU
# 13B model with flash attention and a batch size of 1 takes 16GB of VRAM on 1 GPU
# 33B model with flash attention and a batch size of 1 takes 36.2GB of VRAM on 1 GPU
# 65B model with flash attention and a batch size of 1 takes 65GB of VRAM on 1 GPU

NUM_GPUS=1
PORT=$(python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')

torchrun --nproc_per_node=$NUM_GPUS --master_port=$PORT \
    camel_chat/train/train_qlora.py \
    --model_name_or_path <path_to_hf_llama_model> \
    --data_path <path_to_json_file>\
    --bf16 True \
    --output_dir ./output \
    --num_train_epochs 3 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 16 \
    --gradient_accumulation_steps 16 \
    --evaluation_strategy steps \
    --eval_steps 250 \
    --max_eval_samples 1000 \
    --save_strategy steps \
    --save_steps 250 \
    --save_total_limit 100 \
    --learning_rate 0.0001 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type constant \
    --logging_steps 1 \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --lazy_preprocess True \
    --double_quant \
    --quant_type nf4 \
    --bits 4 \
    --lora_r 64 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --max_memory_MB 80000 \
    --ddp_find_unused_parameters False \
    --max_grad_norm 0.3 \
    # --resume_from_checkpoint True \
    # --checkpoint_path ./camel65B-qlora-camel229K_sharegpt107K_alpaca52K_len2048/checkpoint-550 \
    # --load_weights False