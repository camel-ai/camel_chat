# Overview
This is a minimalistic repository to reproduce and serve CAMEL models. CAMEL models are conversational language models obtained by finetuning LLaMA foundation language models on data collected through the CAMEL framework.

# Installation
Setup camel_chat conda environment:
```
# Create a conda virtual environment
conda create --name camel_chat python=3.10

# Activate camel conda environment
conda activate camel_chat

# Install PyTorch with CUDA=11.7 support
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia

# Install dependencies
pip install -r requirements.txt

# Install flash attention. Assumes CUDA and gcc available for compilation
pip install 

# Clone camel_chat repository
https://github.com/camel-ai/camel_chat.git

# Change directory to camel_chat
cd camel_chat

# Install camel_chat
pip install -e .

# Install pre-commmit within camel_chat env (only needed for opening pull requests)
pip install pre-commit
pre-commit install
```
# Training
To train conversational model on CAMEL data or your custom data, we provided ready bash scripts for you to launch. We provide scripts for regular training which requires server grade GPUs. We also include a QLoRA alternative for finetuning the 7B and 13B models on consumer grade GPUs.

We use flash attention in all our models for faster training. We provide bash scripts that use PyTorch FSDP to finetune the 7B (`train_camel_7B.sh`) and 13B (`train_camel_13B.sh`) models on a single node with 4 and 8 A100-80GB GPUs respectively. We also include an example of how to finetune 13B model on multiple nodes (`train_camel_13B_distributed.sh`). Finally, if you have the resources, we provide a DeepSpeed example of how to finetune the 30B model on 16 A100-80GB GPUs with parameter and optimizer offloading. 

If you do not have server grade GPUs, you may consider using QLoRA that makes it possible to finetune 13B model on consumer grade GPUs. We include an example script (`train_camel_qlora.sh`) of how to use it. Note that our 30B and 65B models were finetuned with QLoRA with a global batch size of 256 on 4 A100-80GB GPUs. Huggingface does not support resuming from a checkpoint for QLoRA. We provide a hacky way to fix that by adding an additional flag `load_weights=False` to `trainer.train()`. Replace the contents of Huggingface's `trainer.py` by `hacked_trianer.py` file. This will resume the optimizer, scheduler, and dataloader states but not the model weights.

To launch a script on a computer cluster with Slurm support use `sbatch scripts/train_camel_7B.sh`. If you're on a node without Slurm, use the command `bash scripts/train_camel_7B.sh`. 

# Serve
You can interact with the finetuned models by terminal or Web GUI.



# Acknowledgements
We heavily borrow from the open source project lm-sys/FastChat and artidoro/qlora. We thank them for sharing their work and contributing to the open source community.