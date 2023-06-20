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

# Install dependencies, flash-attn requires CUDA>11.0 and gcc compilers
pip install -r requirements.txt

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
# Data Pre-processing
Download our data from Huggingface website. They are available under the account [camel-ai](https://huggingface.co/camel-ai). Our combined-data models also make use of ShareGPT data available [here](https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/tree/main/HTML_cleaned_raw_dataset) and Alpaca instruction dataset available [here](https://github.com/tatsu-lab/stanford_alpaca/blob/761dc5bfbdeeffa89b8bff5d038781a4055f796a/alpaca_data.json).

### Download datasets collected with CAMEL framework and convert them to conversation format suitable for training
```
cd data
# Download CAMEL datasets into a folder called datasets
python download_hf_camel_datasets.py
# Convert them to conversation format for training
python convert_camel_datasets_to_conversation_format.py
```

### ShareGPT Cleaning
We follow lm-sys/FastChat in cleaning and pre-processing ShareGPT data by following the steps below:
```
# Convert html to markdown
python clean_sharegpt --in sharegpt_html.json --out sharegpt_markdown.json

# Keep English language conversations
python optional_clean --in sharegpt_markdown.json --out sharegpt_markdown_en.json --keep-lang en
```
### Convert Alpaca instruction dataset into conversation format
```
python convert_alpaca_dataset_to_conversation.py
```
### Merge datasets into one dataset.json file
```
python merge.py --in-file datasets/alpaca_data_conv.json datasets/camel_datasets_conv.json datasets/sharegpt_markdown_en.json --out-file datasets/dataset.json
```
### Split long conversations to 2048 context length
```
python split_long_conversation.py --in dataset/dataset.json --out dataset/dataset_2048.json --model-name /path/to/base/model/tokenizer

```
# Training
To train conversational model on CAMEL data or your custom data, we provided ready bash scripts for you to launch. We provide scripts for regular training which requires server grade GPUs. We also include a QLoRA alternative for finetuning the 7B and 13B models on consumer grade GPUs.

We use flash attention in all our models for faster training. We provide bash scripts that use PyTorch FSDP to finetune the 7B (`train_camel_7B.sh`) and 13B (`train_camel_13B.sh`) models on a single node with 4 and 8 A100-80GB GPUs respectively. We also include an example of how to finetune 13B model on multiple nodes (`train_camel_13B_distributed.sh`). Finally, if you have the resources, we provide a DeepSpeed example of how to finetune the 30B model on 16 A100-80GB GPUs with parameter and optimizer offloading. 

If you do not have server grade GPUs, you may consider using QLoRA that makes it possible to finetune 13B model on consumer grade GPUs. We include an example script (`train_camel_qlora.sh`) of how to use it. Note that our 30B and 65B models were finetuned with QLoRA with a global batch size of 256 on 4 A100-80GB GPUs. Huggingface does not support resuming from a checkpoint for QLoRA. We provide a hacky way to fix that by adding an additional flag `load_weights=False` to `trainer.train()`. Replace the contents of Huggingface's `trainer.py` by `train/hacked_trianer.py` file. This will resume the optimizer, scheduler, and dataloader states but not the model weights.

To launch a script on a computer cluster with Slurm support use `sbatch scripts/train_camel_7B.sh`. If you're on a node without Slurm, use the command `bash scripts/train_camel_7B.sh`. 

If you have trained a model with QLoRA, you will have to merge the adapter with base model. Run the following:
```
python camel_chat.model.apply_lora --base-model-path /path/to/llama/base/model --target-model-path /path/to/save/folder --lora-path /path/to/adapter
```
# Serve
We use the same serving as lm-sys/FastChat. You can interact with the finetuned models by terminal or Web GUI.

## Serve in Terminal

To interact with the finetuned model in terminal, use the following command:
```
python -m camel_chat.serve.cli --model-path /path/to/model
``` 
If one GPU is not enough to fit the model, you can load it on 2 GPUs
```
python -m camel_chat.serve.cli --model-path /path/to/model --num-gpus 2
```
If you do not have enough VRAM and you can load the model in 8-bit
```
python -m camel_chat.serve.cli --model-path /path/to/model --load-8bit
```
## Serve in Web GUI
Launch the controller
```
python -m camel_chat.serve.controller
```
Launch the model worker(s)
```
python -m camel_chat.serve.model_worker --model-path lmsys/vicuna-7b-v1.3
```
Wait until the process finishes loading the model and you see "Uvicorn running on ...". The model worker will register itself to the controller.

To ensure that your model worker is connected to your controller properly, send a test message using the following command:
```
python -m camel_chat.serve.test_message --model-name vicuna-7b-v1.3
```
You will see a short output.

Launch the Gradio web server
```
python -m camel_chat.serve.gradio_web_server
```
This is the user interface that users will interact with.

# Acknowledgements
We heavily borrow from the open source projects lm-sys/FastChat and artidoro/qlora. We thank them for sharing their work and contributing to the open source community.