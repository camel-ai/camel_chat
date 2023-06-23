# Overview
This is a minimalistic repository to reproduce and serve CAMEL models based on lm-sys/FastChat [repository](https://github.com/lm-sys/FastChat). CAMEL models are conversational language models obtained by finetuning LLaMA foundation language models on data collected through the CAMEL framework. CAMEL is an opensource project doing research on collaborative AI agents and LLMs. Find out more about the CAMEL project by visiting our [website](https://www.camel-ai.org/).

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
Collected data through CAMEL framework are hosted on Huggingface website. They are available under the account [camel-ai](https://huggingface.co/camel-ai). Our combined-data models also make use of ShareGPT data available [here](https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/tree/main/HTML_cleaned_raw_dataset) and Alpaca instruction dataset available [here](https://github.com/tatsu-lab/stanford_alpaca/blob/761dc5bfbdeeffa89b8bff5d038781a4055f796a/alpaca_data.json). We follow lm-sys/FastChat in cleaning and pre-processing ShareGPT data by converting HTML to markdown. We choose to keep only English language for ShareGPT dataset.

You can select which datasets you want to download and preprocess. The option are as follows: [ai_society, code, math, physics, chemistry, biology, alpaca, sharegpt, all]. You can pass several datasets as command line arguments as follows:

To download all datasets, please run the follwoing command:
```
python -m camel_chat.data_preprocessing.download_datasets --download_directory datasets --datasets all
```
To preprocess the data into conversation format, please run the following:
```
python -m camel_chat.data_preprocessing.convert_datasets_to_conversation_format --download_directory datasets --datasets all
```
You can also select subset of datasets and pass them as arguments as follows
```
# Downloads ai_society and biology only
python -m camel_chat.data_preprocessing.download_datasets --download_directory datasets --datasets ai_society biology
# Processes ai_society and biology only
python -m camel_chat.data_preprocessing.convert_datasets_to_conversation_format --download_directory datasets --datasets ai_society biology
```
### Merge datasets and split long conversations to 2048 tokens and save them into one dataset.json file 
```
python merge.py --in-directory datasets --out-file dataset.json --tokenizer-path /path/to/tokenizer --max-length 2048
```
# Training
To train conversational model on CAMEL data or your custom data, we provided ready bash scripts for you to launch. We provide scripts for regular training with PyTorch FSDP which requires server grade GPUs. We also include a QLoRA alternative for finetuning the 7B and 13B models on consumer grade GPUs. We use flash attention in all our models for faster training and inference.

### Regular training with FSDP or DeepSpeed
 We provide bash scripts that use PyTorch FSDP to finetune the 7B (`train_camel_7B.sh`) and 13B (`train_camel_13B.sh`) models on a single node with 4 and 8 A100-80GB GPUs respectively. We also include an example of how to finetune 13B model on multiple nodes (`train_camel_13B_distributed.sh`). Finally, if you have the resources, we provide a DeepSpeed example of how to finetune the 30B model on 16 A100-80GB GPUs with parameter and optimizer CPU offloading.

 To launch a script on a computer cluster with Slurm support use `sbatch scripts/train_camel_7B.sh`. If you're on a node without Slurm, use the command `bash scripts/train_camel_7B.sh`.

### QLoRA training
If you do not have server grade GPUs, you may consider using QLoRA that makes it possible to finetune 13B model on consumer grade GPUs. We include an example script (`train_camel_qlora.sh`) of how to use it. Note that our 30B and 65B models were finetuned with QLoRA with a global batch size of 256 on 4 A100-80GB GPUs. Huggingface does not support resuming from a checkpoint for QLoRA. We provide a way to fix that by adding an additional flag `load_weights=False` to `trainer.train()`. This will resume the optimizer, scheduler, and dataloader states but not the model weights since the base weight and adapter weights are loaded before training loop.

If you have trained a model with QLoRA, you will have to merge the adapter with base model. Run the following:
```
python -m camel_chat.model.apply_lora --base-model-path /path/to/llama/base/model --target-model-path /path/to/save/folder --lora-path /path/to/adapter
```
# Serve
We use the same serving as lm-sys/FastChat. You can interact with the finetuned models by terminal or Web GUI.

## Prompt format
We use the same prompt format as Vicuna_v1.1. It assumes a conversation between a user and an assistant. The user roles are separated from content by a colon. Assistant and user content are separated by space. Assistant role is terminated by <\/s>.

```
USER: xxxxxxxxx ASSISTANT: xxxxxxxxx </s>
``` 
For more details, check [here](https://github.com/camel-ai/camel_chat/blob/9c889e9b964eb36963dbe9cec8a034dafc844179/camel_chat/conversation.py#L231). You do not have to worry about this if you are using the serving included in this repository.

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
If you do not have a GPU, you can use CPU only inference, but this requires 60GB of CPU RAM for 13B models.
```
python3 -m camel_chat.serve.cli --model-path /path/to/model --device cpu
```
## Serve in Web GUI
Launch the controller
```
python -m camel_chat.serve.controller
```
Launch the model worker(s)
```
python -m camel_chat.serve.model_worker --model-path /path/to/model
```
Wait until the process finishes loading the model and you see "Uvicorn running on ...". The model worker will register itself to the controller.

To ensure that your model worker is connected to your controller properly, send a test message using the following command:
```
python -m camel_chat.serve.test_message --model-name /path/to/model
```
You will see a short output.

Launch the Gradio web server
```
python -m camel_chat.serve.gradio_web_server
```
This is the user interface that users will interact with.

# Acknowledgements
We heavily borrow from the open source projects [lm-sys/FastChat](https://github.com/lm-sys/FastChat) and [artidoro/qlora](https://github.com/artidoro/qlora/tree/main). We thank them for sharing their work and contributing to the open source community.