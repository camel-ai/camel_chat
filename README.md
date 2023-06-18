# Overview
This is a minimalistic repository to reproduce and serve CAMEL models. CAMEL models are conversational language models obtained by finetuning LLaMA foundation language models on data collected through the CAMEL framework.

# Installation
Setup camel_chat conda environment:
```
# Create a conda virtual environment
conda create --name camel_chat python=3.10

# Activate camel conda environment
conda activate camel_chat

# Install dependencies
pip install -r requirements.txt
```

# Acknowledgements
We heavily borrow from the open source project lm-sys/FastChat. We thank them for sharing their work and contributing to the open source community.