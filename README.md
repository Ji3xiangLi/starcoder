# 💫 StarCoder

[Paper](https://drive.google.com/file/d/1cN-b9GnWtHzQRoE7M7gAEyivY0kl4BYs/view) | [Model](https://huggingface.co/bigcode/starcoder) | [Playground](https://huggingface.co/spaces/bigcode/bigcode-playground) | [VSCode](https://marketplace.visualstudio.com/items?itemName=HuggingFace.huggingface-vscode) | [Chat](https://huggingface.co/spaces/HuggingFaceH4/starchat-playground)

# What is this about?
💫 StarCoder is a language model (LM) trained on source code and natural language text. Its training data incorporates more that 80 different programming languages as well as text extracted from GitHub issues and commits and from notebooks. This repository showcases how we get an overview of this LM's capabilities.

# News

* **May 9, 2023:** We've fine-tuned StarCoder to act as a helpful coding assistant 💬! Check out the `chat/` directory for the training code and play with the model [here](https://huggingface.co/spaces/HuggingFaceH4/starchat-playground).

# Disclaimer

Before you can use the model go to `hf.co/bigcode/starcoder` and accept the agreement. And make sure you are logged into the Hugging Face hub with:
```bash
huggingface-cli login
```

# Table of Contents
1. [Quickstart](#quickstart)
    - [Installation](#installation)
    - [Code generation with StarCoder](#code-generation)
    - [Text-generation-inference code](#text-generation-inference)
2. [Fine-tuning](#fine-tuning)
    - [Step by step installation with conda](#step-by-step-installation-with-conda)
    - [Datasets](#datasets)
      - [Stack Exchange](#stack-exchange-se)
    - [Merging PEFT adapter layers](#merging-peft-adapter-layers)
3. [Evaluation](#evaluation)
4. [Inference hardware requirements](#inference-hardware-requirements)

# Quickstart
StarCoder was trained on GitHub code, thus it can be used to perform code generation. More precisely, the model can complete the implementation of a function or infer the following characters in a line of code. This can be done with the help of the 🤗's [transformers](https://github.com/huggingface/transformers) library.

## Installation
First, we have to install all the libraries listed in `requirements.txt`
```bash
pip install -r requirements.txt
```
## Code generation
The code generation pipeline is as follows

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

checkpoint = "bigcode/starcoder"
device = "cuda" # for GPU usage or "cpu" for CPU usage

tokenizer = AutoTokenizer.from_pretrained(checkpoint)
# to save memory consider using fp16 or bf16 by specifying torch_dtype=torch.float16 for example
model = AutoModelForCausalLM.from_pretrained(checkpoint).to(device)

inputs = tokenizer.encode("def print_hello_world():", return_tensors="pt").to(device)
outputs = model.generate(inputs)
# clean_up_tokenization_spaces=False prevents a tokenizer edge case which can result in spaces being removed around punctuation
print(tokenizer.decode(outputs[0], clean_up_tokenization_spaces=False))
```
or
```python
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
checkpoint = "bigcode/starcoder"

model = AutoModelForCausalLM.from_pretrained(checkpoint)
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device=0)
print( pipe("def hello():") )
```
For hardware requirements, check the section [Inference hardware requirements](#inference-hardware-requirements).

## Text-generation-inference

```bash
docker run -p 8080:80 -v $PWD/data:/data -e HUGGING_FACE_HUB_TOKEN=<YOUR BIGCODE ENABLED TOKEN> -d  ghcr.io/huggingface/text-generation-inference:latest --model-id bigcode/starcoder --max-total-tokens 8192
```
For more details, see [here](https://github.com/huggingface/text-generation-inference).

# Fine-tuning

Here, we showcase how we can fine-tune this LM on a specific downstream task.

介绍如何对Starcoder进行微调。

## Step by step installation with conda 

Create a new conda environment and activate it

创建Conda虚拟环境。
```bash
conda create -n env
conda activate env
```
Install the `pytorch` version compatible with your version of cuda [here](https://pytorch.org/get-started/previous-versions/), for example the following command works with cuda 11.6
```bash
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.6 -c pytorch -c nvidia
```
Install `transformers` and `peft`
```bash
conda install -c huggingface transformers 
pip install git+https://github.com/huggingface/peft.git
```
Note that you can install the latest stable version of transformers by using

```bash
pip install git+https://github.com/huggingface/transformers
```

Install `datasets`, `accelerate` and `huggingface_hub`

```bash
conda install -c huggingface -c conda-forge datasets
conda install -c conda-forge accelerate
conda install -c conda-forge huggingface_hub
```

Finally, install `bitsandbytes` and `wandb`
```bash
pip install bitsandbytes
pip install wandb
```
To get the full list of arguments with descriptions you can run the following command on any script:
```
python scripts/some_script.py --help
```
Before you run any of the scripts make sure you are logged in and can push to the hub:
```bash
huggingface-cli login
```
Make sure you are logged in `wandb`:

！需要注意 wandb 需要登陆
```bash
wandb login
```
Now that everything is done, you can clone the repository and get into the corresponding directory.

## Datasets
💫 StarCoder can be fine-tuned to achieve multiple downstream tasks. Our interest here is to fine-tune StarCoder in order to make it follow instructions. [Instruction fine-tuning](https://arxiv.org/pdf/2109.01652.pdf) has gained a lot of attention recently as it proposes a simple framework that teaches language models to align their outputs with human needs. That procedure requires the availability of quality instruction datasets, which contain multiple `instruction - answer` pairs. Unfortunately such datasets are not ubiquitous but thanks to Hugging Face 🤗's [datasets](https://github.com/huggingface/datasets) library we can have access to some good proxies. To fine-tune cheaply and efficiently, we use Hugging Face 🤗's [PEFT](https://github.com/huggingface/peft) as well as Tim Dettmers' [bitsandbytes](https://github.com/TimDettmers/bitsandbytes).

给了一个对starcoder 进行指令微调的例子，指令微调可以让预训练模型输出和人类需求对齐。需要的训练数据是<指令，回答>这种形式的。使用的数据集来自StackExchange SE 一个知名的问答网站（知乎？），用来提高模型回答问题的能力。
### Stack Exchange SE
[Stack Exchange](https://en.wikipedia.org/wiki/Stack_Exchange) is a well-known network of Q&A websites on topics in diverse fields. It is a place where a user can ask a question and obtain answers from other users. Those answers are scored and ranked based on their quality. [Stack exchange instruction](https://huggingface.co/datasets/ArmelR/stack-exchange-instruction) is a dataset that was obtained by scrapping the site in order to build a collection of Q&A pairs. A language model can then be fine-tuned on that dataset to make it elicit strong and diverse question-answering skills.

To execute the fine-tuning script run the following command:

模型的微调运行方法
```bash
python finetune/finetune.py \
  --model_path="bigcode/starcoder"\
  --dataset_name="ArmelR/stack-exchange-instruction"\
  --subset="data/finetune"\
  --split="train"\
  --size_valid_set 10000\
  --streaming\
  --seq_length 2048\
  --max_steps 1000\
  --batch_size 1\
  --input_column_name="question"\
  --output_column_name="response"\ 
  --gradient_accumulation_steps 16\
  --learning_rate 1e-4\
  --lr_scheduler_type="cosine"\
  --num_warmup_steps 100\
  --weight_decay 0.05\
  --output_dir="./checkpoints" \
```
The size of the SE dataset is better manageable when using streaming. We also have to precise the split of the dataset that is used. For more details, check the [dataset's page](https://huggingface.co/datasets/ArmelR/stack-exchange-instruction) on 🤗. Similarly we can modify the command to account for the availability of GPUs

参数中使用straming 可以更好的控制SE 数据集的大小，也可以修改以下参数来配置GPU的设备。
```bash
python -m torch.distributed.launch \
  --nproc_per_node number_of_gpus finetune/finetune.py \
  --model_path="bigcode/starcoder"\
  --dataset_name="ArmelR/stack-exchange-instruction"\
  --subset="data/finetune"\
  --split="train"\
  --size_valid_set 10000\
  --streaming \
  --seq_length 2048\
  --max_steps 1000\
  --batch_size 1\
  --input_column_name="question"\
  --output_column_name="response"\ 
  --gradient_accumulation_steps 16\
  --learning_rate 1e-4\
  --lr_scheduler_type="cosine"\
  --num_warmup_steps 100\
  --weight_decay 0.05\
  --output_dir="./checkpoints" \
```
## Merging PEFT adapter layers
If you train a model with PEFT, you'll need to merge the adapter layers with the base model if you want to run inference / evaluation. To do so, run:

如果使用peft来训练，那么训练中添加的模型结构和参数就需要添加到base模型上才能为后续训练和评估使用。需要运行`finetune/merge_peft_adapters.py`.
```bash
python finetune/merge_peft_adapters.py --base_model_name_or_path model_to_merge --peft_model_path model_checkpoint

# Push merged model to the Hub
python finetune/merge_peft_adapters.py --base_model_name_or_path model_to_merge --peft_model_path model_checkpoint --push_to_hub
```
For example

```bash
python finetune/merge_peft_adapters.py --model_name_or_path bigcode/starcoder --peft_model_path checkpoints/checkpoint-1000 --push_to_hub
```

# Evaluation
To evaluate StarCoder and its derivatives, you can use the [BigCode-Evaluation-Harness](https://github.com/bigcode-project/bigcode-evaluation-harness) for evaluating Code LLMs.

评估starcoder 和其变体，可以查看上面这个连接。
# Inference hardware requirements
In FP32 the model requires more than 60GB of RAM, you can load it in FP16 or BF16 in ~30GB, or in 8bit under 20GB of RAM with

FP32的模型，推理需要至少60GB的内存。FP16 大约30GB。8Bit 20GB。
```python
# make sure you have accelerate and bitsandbytes installed
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bigcode/starcoder")
# for fp16 replace with  `load_in_8bit=True` with   `torch_dtype=torch.float16`
model = AutoModelForCausalLM.from_pretrained("bigcode/starcoder", device_map="auto", load_in_8bit=True)
print(f"Memory footprint: {model.get_memory_footprint() / 1e6:.2f} MB")
````
```
Memory footprint: 15939.61 MB
```
You can also try [starcoder.cpp](https://github.com/bigcode-project/starcoder.cpp), a C++ implementation with [ggml](https://github.com/ggerganov/ggml) library.
