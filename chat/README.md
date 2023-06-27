# Fine-tuning StarCoder for chat-based applications

This is a fully-working example to fine-tune `StarCoder` on a corpus of multi-turn dialogues and thus create a coding assistant that is chatty and helpful. The resulting model is quite good at generating code for plots and other programming tasks. For example, given the prompt:

这是一个用多轮对话微调StarCoder的例子，用这种方式让那个代码助手变得善于聊天、辅助写代码。这个结果模型挺适合生成**画图代码**和其他代码任务的。
```
Draw me a map of the world using geopandas. Make it so that only Germany and Spain are colored red.
```

the model will produce:

```python
import pandas as pd
from matplotlib import pyplot as plt
import geopandas as gpd
from shapely.geometry import Point, Polygon
%matplotlib inline
# Load data from URL or local file
url = 'https://raw.githubusercontent.com/datasets/geo-countries/master/data/countries.geojson'
df = gpd.read_file(url)
 
# Create plot
fig, ax = plt.subplots()
ax.set_aspect('equal')
world = gpd.GeoDataFrame({'name': ['World'],
                          'geometry': [Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])]})
germany = df[df['ADMIN'] == "Germany"]
spain = df[df['ADMIN'] == "Spain"]
other = df[(df['ADMIN']!= "Germany") & (df['ADMIN']!= "Spain")]
world.plot(color='lightgrey', edgecolor='white', ax=ax)
germany.plot(color="red", ax=ax)
spain.plot(color="red", ax=ax)
other.plot(color="skyblue", ax=ax)
plt.title("European Countries")
plt.show()
```

Check out our [blog post](https://huggingface.co/blog/starchat-alpha) for more details.

## Getting started

To run the `train.py` script, first create a Python virtual environment using e.g. Conda:

```shell
conda create -n chat python=3.10 && conda activate chat
```

Next, install PyTorch v1.13.1. Since this is hardware-dependent, we direct you to the [PyTorch Installation Page](https://pytorch.org/get-started/previous-versions/#v1131) for this step. Next, install the rest of the project dependencies:

```shell
pip install -r requirements.txt
```

You'll also need to be logged into both your Hugging Face account. To do so, run:

```shell
huggingface-cli login
```

Finally, install Git LFS with:

```shell
sudo apt-get install git-lfs
```

## Prepare your dataset

For training and inference, we use _dialogue templates_ to format each message in a conversation. For example, a typical dialogue between a human user and AI assistant takes the form:

使用的多轮对话数据格式是这个样子的，请确保你的数据集也要转换成这个样子。注意要像下面这个例子一样包含"messages"列。可以在``config.yaml`中修改文件。
```json
{
    "messages": [
        {
            "content": "Is it possible to imagine a society without law?", 
            "role": "user"},
        {
            "content": "It is difficult to imagine a society that is able to be maintained without any semblance of Law.",
            "role": "assistant",
        },
        {
            "content": "It seems like you consider the absence of law equal to the absence of anything that could guide the behaviour of the individual.",
            "role": "user",
        },
        {
            "content": "You are correct that there are other factors that can guide behavior in a society and play a role in shaping individuals' behavior and interactions with each other. However, even in societies where these factors are present, laws still serve an important role in maintaining social order and resolving conflicts.",
            "role": "assistant",
        }
    ]
}
```

Make sure you convert your dataset according to this schema, in particular you need to include a `messages` column like the above. You can adjust the model, dataset, and hyperparamters in the `config.yaml` file.

## Launch training

We use DeepSpeed ZeRO-3 to shard the model and optimizer across 8 x A100 (80GB) GPUs. To fine-tune run:

使用的DeepSpeed ZeRo-3来切分模型，然后使用了8块A100来训练。启动微调使用以下：

```
TRANSFORMERS_VERBOSITY=info torchrun --nproc_per_node=8 train.py config.yaml --deepspeed=deepspeed_z3_config_bf16.json
```

By default, this will save the model checkpoint in the `data/` directory and also push it to the Hugging Face Hub.


## Generate samples

To generate a few coding examples from your model, run:

使用微调后的代码生成，请使用以下代码。

```shell
python generate.py --model_id path/to/your/model
```

