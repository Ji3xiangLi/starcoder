"""
StarCoderPlus, StarChat-beta 等starcoder 变体的官方代码给的使用代码。

StarCoder: 通过程序语言进行了预训练得到StarCoderBase，然后在该模型基础上使用python数据集微调得到。

StarCoderPlus: 在StarCoder基础上使用英文语料进行了微调。

StarChat-beta: 在StarCoderPlus基础上经过指令微调和人类对齐。
"""

def starcoder_generation():
    # pip install -q transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    checkpoint = "bigcode/starcoder"
    device = "cuda"  # for GPU usage or "cpu" for CPU usage

    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForCausalLM.from_pretrained(checkpoint).to(device)

    inputs = tokenizer.encode("def print_hello_world():", return_tensors="pt").to(device)
    outputs = model.generate(inputs)
    print(tokenizer.decode(outputs[0]))

    # 法二
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
    checkpoint = "bigcode/starcoder"

    model = AutoModelForCausalLM.from_pretrained(checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)

    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device=0)
    print(pipe("def hello():"))


def starcoderplus_generation():
    """
    使用starcoder plus 生成代码
    """
    # pip install -q transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    checkpoint = "bigcode/starcoderplus"
    device = "cuda"  # for GPU usage or "cpu" for CPU usage

    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForCausalLM.from_pretrained(checkpoint).to(device)

    inputs = tokenizer.encode("def print_hello_world():", return_tensors="pt").to(device)
    outputs = model.generate(inputs)
    print(tokenizer.decode(outputs[0]))


def starchat_generaton():
    import torch
    from transformers import pipeline

    pipe = pipeline("text-generation", model="HuggingFaceH4/starchat-beta", torch_dtype=torch.bfloat16,
                    device_map="auto")

    # We use a variant of ChatML to format each message
    prompt_template = "<|system|>\n<|end|>\n<|user|>\n{query}<|end|>\n<|assistant|>"
    prompt = prompt_template.format(query="How do I sort a list in Python?")
    # We use a special <|end|> token with ID 49155 to denote ends of a turn
    outputs = pipe(prompt, max_new_tokens=256, do_sample=True, temperature=0.2, top_k=50, top_p=0.95,
                   eos_token_id=49155)
    # You can sort a list in Python by using the sort() method. Here's an example:\n\n```\nnumbers = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5]\nnumbers.sort()\nprint(numbers)\n```\n\nThis will sort the list in place and print the sorted list.
