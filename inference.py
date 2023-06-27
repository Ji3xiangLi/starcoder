import torch
from accelerate import init_empty_weights, load_checkpoint_and_dispatch
from huggingface_hub import snapshot_download
from chat.dialogues import DialogueTemplate, get_dialogue_template
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          GenerationConfig, set_seed)


class BaseInference:
    """
    starcoder 的基础类。
    """
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def forward(self, input_text) -> str:
        """
        模型推理
        """
        pass
        return ""

    def log(self, label: str, input_text: str, output_text: str):
        """
        简单记录模型的输入和输出.
        label 为人工指定的标记
        """
        with open("logs.txt", "a") as f:
            f.write("=" * 10 + label + "=" * 10 + "\n")
            f.write("\n".join(["[input text]: ", input_text, "\n", "[output text]: ", output_text, ""]))
            f.write("=" * 25 + "\n")

    def __call__(self, input_text):
        """
        输入文本，进行推理并且将输入输出情况记录到logs.txt中。
        """
        output_text = self.forward(input_text)
        self.log("Base", input_text, output_text)
        return output_text


class StarCoderInference(BaseInference):
    def __init__(self, model, tokenizer):
        super(StarCoderInference, self).__init__(model, tokenizer)
        self.generation_config = GenerationConfig(
            temperature=0.2,
            top_k=50,
            top_p=0.95,
            repetition_penalty=1.2,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.convert_tokens_to_ids("<|end|>"),
            min_new_tokens=32,
            max_new_tokens=256,
        )

    def forward(self, input_text) -> str:
        inputs = self.tokenizer(input_text, return_tensors="pt")
        inputs = inputs.to(0)
        outputs = self.model.generate(inputs["input_ids"], generation_config=self.generation_config)
        output_text = self.tokenizer.decode(outputs[0].tolist(), skip_special_tokens=False).lstrip()
        return output_text

    def __call__(self, input_text):
        output_text = self.forward(input_text)
        self.log("StarCoderInference", input_text, output_text)
        return output_text


class StarCoderChat(BaseInference):
    def __init__(self, model, tokenizer):
        super(StarCoderChat, self).__init__(model, tokenizer)

        try:
            self.dialogue_template = DialogueTemplate.from_pretrained("bigcode/starcoder", revision=None)
        except Exception:
            print("No dialogue template found in model repo. Defaulting to the `no_system` template.")
            self.dialogue_template = get_dialogue_template("no_system")

        self.generation_config = GenerationConfig(
            temperature=0.2,
            top_k=50,
            top_p=0.95,
            repetition_penalty=1.2,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.convert_tokens_to_ids(self.dialogue_template.end_token),
            min_new_tokens=32,
            max_new_tokens=256,
        )

    def forward(self, input_text):
        prompt = [{
            "role": "user",
            "content": input_text
        }]
        # todo dialogue_template这是什么？
        self.dialogue_template.messages = prompt
        formatted_prompt = self.dialogue_template.get_inference_prompt()

        print("input_text: ")
        print(formatted_prompt)

        # generation config的设置和inference出了eos_token_id 其他都一样
        generation_config = GenerationConfig(
            temperature=0.2,
            top_k=50,
            top_p=0.95,
            repetition_penalty=1.2,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.convert_tokens_to_ids(self.dialogue_template.end_token),
            min_new_tokens=32,
            max_new_tokens=256,
        )
        # batch 是什么？
        batch = self.tokenizer(formatted_prompt, return_tensors="pt", return_token_type_ids=False).to(0)

        generated_ids = self.model.generate(**batch, generation_config=generation_config)
        generated_text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=False).lstrip()

        print(f"=== complete conversation===")
        print()
        print(generated_text)
        print()
        print("======================")
        print()
        return generated_text

    def __call__(self, input_text):
        output_text = self.forward(input_text)
        self.log("StarCoderChat", input_text, output_text)
        return output_text


def load_dispatched_model_and_tokenizer(model_id):
    """
    加载tokenizer 和可以被拆分装载到多个GPU上的模型。
    model_id: huggingface 上模型的地址 或者 本地模型存放文件夹路径

    默认加载float16
    """

    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=None)

    # 赋予模型随机初始化的参数
    with init_empty_weights():
        raw_model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16)

    raw_model.tie_weights()
    model = load_checkpoint_and_dispatch(
        raw_model,
        checkpoint=snapshot_download(model_id),
        device_map="auto",
        no_split_module_classes=["GPTBigCodeBlock"],  # 根据模型结构修改，不能拆分到多个设备的modulename，含有残差的module都应该写上去。
        dtype=torch.float16)
    return model, tokenizer


if __name__ == '__main__':
    set_seed(42)
    starcoder_id = "bigcode/starcoder"
    starcoder_model, starcoder_tokenizer = load_dispatched_model_and_tokenizer(starcoder_id)

    # 实例化聊天模式
    chat = StarCoderChat(starcoder_model, starcoder_tokenizer)

    # 实例化推理模式，调用infer(input_text) 直接将input_text输入到模型中返回模型的全部生成值
    inference = StarCoderInference(starcoder_model, starcoder_tokenizer)

    print(chat("def hello_world():"))
