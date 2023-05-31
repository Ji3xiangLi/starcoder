import torch
from chat.dialogues import DialogueTemplate, get_dialogue_template
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          GenerationConfig, set_seed)
from accelerate import init_empty_weights, load_checkpoint_and_dispatch
from huggingface_hub import snapshot_download

from flask import Flask, request
from flask_cors import *
model_id = "bigcode/starcoder"
app = Flask(__name__)

@app.route("/inference", methods=["GET"])
@cross_origin()
def get_infrence_response():
    input_text = request.args.get("input", "")
    res = infer(input_text)
    return {"success": 1, "response": res}


@app.route("/chat", methods=["GET"])
@cross_origin()
def get_chat_response():
    input_text = request.args.get("input", "")
    res = chat(input_text)
    return {"success":1, "response": res}

def log(predict_type, input_text, output_text):
    with open("logs.txt", "a") as f:
        f.write("=" * 10 + predict_type + "=" * 10 + "\n")
        f.write("\n".join(["[input text]: ", input_text, "\n", "[output text]: ", output_text, ""]))
        f.write("=" * 25 + "\n")


class Inference:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def forward(self, input_text: str):
        inputs = self.tokenizer(input_text, return_tensors="pt")
        inputs = inputs.to(0)
        generation_config = GenerationConfig(
            temperature=0.2,
            top_k=50,
            top_p=0.95,
            repetition_penalty=1.2,
            do_sample=True,
            pad_token_id= self.tokenizer.eos_token_id,
            eos_token_id= self.tokenizer.convert_tokens_to_ids("<|end|>"),
            min_new_tokens=32,
            max_new_tokens=256,
        )
        outputs = self.model.generate(inputs["input_ids"], generation_config=generation_config)
        output_text = self.tokenizer.decode(outputs[0].tolist(), skip_special_tokens=False).lstrip()

        log("inference", input_text, output_text)
        return output_text

    def __call__(self, input_text):
        return self.forward(input_text)



class Chat:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        try:
            self.dialogue_template = DialogueTemplate.from_pretrained(model_id, revision=None)
        except Exception:
            print("No dialogue template found in model repo. Defaulting to the `no_system` template.")
            self.dialogue_template = get_dialogue_template("no_system")

    def forward(self, input_text):
        prompt = [{
            "role": "user",
            "content": input_text
        }]
        self.dialogue_template.messages = prompt
        formatted_prompt = self.dialogue_template.get_inference_prompt()

        print("input_text: ")
        print(formatted_prompt)

        generation_config = GenerationConfig(
            temperature=0.2,
            top_k=50,
            top_p=0.95,
            repetition_penalty=1.2,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.convert_tokens_to_ids(self.dialogue_template.end_token),
            min_new_tokens=32,
            max_new_tokens=256,
        )
        batch = self.tokenizer(formatted_prompt,  return_tensors="pt", return_token_type_ids=False).to(0)
        generated_ids = model.generate(**batch, generation_config=generation_config)
        generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=False).lstrip()

        log("chat", input_text, generated_text)



        print(f"=== complete conversation===")
        print()
        print(generated_text)
        print()
        print("======================")
        print()
        return generated_text

    def __call__(self, input_text):
        return self.forward(input_text)


if __name__ == '__main__':
    set_seed(42)
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=None)

    with init_empty_weights():
        raw_model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16)

    raw_model.tie_weights()
    model = load_checkpoint_and_dispatch(
            raw_model,
            checkpoint=snapshot_download(model_id),
            device_map="auto",
            no_split_module_classes=["GPTBigCodeBlock"],  # 根据模型结构修改，不能拆分到多个设备的modulename，含有残差的module都应该写上去。
            dtype=torch.float16)

    chat = Chat(model, tokenizer)
    infer = Inference(model, tokenizer)
    app.run(host="0.0.0.0", port=12342)
