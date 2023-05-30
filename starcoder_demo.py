from transformers import AutoModelForCausalLM, AutoTokenizer
from accelerate import Accelerator
import argparse

def set_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_text", type=str, default="def print_hello_world():")
    return parser.parse_args()


if __name__ == '__main__':
    args = set_args()
    model = AutoModelForCausalLM.from_pretrained("bigcode/starcoder")
    tokenizer = AutoTokenizer.from_pretrained("bigcode/starcoder")

    accelerator = Accelerator(mixed_precision="fp16")
    accelerator.print(accelerator.device)
    model = accelerator.prepare(model)
    inputs = tokenizer.encode(args.input_text, return_tensors="pt")
    outputs = model.generate(inputs)
    print(tokenizer.decode(outputs[0]))