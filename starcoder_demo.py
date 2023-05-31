from transformers import AutoModelForCausalLM, AutoTokenizer
from accelerate import Accelerator, load_checkpoint_and_dispatch, init_empty_weights
import torch
import argparse

def set_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_text", type=str, default="def print_hello_world():")
    return parser.parse_args()

def main():
    args = set_args()
    print("Waiting for all devices to be ready, it may take a few minutes...")
    with init_empty_weights():
        raw_model = AutoModelForCausalLM.from_pretrained("bigcode/starcoder", torch_dtype=torch.float16)
    raw_model.tie_weights()
    model = load_checkpoint_and_dispatch(
        raw_model, "bigcode/starcoder", device_map="auto", dtype=torch.float16
    )
    # model = AutoModelForCausalLM.from_pretrained("bigcode/starcoder")
    tokenizer = AutoTokenizer.from_pretrained("bigcode/starcoder")

    accelerator = Accelerator(mixed_precision="fp16")
    accelerator.print("device: ", accelerator.device)
    model = accelerator.prepare(model)
    inputs = tokenizer.encode(args.input_text, return_tensors="pt")
    outputs = model.generate(inputs)
    print(tokenizer.decode(outputs[0]))

if __name__ == '__main__':
    main()