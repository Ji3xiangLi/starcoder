from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch
"""
如果微调的方式是Adapters，那么就是用以下代码保存新添加的结构和参数，以便于后续的推理和再训练。
可以根据需要选择是否push到hub上。
"""
import os
import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model_name_or_path", type=str, default="bigcode/large-model")  # 原始模型位置
    parser.add_argument("--peft_model_path", type=str, default="/")  # 新得到的adapter结构和参数文件地址。
    parser.add_argument("--push_to_hub", action="store_true", default=True)

    return parser.parse_args()

def main():
    args = get_args()

    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model_name_or_path,
        return_dict=True,
        torch_dtype=torch.float16 
    )

    model = PeftModel.from_pretrained(base_model, args.peft_model_path)  # 一个是原始模型地址，还有一个是训练后新增的微调模型地址
    model = model.merge_and_unload()  # 将adapters插到原模型上

    tokenizer = AutoTokenizer.from_pretrained(args.base_model_name_or_path)

    if args.push_to_hub:
        # 上传到hub
        print(f"Saving to hub ...")
        model.push_to_hub(f"{args.base_model_name_or_path}-merged", use_temp_dir=False, private=True)
        tokenizer.push_to_hub(f"{args.base_model_name_or_path}-merged", use_temp_dir=False, private=True)
    else:
        # 保存到本地
        model.save_pretrained(f"{args.base_model_name_or_path}-merged")
        tokenizer.save_pretrained(f"{args.base_model_name_or_path}-merged")
        print(f"Model saved to {args.base_model_name_or_path}-merged")


if __name__ == "__main__" :
    main()
