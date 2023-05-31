from typing import Union, List, Tuple, Optional
import torch
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from accelerate import init_empty_weights, load_checkpoint_and_dispatch


class Inference:
    def __init__(self):
        """
        Initializes the MossModel with a given model or loads a model from the specified directory.
        """
        self.checkpoint = "bigcode/starcoder"
        self.model = self.Init_Model_Parallelism()
        self.tokenizer = AutoTokenizer.from_pretrained(self.checkpoint)

    def Init_Model_Parallelism(self):
        """
        Initializes model parallelism for the given model and device map.
        References:
            https://github1s.com/huggingface/accelerate/blob/HEAD/src/accelerate/big_modeling.py#L407
        """
        print("Model Parallelism Devices: ", torch.cuda.device_count())
        # Initialize an empty model with the loaded configuration and set the data type to float16
        config = AutoConfig.from_pretrained(self.checkpoint)
        with init_empty_weights():
            raw_model = AutoModelForCausalLM.from_config(config, torch_dtype=torch.float16)
        # Tie the model's weights
        raw_model.tie_weights()

        # Load the checkpoint and dispatch the model to the specified devices
        model = load_checkpoint_and_dispatch(
            raw_model,
            checkpoint=snapshot_download(self.checkpoint),
            device_map="auto",
            no_split_module_classes=["GPTBigCodeBlock"],
            # no_split_module_class = ["GPTJBlock"]  的意思是modules为GPTJBlock的都不会拆分到多个设备上，这里所有需要残差计算的模型都应该写上去。
            dtype=torch.float16
        )

        return model

    def preprocess(self, raw_text: str) -> Tuple[torch.Tensor, torch.Tensor]:
        text = raw_text
        tokens = self.tokenizer.batch_encode_plus([text], return_tensors="pt")
        input_ids = tokens['input_ids']
        return input_ids

    def forward(self, input_text: str):
        inputs = self.tokenizer(input_text, return_tensors="pt")
        inputs = inputs.to(0)
        outputs = self.model.generate(inputs["input_ids"])
        return self.tokenizer.decode(outputs[0].tolist())

    def __call__(self, input_text):
        return self.forward(input_text)


if __name__ == "__main__":
    #import os
    #os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"

    # Create an Inference instance with the specified model directory.
    infer = Inference()

    # ！！！如果需要运行量化版本，请以以下方式load模型！！！
    # If you need to load a quantized model, please instead load the model and then pass it into Inference.__init__.
    # model = MossForCausalLM.from_pretrained("fnlp/moss-moon-003-sft-int4").half().cuda()
    # infer = Inference(model, device_map="auto")

    # Define a test case string.
    test_case = "def hello_world():"

    # Generate a response using the Inference instance.
    res = infer(test_case)
    print(res)
