from typing import Tuple
import torch
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from accelerate import init_empty_weights, load_checkpoint_and_dispatch
from flask import Flask

app = Flask(__name__)

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

        model = load_checkpoint_and_dispatch(
            raw_model,
            checkpoint=snapshot_download(self.checkpoint),
            device_map="auto",
            no_split_module_classes=["GPTBigCodeBlock"],  # 根据模型结构修改，不能拆分到多个设备的modulename，含有残差的module都应该写上去。
            dtype=torch.float16)

        return model

    def forward(self, input_text: str):
        inputs = self.tokenizer(input_text, return_tensors="pt")
        inputs = inputs.to(0)
        outputs = self.model.generate(inputs["input_ids"])
        return self.tokenizer.decode(outputs[0].tolist())

    def __call__(self, input_text):
        return self.forward(input_text)

@app.run("/", methods=["GET"])
@cross_origin()
def run():
    input_text = request.args.get("input", "")
    res = infer(input_text)
    return {"success":1, "response": res}

if __name__ == "__main__":
    infer = Inference()
    app.run(host = "0.0.0.0", port=12342)
    # test_case = "please write a quick sort code"
    # res = infer(test_case)
    # print(res)
