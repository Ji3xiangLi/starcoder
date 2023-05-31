from typing import Union, List, Tuple, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from huggingface_hub import snapshot_download
from accelerate import init_empty_weights, load_checkpoint_and_dispatch


PREFIX = None


class Inference:
    def __init__(
        self,
        model: Optional = None,
        model_dir: Optional[str] = None,
        parallelism: bool = True,
        device_map: Optional[Union[str, List[int]]] = None,
    ) -> None:
        """
        Initializes the MossModel with a given model or loads a model from the specified directory.

        Args:
            model (Optional[MossForCausalLM], optional): An existing model to use. Defaults to None.
            model_dir (Optional[str], optional): The directory containing the pre-trained model files. Defaults to None.
            parallelism (bool, optional): Whether to initialize model parallelism. Defaults to True.
            device_map (Optional[Union[str, List[int]]], optional): The list of GPU device indices for model parallelism or "auto" to use the default device map. Defaults to None.
        """
        self.model_dir = "bigcode/starcoder" if not model_dir else model_dir

        if model:
            self.model = model
        else:
            self.model = (
                self.Init_Model_Parallelism(raw_model_dir=self.model_dir, device_map=device_map)
                if parallelism
                else AutoModelForCausalLM.from_pretrained(self.model_dir)
            )

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir)

    def Init_Model_Parallelism(self, raw_model_dir: str, device_map: Union[str, List[int]] = "auto") -> MossForCausalLM:
        """
        Initializes model parallelism for the given model and device map.

        Args:
            raw_model_dir (str): The directory containing the pre-trained model files.
            device_map (Union[str, List[int]], optional): The list of GPU device indices for model parallelism, or "auto" to use the default device map. Defaults to "auto".

        Returns:
            MossForCausalLM: The model with model parallelism initialized.

        References:
            https://github1s.com/huggingface/accelerate/blob/HEAD/src/accelerate/big_modeling.py#L407
        """
        # Print the number of CUDA devices available
        print("Model Parallelism Devices: ", torch.cuda.device_count())
        if not os.path.exists(raw_model_dir):
            raw_model_dir = snapshot_download(raw_model_dir)

        # Initialize an empty model with the loaded configuration and set the data type to float16
        with init_empty_weights():
            raw_model = AutoModelForCausalLM.from_pretrained(raw_model_dir, torch_dtype=torch.float16)

        # Tie the model's weights
        raw_model.tie_weights()

        # Load the checkpoint and dispatch the model to the specified devices
        model = load_checkpoint_and_dispatch(
            raw_model,
            raw_model_dir,
            device_map="auto" if not device_map else device_map,
            dtype=torch.float16
        )

        return model

    def preprocess(self, raw_text: str) -> Tuple[torch.Tensor, torch.Tensor]:
        text = raw_text
        tokens = self.tokenizer.batch_encode_plus([text], return_tensors="pt")
        input_ids = tokens['input_ids']
        return input_ids

    def forward(self, data: str):

        input_ids = self.preprocess(data)
        outputs = self.model.generate(input_ids)
        print(self.tokenizer.ecode(outputs[0]))


    def __call__(self, input):
        return self.forward(input)


if __name__ == "__main__":
    import os

    # os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

    # Create an Inference instance with the specified model directory.
    infer = Inference(device_map="auto")

    # ！！！如果需要运行量化版本，请以以下方式load模型！！！
    # If you need to load a quantized model, please instead load the model and then pass it into Inference.__init__.
    # model = MossForCausalLM.from_pretrained("fnlp/moss-moon-003-sft-int4").half().cuda()
    # infer = Inference(model, device_map="auto")

    # Define a test case string.
    test_case = "def hello_world():"

    # Generate a response using the Inference instance.
    res = infer(test_case)

    # Print the generated response.
    print(res)
