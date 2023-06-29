"""
单机多卡的方式启动starcoder模型，并且启动两个推理接口。
启动前需要提前指定visible devices

export CUDA_VISIBLE_DEVICES="4,5"
nohup python api.py &
"""
from inference import load_dispatched_model_and_tokenizer, StarCoderChat, StarCoderInference
from flask import Flask, request
from flask_cors import *

app = Flask(__name__)

@app.route("/inference", methods=["GET"])
@cross_origin()
def get_infrence_response():
    """
    get方式请求<host>:<port>/inference，param key:"input", value:<输入文本>
    """
    input_text = request.args.get("input", "")
    try:
        res = inference(input_text)
        return {"success": 1, "response": res}
    except Exception as e:
        return {"success": 0, "response":"/", "error": e}


@app.route("/chat", methods=["GET"])
@cross_origin()
def get_chat_response():
    input_text = request.args.get("input", "")
    res = chat(input_text)
    try:
        return {"success": 1, "response": res}
    except Exception as e:
        return {"success": 0, "response": "/", "error": e}


if __name__ == '__main__':
    # starcoder_id = "bigcode/starcoder"
    starcoder_path = "../models/starcoder"
    starcoder_model, starcoder_tokenizer = load_dispatched_model_and_tokenizer(starcoder_path)

    # 实例化聊天模式
    chat = StarCoderChat(starcoder_model, starcoder_tokenizer)

    # 实例化推理模式，调用infer(input_text) 直接将input_text输入到模型中返回模型的全部生成值
    inference = StarCoderInference(starcoder_model, starcoder_tokenizer)

    app.run(host="0.0.0.0", port=12342)
