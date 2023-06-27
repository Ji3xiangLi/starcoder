import pandas as pd
"""将外部数据集调整成finetune格式。"""

# leetcode-instructions
a = pd.read_json("leetcode_instructions.jsonl", lines=True)
print(a.head(2))
a["input"] = a.apply(lambda x: "\n".join([x["instruction"], x["input"]]), axis=1)  # 修改输入

a = a.drop(["instruction"], axis=1)
a.to_json("data/leetcode_instructions.jsonl", orient="records", force_ascii=False, lines=True)