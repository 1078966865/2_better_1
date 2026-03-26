import json
import re
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
from tqdm import tqdm
import math
import numpy as np


prddata = []

def calculate_perplexity(sentence, model, tokenizer):
    # 对输入句子进行编码
    inputs = tokenizer(sentence, return_tensors="pt").to("cuda")
    with torch.no_grad():
        # 获取模型的输出
        outputs = model(**inputs, labels=inputs["input_ids"])
        # 计算交叉熵损失
        loss = outputs.loss
        # 困惑度是交叉熵损失的指数
        perplexity = torch.exp(loss).item()
    return perplexity
  
for line in tqdm(prddata):
    line = deleteid(line)
    if len(line.split(" ")) > 6 and len(line.split(" ")) < 1024:  # 句子长度大于5
        temp = calculate_perplexity(line, model, tokenizer)
        if math.isnan(temp) == False:
            res.append(temp)


res = torch.Tensor(res)
print(torch.mean(res))
