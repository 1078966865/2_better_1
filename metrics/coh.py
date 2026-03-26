import re
import json
import os
import sacrebleu
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import json
import torch
import time
from tqdm import tqdm
import numpy as np
import math
from transformers import pipeline
import torch.nn.functional as F
from tqdm.auto import tqdm
from torch.utils.data import Dataset

import torch.utils.data as data


class ListDataset(data.Dataset):
    def __init__(self, data_list):
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        return self.data_list[index]


file_prd_lines = []
max_seq_len = 512
overall_res = []


def delete_id_into_sent(text):
    # 使用正则表达式匹配句子前面的#id，并去除
    pattern = r"#\d+\s+"
    result = re.sub(pattern, "# ", text)
    # 按照句子分割文本，并去除空字符串
    sentences = [sentence.strip() for sentence in result.split("#") if sentence.strip()]

    return sentences
  
pre = [] # 前一句
cont = []# 后一句
for line in file_prd_lines:
    hyp_sents = delete_id_into_sent(line)
    pre.extend(hyp_sents[:-1])
    cont.extend(hyp_sents[1:])
    assert len(pre) == len(cont)
print(len(pre))
print("start embedding")
  pipe = pipeline(
      "feature-extraction",
      model="princeton-nlp/sup-simcse-roberta-large",
      device="cuda",
      truncation=True,
  )
predataset = ListDataset(pre)
contdataset = ListDataset(cont)
embpre = []
embcont = []
for line in tqdm(pipe(predataset, batch_size=64)):
    embpre.append(line)
for line in tqdm(pipe(contdataset, batch_size=64)):
    embcont.append(line)
res = []
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

for idx in tqdm(range(0, len(embpre))):
    emba = torch.Tensor(embpre[idx]).to(device)
    embb = torch.Tensor(embcont[idx]).to(device)
    emba = torch.mean(emba, dim=1)
    embb = torch.mean(embb, dim=1)
    res.append(F.cosine_similarity(emba, embb).cpu().numpy())
# print(res)
res = np.array(res)
print(np.mean(res))
overall_res.append(np.mean(res))

print(overall_res)
