from stopes.eval.alti.wrappers.transformer_wrapper import FairseqTransformerHub
from stopes.eval.alti.alti_metrics.alti_metrics_utils import (
    compute_alti_nllb,
    compute_alti_metrics,
)
from tqdm import tqdm
import numpy as np
import torch
import argparse

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description="Compute ALTI metrics for translations.")
parser.add_argument("--src_lang", type=str, required=True, help="Source language code")
parser.add_argument("--tgt_lang", type=str, required=True, help="Target language code")


# load the model, vocabulary and the sentencepiece tokenizer
hub = FairseqTransformerHub.from_pretrained(
    checkpoint_dir="/path/to/your/ALTI/stopes-main",
    checkpoint_file="checkpoint_best.pt",
    data_name_or_path="/path/to/your/ALII/stopes-main/wmt18_de-en",
    bpe="sentencepiece",
    sentencepiece_model="/path/to/your/ALII/stopes-main/sentencepiece.joint.bpe.model",
)
hub = hub.to(device)
scores = []

src_file = open(f"path/to/your/src", "r")
src_lines = src_file.readlines()

tgt_file = open(f"path/to/your/tgt", "r")
tgt_lines = tgt_file.readlines()
# translate an example of a German sentence to English.
# the source sentence means "The breakfast buffet is very good and varied.", so the translation is wrong.
res = []
for idx in tqdm(range(len(src_lines))):
    src = src_lines[idx]
    tgt = tgt_lines[idx]
    scores.append(compute_alti_metrics(*compute_alti_nllb(hub, src, tgt))["avg_sc"])

print(res)

