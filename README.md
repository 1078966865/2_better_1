# 2_better_1
Official Repository for our ACL paper : Two Intermediate Translations Are Better Than One: Fine-tuning LLMs for Document-level Translation Refinement.

https://arxiv.org/abs/2504.05614

Our checkpoints and datasets are available in https://drive.google.com/drive/folders/1TmcKEdIN6-bGdVY6PaoUQxFMgk22H8-v?usp=sharing

# Installation
Download official LLaMA-Factory https://github.com/hiyouga/LlamaFactory/tree/v0.8.2 .

Then, download 'LLaMA-Factory for training' and replace the folders and files in the official repo. 

```
pip install transformers==4.43.2
```
Then, replace our 'modified transformers-4.43-2/transformers' into '/path/to/your/conda_env/site_packages/transformers' .

# Train
Please register the datasets in training_set_with_QAWeight to datasets.info in LLaMA-Factory for training. 
Then, you can run scripts here for training.
```
llamafactory-cli train config.yaml
```

```
### config.yaml
model_name_or_path: /path/to/your/llama3-8b-Instruct
adapter_name_or_path: /path/to/your/checkpoint
stage: sft
do_train: true
finetuning_type: lora
lora_target: q_proj,v_proj

### ddp
ddp_timeout: 180000000

### dataset

dataset: x1-x2.qaweight.json

# llama3 base need set default
template: llama3
cutoff_len: 4096
# max_samples: 1000
overwrite_cache: true
preprocessing_num_workers: 8

### output
output_dir: /path/to/your/output/dir
logging_steps: 50
save_strategy: epoch
plot_loss: true
overwrite_output_dir: false

### train
per_device_train_batch_size: 2
gradient_accumulation_steps: 8
learning_rate: 0.0001
num_train_epochs: 1
lr_scheduler_type: cosine
warmup_ratio: 0.1
fp16: true

###IMPORTANT!!!
remove_unused_columns: false 
```

### Test
You can use official LLaMA-Factory repo (with files not replaced). You can also use other repos (e.g. ms-swift) for inference. You can directly use the checkpoints in our Google Drive.
```
python src/train_bash.py \
    --stage sft \
    --model_name_or_path /path/to/your/llama3-8b-Instruct \
    --adapter_name_or_path /path/to/your/checkpoint(or our checkpoints)/ \
    --do_predict \
    --dataset /path/to/datasets/with/our/prompts \
    --template llama3 \
    --finetuning_type lora \
    --output_dir path/to/predict/result/ \
    --predict_with_generate \
    --per_device_eval_batch_size 1 \
    --fp16 \
    --cutoff_len 4096 \
    --max_new_tokens 4096 \
    --num_beams 1 \
    --do_sample False \
    >log/sentence_test_base.log

```

