import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"#"1"#
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # To avoid warnings about parallelism in tokenizers
import sys
sys.path.append("..")
import torch
from trl import DPOConfig
from transformers import AutoModelForCausalLM,AutoTokenizer,BitsAndBytesConfig,Qwen2Tokenizer
from datasets import Dataset
import re

from trl.trainer.dpo_trainer import DPOTrainer

def check_length(example:dict,max_length,tokenizer:Qwen2Tokenizer):
    conversations=[{"role":"user","content":example["prompt"]}
                   ,{"role":"assistant","content":example["chosen"]}]

    input_ids = tokenizer.apply_chat_template(conversations)
    if len(input_ids)>=max_length:
        return False

    conversations=[{"role":"user","content":example["prompt"]}
                   ,{"role":"assistant","content":example["rejected"]}]

    input_ids = tokenizer.apply_chat_template(conversations)
    if len(input_ids)>=max_length:
        return False

    return True


def full_dpo_data_pre(example:dict,model_type="qwen"):
    #形成conversation
    if model_type=="qwen":
        prompt_no_input = (
            "<|im_start|>user\n{instruction}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
    elif model_type=="llama3":
        prompt_no_input = (
            "<|start_header_id|>user<|end_header_id|>\n\n"
            "{instruction}\n<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        )
    else:
        raise ValueError("model_type must be qwen or llama3")

    #如果example有prompt
    if "prompt" in example:
        instruction=example["prompt"]
    else:
        instruction=example["instruction"]

    prompt=prompt_no_input.format(instruction=instruction.strip())

    if "chosen" in example and "rejected" in example:
        chosen_response=example["chosen"]
        rejected_response=example["rejected"]
    else:
        chosen_response=example["chosen_response"]
        rejected_response=example["rejected_response"]

    if model_type=="qwen":
        eos_token="<|im_end|>"
    elif model_type=="llama3":
        eos_token="<|eot_id|>"
    else:
        raise ValueError("model_type must be qwen or llama3")

    chosen_response=chosen_response+eos_token
    rejected_response=rejected_response+eos_token

    return {"prompt":prompt,"chosen":chosen_response,"rejected":rejected_response}


def main():
    max_seq_length=2048 # Supports automatic RoPE Scaling, so choose any number.
    # 加载模型
    model_name="//data_backup/yyj/longrope训练/fineweb_2b_zh_magpie/checkpoint-37282"#"//data/models/LLM-Research/Meta-Llama-3-8B-Instruct"
    output_dir="./dpo_output/2b-magpie-zh-ultrafeedback-zh-2k_2epoch"#"./dpo_output/llama3-8b-ins-patient-cot-dpo-full2"
    print("output_dir:",output_dir)
    training_args = DPOConfig(
        output_dir=output_dir,
        max_length=max_seq_length,
        max_prompt_length=500,
        max_completion_length=2000,

        evaluation_strategy="no",
        eval_steps=1,
        report_to="wandb",
        logging_strategy='steps',
        logging_steps=10,
        logging_dir=os.path.join(output_dir, 'logs'),
        save_strategy='steps',
        save_steps=250,
        num_train_epochs=2,  #只训练1个epoch
        remove_unused_columns=False,
        ignore_data_skip=True,
        save_only_model=True,

        optim="adamw_torch",  #'paged_adamw_8bit',#
        weight_decay=0,

        lr_scheduler_type="linear",#"constant_with_warmup",  # ,  #
        warmup_ratio=0.05,
        learning_rate=5e-6,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        max_grad_norm=1.0,

        # max_steps=1,

        fp16=False,
        bf16=True,
        deepspeed='ds_config_constant.json',
        auto_find_batch_size=False,
        load_best_model_at_end=False,
        #torch_compile_backend="inductor",
    )


    # 准备tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name,
                                               #use_fast=True,
                                               padding_side='right',
                                               add_bos_token=False,
                                               add_eos_token=False,
                                               #legacy=False,
                                               model_max_length=max_seq_length,
                                               trust_remote_code=True,)
    if tokenizer.eos_token is None:
        tokenizer.eos_token="</s>"
        tokenizer.eos_token_id=tokenizer.convert_tokens_to_ids(tokenizer.eos_token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    if tokenizer.bos_token is None:
        tokenizer.bos_token = tokenizer.eos_token
        tokenizer.bos_token_id = tokenizer.eos_token_id

    # 加载数据集
    train_dataset = Dataset.from_parquet("./ultrafeedback_zh/ultrafeedback_zh_binarized.parquet")

    with training_args.main_process_first(desc="Preprocessing dataset"):
        train_dataset = train_dataset.map(lambda x: full_dpo_data_pre(x,model_type="qwen"),
                                          batched=False,
                                          remove_columns=train_dataset.column_names,
                                          load_from_cache_file=True,
                                          num_proc=32)
        #删除长度过长的数据
        train_dataset = train_dataset.filter(lambda x: check_length(x,2048,tokenizer),batched=False,num_proc=32)
        print("train dataset length:",len(train_dataset))

    model = AutoModelForCausalLM.from_pretrained(model_name,
                                             torch_dtype=torch.bfloat16,
                                             #quantization_config=bnb_config,
                                             trust_remote_code=True,
                                             attn_implementation="flash_attention_2",
                                             # device_map="auto"
                                             )

    ref_model = AutoModelForCausalLM.from_pretrained(model_name,
                                             torch_dtype=torch.bfloat16,
                                             #quantization_config=bnb_config,
                                             trust_remote_code=True,
                                             attn_implementation="flash_attention_2",
                                             # device_map="auto"
                                             )

    # 准备lora模型
    #model = get_lora_model(model, quantized=False)

    dpo_trainer = DPOTrainer(
        model,
        ref_model=ref_model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
    )
    dpo_trainer.train(resume_from_checkpoint=False)

    print("save to", output_dir)

if __name__ == '__main__':
    main()

    # nohup deepspeed --include localhost:0,1,2,3,4,5,6,7 --master_port='29509' 训练qwen-dpo多卡-ultrafeedback.py >训练qwen-dpo.log 2>&1 &
    # wandb login --relogin fe82adb2256d08a1241a8a31c4dab0d962710045