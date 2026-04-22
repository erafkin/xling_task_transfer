from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, AutoModelForMaskedLM
from trl import SFTTrainer, SFTConfig
from datasets import Dataset, DatasetDict

import wandb
import torch
import requests

def get_unimorph_data(lang:str = "eng") -> DatasetDict:
    base = "https://raw.githubusercontent.com/unimorph/"
    url = f"{base}{lang}/master/{lang}"
    text = requests.get(url).text.strip().split("\n")
    dataset_as_list = [
        {
            "root": l[0],
            "inflection": l[1],
            "features": l[2] if len(l) > 2 else "None"
        }
        for line in text
        for l in [line.split("\t")]
        if len(l) >= 2
    ]
    dataset = Dataset.from_list(dataset_as_list)
    dataset_dict_temp = dataset.train_test_split(test_size=0.2, seed=42)
    dataset_dict_temp1 = dataset_dict_temp["test"].train_test_split(test_size=0.5, seed=42)
    dataset_dict = DatasetDict({"train": dataset_dict_temp["train"], "validation": dataset_dict_temp1["train"], "test": dataset_dict_temp1["test"]})
    return dataset_dict

def train_model_causal(model_checkpoint):
    """
        parse dataset to be text-to-text
    """
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, trust_remote_code=True)
    if "bert" in model_checkpoint:
        config = AutoConfig.from_pretrained(model_checkpoint)
        config.is_decoder = True
        config.add_cross_attention = False

        model = AutoModelForMaskedLM.from_pretrained(
            model_checkpoint,
            config=config,
        )
    else:
    
        model = AutoModelForCausalLM.from_pretrained(
            model_checkpoint,
            dtype=torch.float32,
            device_map="auto",
        )
    if tokenizer.eos_token is None:
        tokenizer.eos_token = "</s>"

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.config.eos_token_id = tokenizer.convert_tokens_to_ids(tokenizer.eos_token)
    model.config.pad_token_id = tokenizer.pad_token_id
    
    train_data = []
    dataset = get_unimorph_data()
    for datum in dataset["train"]:
        train_data.append(
            {
                "text": (
                    f"Root: {datum['root']}\n Features: {datum['features']}\n Inflection: {datum['inflection']}"
                )
            }
        )
    validation_data = []
    for datum in dataset["validation"]:
        validation_data.append(
            {
                "text": (
                    f"Root: {datum['root']}\n Features: {datum['features']}\n Inflection: {datum['inflection']}"
                )
            }
        )
    train_dataset = Dataset.from_list(train_data)
    validation_dataset = Dataset.from_list(validation_data)
    if "roberta" in model_checkpoint:
        output_prefix = "xlm-roberta/base_finetuned"
        run_prefix = "roberta"
    elif "bert" in model_checkpoint:
        output_prefix = "bert-multilingual/base_finetuned"
        run_prefix = "bert"
    elif "granite" in model_checkpoint:
        output_prefix = "granite/base_finetuned"
        run_prefix = "granite"
    else:
        output_prefix = "qwen/base_finetuned"
        run_prefix = "qwen"


    training_args = SFTConfig(
            output_dir=f"{output_prefix}/morph_en",
            eval_strategy="epoch",
            learning_rate=1e-5,
            num_train_epochs=3, 
            weight_decay=0.01,
            per_device_train_batch_size=16 if "granite" in model_checkpoint else 8,
            per_device_eval_batch_size=8 if "granite" in model_checkpoint else 4,
            push_to_hub=False,
            save_strategy="no",
            warmup_ratio=0.01,
            lr_scheduler_type="linear",
            max_grad_norm=1.0,
            bf16=True,
            max_length=512,
            report_to='wandb',
            project='xlt',
            run_name=f"{run_prefix}_morph_en",
    )
    if "granite" in model_checkpoint:
        model.config.use_cache = False
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset
        )    
    
    trainer.train()
    trainer.save_model(f"{output_prefix}/QA_en")
    wandb.finish()

if __name__ == "__main__":
    roberta = "FacebookAI/xlm-roberta-base"
    bert = "google-bert/bert-base-multilingual-cased"
    qwen = "Qwen/Qwen3-0.6B"
    granite = "ibm-granite/granite-4.0-350m"
    for m in [bert, roberta, qwen, granite]:
        train_model_causal(m)
