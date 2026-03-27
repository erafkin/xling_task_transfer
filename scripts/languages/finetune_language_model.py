from transformers import (
    AutoModelForMaskedLM, 
    AutoModelForCausalLM,
    AutoTokenizer, 
    DataCollatorForLanguageModeling,
    TrainingArguments, 
    Trainer,
    default_data_collator
)
import torch
from datasets import load_dataset
from tqdm import tqdm

import os

def train_language_model(model_checkpoint: str, 
                         language: str, 
                         output_dir: str, 
                         mlm: bool = True,
                         mlm_prob: float = 0.15, 
                         num_samples:int = 500000, 
                         batch_size:int = 32):
    
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    
    def preprocess_function(examples):
        tokenized = tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=128
        )
        if not mlm:
            labels = []
        
            for ids in tokenized["input_ids"]:
                labels.append([
                    token if token != tokenizer.pad_token_id else -100
                    for token in ids
                ])

            tokenized["labels"] = labels
        return tokenized
    if mlm:
        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer,mlm=mlm, mlm_probability=mlm_prob)
    else:
        #data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
        data_collator = default_data_collator
    if num_samples > 0:
        lang_dataset = load_dataset("wikimedia/wikipedia", f"20231101.{language}", streaming=True, split="train", cache_dir="/home/scratch/epr41")
        lang_dataset = lang_dataset.take(num_samples)
        lang_dataset = lang_dataset.map(
                preprocess_function,
                batched=True,
                remove_columns=["text"]
            )
        
        max_steps = (num_samples + batch_size - 1) // batch_size
    
    if mlm:
        model = AutoModelForMaskedLM.from_pretrained(
            model_checkpoint,
            dtype=torch.float32,
            device_map="auto"
        )     
    else:
       
        model = AutoModelForCausalLM.from_pretrained(
            model_checkpoint,
            dtype=torch.float32,
            # quantization_config=bnb_config,
            device_map="auto"
        )
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            model.config.pad_token_id = tokenizer.eos_token_id
    training_args = TrainingArguments(
        output_dir=f"{output_dir}/language_{language}",
        eval_strategy="no",
        save_strategy="steps",
        save_steps=100000,
        eval_steps=100000,
        learning_rate=1e-5,
        warmup_ratio=0.05,
        lr_scheduler_type="linear",
        max_grad_norm=1.0,
        per_device_train_batch_size=batch_size,
        max_steps=max_steps,
        num_train_epochs=3,
        overwrite_output_dir=True,
        optim="adamw_torch",
        weight_decay=0.01,
        push_to_hub=False,
        logging_steps=1000,
        gradient_accumulation_steps=2,
        fp16=True,
        report_to='wandb',
        remove_unused_columns=False,
        dataloader_pin_memory=True,
        load_best_model_at_end=False,
        save_total_limit=2,
        greater_is_better=False,
        project='xlt',
        run_name=language

    ) 
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=lang_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,  #
    )
    
    trainer.train()
    trainer.save_model(f"{output_dir}/language_{language}_done")

if __name__ == "__main__":
    languages = ["en", "hi", "es", "de", "zh", "ru", "fr"]
    languages = ["en", "hi", "es"]
    # mlm_model_checkpoints = ["google-bert/bert-base-multilingual-cased", "FacebookAI/xlm-roberta-base"]
    # model_checkpoints = ["Qwen/Qwen3-0.6B"] 
    model_checkpoints = ["ibm-granite/granite-4.0-350m"] 
    for i, checkpoint in enumerate(model_checkpoints):
        if "bert-base" in checkpoint:
            output_dir = "bert-multilingual"
            mlm = True
            batch_size = 32
        elif "roberta-base" in checkpoint:
            output_dir = "xlm-roberta"
            mlm = True
            batch_size = 32
        elif "Qwen" in checkpoint:
            output_dir = "qwen"
            mlm = False
            batch_size = 8
        elif "granite" in checkpoint:
            output_dir = "granite"
            mlm = False
            batch_size = 32
        else:
            raise Exception(f"{checkpoint} not allowed")
        for language in tqdm(languages):
            if not os.path.exists(f"{output_dir}/language_{language}_done"):
                train_language_model(model_checkpoint=checkpoint, language=language, mlm=mlm, output_dir=output_dir,num_samples=2000000, batch_size=batch_size)
           
