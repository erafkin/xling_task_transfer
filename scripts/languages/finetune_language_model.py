from transformers import (
    AutoModelForMaskedLM, 
    AutoModelForCausalLM,
    AutoTokenizer, 
    DataCollatorForLanguageModeling,
    TrainingArguments, 
    Trainer,
    BitsAndBytesConfig
)
import torch
from datasets import load_dataset, Dataset
from tqdm import tqdm
from peft import LoraConfig, get_peft_model,  prepare_model_for_kbit_training

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
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)
    if mlm:
        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer,mlm=mlm, mlm_probability=mlm_prob)
    else:
        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=mlm)

    if num_samples > 0:
        lang_dataset = load_dataset("wikimedia/wikipedia", f"20231101.{language}", streaming=True, split="train", cache_dir="/home/scratch/epr41")
        lang_dataset = lang_dataset.take(num_samples)
        lang_dataset = lang_dataset.map(
                preprocess_function,
                batched=True,
                remove_columns=lang_dataset.column_names
            )
        max_steps = (num_samples + batch_size - 1) // batch_size
    else:
        lang_dataset = load_dataset("wikimedia/wikipedia", f"20231101.{language}",split="train")
        lang_dataset = lang_dataset.map(
                preprocess_function,
                batched=True,
                remove_columns=lang_dataset.column_names,
                num_proc=4
            )
        max_steps = -1
      
    if mlm:
        model = AutoModelForMaskedLM.from_pretrained(
            model_checkpoint,
            dtype=torch.float32,
            device_map="auto"
        )     
    else:
        #PEFT
        lora_config = LoraConfig(
            r=32,
            lora_alpha=64,
            lora_dropout=0.05,
            bias="none",
            target_modules=["q_proj","v_proj"],
        )   
        # quantize    
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_checkpoint,
            dtype=torch.float32,
            quantization_config=bnb_config,
            device_map="auto"
        )
        model = prepare_model_for_kbit_training(model)
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            model.config.pad_token_id = tokenizer.eos_token_id
    training_args = TrainingArguments(
        output_dir=f"{output_dir}/language_{language}",
        eval_strategy="no",
        save_strategy="steps",
        save_steps=100000,
        eval_steps=100000,
        learning_rate=2e-5,
        per_device_train_batch_size=batch_size,
        max_steps=max_steps,
        num_train_epochs=3,
        overwrite_output_dir=True,
        optim="adamw_torch",
        weight_decay=0.01,
        push_to_hub=False,
        logging_steps=10000,
        gradient_accumulation_steps=2,
        fp16=True,
        report_to=None,
        remove_unused_columns=False,
        dataloader_pin_memory=True,
        load_best_model_at_end=False,
        save_total_limit=2,
        greater_is_better=False
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
    mlm = False
    mlm_model_checkpoints = ["google-bert/bert-base-multilingual-cased", "FacebookAI/xlm-roberta-base"]
    model_checkpoints = ["Qwen/Qwen3-0.6B"]  
    batch_size = 8 # need to lower batch size for qwen.
    if mlm:
        batch_size = 32
        model_checkpoints = mlm_model_checkpoints
    for i, checkpoint in enumerate(model_checkpoints):
        if "bert-base" in checkpoint:
            output_dir = "bert-multilingual"
        elif "roberta-base" in checkpoint:
            output_dir = "xlm-roberta"
        elif "Qwen" in checkpoint:
            output_dir = "qwen"
        else:
            raise Exception(f"{checkpoint} not allowed")
        for language in tqdm(languages):
            if not os.path.exists(f"{output_dir}/language_{language}_done"):
                train_language_model(model_checkpoint=checkpoint, language=language, mlm=mlm, output_dir=output_dir,num_samples=1000000, batch_size=batch_size)
           
