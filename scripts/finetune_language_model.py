from transformers import (
    AutoModelForMaskedLM, 
    AutoTokenizer, 
    DataCollatorForLanguageModeling,
    TrainingArguments, 
    Trainer
)
import torch
from datasets import load_dataset, Dataset
from tqdm import tqdm

def train_language_model(language: str, mlm_prob: float = 0.15):
    model_checkpoint = "google-bert/bert-base-multilingual-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    def preprocess_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=mlm_prob)
    if language == "en":
        lang_dataset = load_dataset("wikimedia/wikipedia", f"20231101.{language}", streaming=True, split="train", cache_dir="/home/scratch/epr41")
        lang_dataset = lang_dataset.take(500000)
        lang_dataset = lang_dataset.map(
                preprocess_function,
                batched=True,
                remove_columns=lang_dataset.column_names
            )
        max_steps = (500000 + 32 - 1) // 32
    else:
        lang_dataset = load_dataset("wikimedia/wikipedia", f"20231101.{language}",split="train")
        lang_dataset = lang_dataset.map(
                preprocess_function,
                batched=True,
                remove_columns=lang_dataset.column_names,
                num_proc=4
            )
        max_steps = -1
    training_args = TrainingArguments(
            output_dir=f"language_{language}",
            eval_strategy="no",
            save_strategy="steps",
            save_steps=500,
            eval_steps=500,
            learning_rate=2e-5,
            per_device_train_batch_size=32,
            max_steps=max_steps,
            num_train_epochs=3,
            overwrite_output_dir=True,
            optim="adamw_torch",
            weight_decay=0.01,
            push_to_hub=False,
            logging_steps=100,
            gradient_accumulation_steps=2,
            fp16=True,
            report_to=None,
            remove_unused_columns=False,
            load_best_model_at_end=False,
            save_total_limit=2,
            greater_is_better=False
        )   
        
    model = AutoModelForMaskedLM.from_pretrained(
        model_checkpoint,
        dtype=torch.float32,
        device_map="auto"
    )
    trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=lang_dataset,
            data_collator=data_collator,
            tokenizer=tokenizer,
        )
    trainer.train()
    trainer.save_model(f"language_{language}")

if __name__ == "__main__":
    #languages = ["hi", "es", "de", "zh", "en"]
    languages = ["en"]
    for language in tqdm(languages):
        train_language_model(language=language)

