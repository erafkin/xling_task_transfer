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
                remove_columns=["text"],
                batch_size=64
            )
        max_steps = (1000000 + 32 - 1) // 32
    else:
        lang_dataset = load_dataset("wikimedia/wikipedia", f"20231101.{language}",split="train")
        lang_dataset = lang_dataset.map(
                preprocess_function,
                batched=True,
                remove_columns=["text"],
                batch_size=64
            )
        max_steps = None
    training_args = TrainingArguments(
            output_dir=f"language_{language}",
            eval_strategy="no",
            learning_rate=2e-5,
            per_device_train_batch_size=32,
            max_steps=max_steps,
            num_train_epochs=3, 
            weight_decay=0.01,
            push_to_hub=False,
            save_strategy="no"
        )   
        
    model = AutoModelForMaskedLM.from_pretrained(
        model_checkpoint,
        dtype=torch.float16,
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
    languages = ["hi", "es", "de", "zh", "en"]
    for language in tqdm(languages):
        train_language_model(language=language)

