from transformers import (
    AutoModelForTokenClassification, 
    AutoTokenizer, 
    BitsAndBytesConfig, 
    DataCollatorForLanguageModeling,
    TrainingArguments, 
    Trainer
)
import torch
from datasets import load_dataset


def train_pos_tagging(language: str, mlm_prob: float = 0.15):
    model_checkpoint = "facebook/xlm-roberta-large"
    model = AutoModelForTokenClassification.from_pretrained(model_checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    def preprocess_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=mlm_prob)
    # lm_dataset = load_dataset("text", data_files={"train": f"{data_folder}/{curricula}/train.train", "val":f"{data_folder}/{curricula}/dev.dev"}) 
    pos_dataset = load_dataset(...)
    pos_dataset = pos_dataset.map(
            preprocess_function,
            batched=True,
            num_proc=4,
        )
    training_args = TrainingArguments(
            output_dir=f"language_{language}",
            eval_strategy="epoch",
            learning_rate=2e-5,
            num_train_epochs=3, 
            weight_decay=0.01,
            push_to_hub=False,
            save_strategy="epoch"
        )
    quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4", 
            bnb_4bit_use_double_quant=True
        )    
        
    model = AutoModelForTokenClassification.from_pretrained(
        "facebook/xlm-roberta-large",
        dtype=torch.float16,
        device_map="auto",
        attn_implementation="flash_attention_2",
        quantization_config=quantization_config
    )
    trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=pos_dataset["train"],
            eval_dataset=pos_dataset["val"],
            data_collator=data_collator,
            tokenizer=tokenizer,
        )
    trainer.train()
    trainer.save_model(f"language_{language}")

    