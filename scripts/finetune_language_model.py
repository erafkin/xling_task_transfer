from transformers import (
    AutoModelForMaskedLM, 
    AutoTokenizer, 
    BitsAndBytesConfig, 
    DataCollatorForLanguageModeling,
    TrainingArguments, 
    Trainer
)
import torch
from datasets import load_dataset
from tqdm import tqdm

def train_language_model(language: str, mlm_prob: float = 0.15):
    model_checkpoint = "FacebookAI/xlm-roberta-large"
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    def preprocess_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=mlm_prob)
    lang_dataset = load_dataset("wikimedia/wikipedia", f"20231101.{language}")
    lang_dataset = lang_dataset.map(
            preprocess_function,
            batched=True,
            num_proc=4,
        )
    training_args = TrainingArguments(
            output_dir=f"language_{language}",
            eval_strategy="epoch",
            learning_rate=2e-5,
            num_train_epochs=1, 
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
        
    model = AutoModelForMaskedLM.from_pretrained(
        model_checkpoint,
        dtype=torch.float16,
        device_map="auto",
        attn_implementation="flash_attention_2",
        quantization_config=quantization_config
    )
    trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=lang_dataset["train"],
            eval_dataset=lang_dataset["val"],
            data_collator=data_collator,
            tokenizer=tokenizer,
        )
    trainer.train()
    trainer.save_model(f"language_{language}")

if __name__ == "__main__":
    languages = ["lt", "es", "de", "he"]
    for language in tqdm(languages):
        train_language_model(language=language)