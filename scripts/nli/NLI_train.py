from transformers import (
    AutoModelForMaskedLM, 
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    DataCollatorForTokenClassification,
    DataCollatorWithPadding,
    TrainingArguments, 
    AutoConfig,
    Trainer
)
import torch
from datasets import load_dataset, DatasetDict, Dataset
import argparse
import numpy as np
from torch import nn

def train_NLI_model(model_checkpoint):
    # Following Ansell: https://github.com/cambridgeltl/composable-sft/blob/main/examples/text-classification/run_text_classification.py
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, local_files_only=True)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    NLI_dataset = load_dataset("facebook/xnli", "en", trust_remote_code=True)
    unique_tags = sorted({ex["label"] for ex in NLI_dataset["train"]})
    label2id = {tag: i for i, tag in enumerate(unique_tags)}
    id2label = {i: tag for tag, i in label2id.items()}
    def preprocess(examples):
        inputs = [examples[column] for column in ["premise", "hypothesis"]]
        tokenized_examples = tokenizer(
            *inputs,
            padding=True,
            max_length=512,
            truncation=True,
        )
        tokenized_examples['label'] = examples['label']
        return tokenized_examples
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        # Simple accuracy calculation
        total = len(predictions)
        correct = sum(1 for pred, lab in zip(predictions, labels) if pred == lab)
        accuracy = correct / total if total > 0 else 0
        return {"accuracy": accuracy}

    tokenized_dataset= NLI_dataset.map(
        preprocess,
        batched=True,
        remove_columns=NLI_dataset["train"].column_names
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        model_checkpoint,
        dtype=torch.float32,
        label2id=label2id,
        id2label=id2label,
        num_labels=len(id2label),
        local_files_only=True,
    ).to(device)

    def set_trainable(mod: nn.Module, train_encoder: bool = False):
        for n, p in mod.named_parameters():
            if "classifier" in n:
                p.requires_grad = True
            else:
                p.requires_grad = train_encoder
        mod.train()

    set_trainable(model, train_encoder=True) 
    training_args = TrainingArguments(
            output_dir=f"bert-multilingual/base_finetuned/NLI_en",
            eval_strategy="epoch",
            learning_rate=2e-5,
            num_train_epochs=3, 
            weight_decay=0.01,
            per_device_train_batch_size=32,
            per_device_eval_batch_size=32,
            push_to_hub=False,
            save_strategy="epoch",
            fp16=False
        )    
    trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset["train"],
            eval_dataset=tokenized_dataset["validation"],
            data_collator=data_collator,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics,
        )

    trainer.train()
    trainer.save_model(f"bert-multilingual/base_finetuned/NLI_en")

if __name__ == "__main__":
    roberta = "FacebookAI/xlm-roberta-base"
    bert = "google-bert/bert-base-multilingual-uncased"
    train_NLI_model(bert)
