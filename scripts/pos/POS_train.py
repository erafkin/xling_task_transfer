from transformers import (
    AutoModelForMaskedLM, 
    AutoTokenizer, 
    DataCollatorForTokenClassification,
    TrainingArguments, 
    AutoConfig,
    Trainer
)
import torch
from datasets import load_dataset, DatasetDict, Dataset
import numpy as np
from torch import nn

from scripts.task_utils import load_conllu_data, TokenClassificationHead

def train_POS_model(model_checkpoint, GUM_folder: str = "GUM_en"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dataset = load_conllu_data(f"{GUM_folder}/en_gum-ud-train.conllu")
    dev_dataset = load_conllu_data(f"{GUM_folder}/en_gum-ud-dev.conllu")
    dataset = DatasetDict({"train": Dataset.from_pandas(train_dataset), "dev": Dataset.from_pandas(dev_dataset)})
    unique_tags = sorted({tag for ex in dataset["train"] for tag in ex["pos_tags"]})
    label2id = {tag: i for i, tag in enumerate(unique_tags)}
    id2label = {i: tag for tag, i in label2id.items()}
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
    def tokenize_and_align_labels(examples):
        # from https://reybahl.medium.com/token-classification-in-python-with-huggingface-3fab73a6a20e
        tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)
        labels = []
        for i, label in enumerate(examples["pos_tags"]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)  # Map tokens to their respective word.
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:  # Set the special tokens to -100.
                if word_idx is None:
                    label_ids.append(-100)
                elif word_idx != previous_word_idx:  # Only label the first token of a given word.
                    label_ids.append(label2id[label[word_idx]])
                else:
                    label_ids.append(-100)
                previous_word_idx = word_idx
            labels.append(label_ids)

        tokenized_inputs["labels"] = labels
        return tokenized_inputs
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=2)

        # Remove ignored labels (-100)
        true_predictions = [
            [p for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [l for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]

        # Simple accuracy calculation
        total = sum(len(pred) for pred in true_predictions)
        correct = sum(1 for pred, lab in zip(true_predictions, true_labels) for p, l in zip(pred, lab) if p == l)
        accuracy = correct / total if total > 0 else 0
        return {"accuracy": accuracy}

    tokenized_dataset= dataset.map(
        tokenize_and_align_labels,
        batched=True,
        remove_columns=["tokens", "pos_tags", "dep_rel", "dep_head"]
    )

    config = AutoConfig.from_pretrained(model_checkpoint)
    mlm_model = AutoModelForMaskedLM.from_pretrained(
        model_checkpoint,
        config=config,
        dtype=torch.float32,
    ).to(device)

    if "roberta" in model_checkpoint:
        encoder = mlm_model.roberta
        is_bert = False
    else:
        encoder = mlm_model.bert
        is_bert = True
    model = TokenClassificationHead(
        encoder=encoder,
        num_labels=len(id2label),
        dropout=0.1,
        bert=is_bert
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
            output_dir=f"xlm-roberta/lang_en_finetuned/POS_en",
            eval_strategy="epoch",
            learning_rate=2e-5,
            num_train_epochs=3, 
            weight_decay=0.01,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            push_to_hub=False,
            save_strategy="no",
            fp16=False
        )    
    trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset["train"],
            eval_dataset=tokenized_dataset["dev"],
            data_collator=data_collator,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics,
        )

    trainer.train()
    trainer.save_model(f"xlm-roberta/lang_en_finetuned/POS_en")

if __name__ == "__main__":
    roberta = "FacebookAI/xlm-roberta-base"
    bert = "google-bert/bert-base-multilingual-uncased"
    train_POS_model(roberta)