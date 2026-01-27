from transformers import (
    AutoModelForMaskedLM, 
    AutoTokenizer, 
    DataCollatorForTokenClassification,
    TrainingArguments, 
    AutoConfig,
    Trainer,
    AutoModelForCausalLM,
)
from seqeval.metrics import f1_score
import re
from trl import SFTTrainer, SFTConfig

import torch
from datasets import load_dataset, Dataset
import numpy as np
from torch import nn

from scripts.task_utils import TokenClassificationHead

def train_NER_model(model_checkpoint):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
    NER_dataset = load_dataset("MultiCoNER/multiconer_v2", "English (EN)", trust_remote_code=True)
    # Build a dense mapping 0 … C‑1
    unique_tags = sorted({tag for ex in NER_dataset["test"] for tag in ex["ner_tags"]})
    label2id = {tag: i for i, tag in enumerate(unique_tags)}
    id2label = {i: tag for tag, i in label2id.items()}
    def tokenize_and_align_labels(examples):
        # from https://reybahl.medium.com/token-classification-in-python-with-huggingface-3fab73a6a20e
        tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)
        labels = []
        for i, label in enumerate(examples[f"ner_tags"]):
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

    tokenized_dataset= NER_dataset.map(
        tokenize_and_align_labels,
        batched=True,
        remove_columns=NER_dataset["test"].column_names
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
    output_prefix = "bert-multilingual/base_finetuned" if is_bert else  "xlm-roberta/base_finetuned" 
    training_args = TrainingArguments(
            output_dir=f"{output_prefix}/NER_en",
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
            train_dataset=tokenized_dataset["test"],
            eval_dataset=tokenized_dataset["validation"],
            data_collator=data_collator,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics,
        )

    trainer.train()
    trainer.save_model(f"{output_prefix}/NER_en")

def train_NER_model_causal(model_checkpoint):
    """
        parse dataset to be text-to-text
        then run trainer on that data? but its input-output ? seq2seq trainer i dont think so maybe trainer with labels...
    """
    def extract_tags(text):
        # expects "... NER:\n TAG1 TAG2 ..."
        m = re.search(r"NER:\s*(.*)", text, re.S)
        return m.group(1).strip().split() if m else []
    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        preds_text = tokenizer.batch_decode(preds, skip_special_tokens=True)
        labels_text = tokenizer.batch_decode(labels, skip_special_tokens=True)

        y_pred = [extract_tags(p) for p in preds_text]
        y_true = [extract_tags(l) for l in labels_text]

        return {"f1": f1_score(y_true, y_pred)}
    NER_dataset = load_dataset("MultiCoNER/multiconer_v2", "English (EN)", trust_remote_code=True)
    train_data = []
    for datum in NER_dataset["test"]:
        train_data.append(
            {
                "text": (
                    f"Sentence: {' '.join(datum['tokens'])}.\n NER:\n {' '.join(datum['ner_tags'])}"
                )
            }
        )
    validation_data = []
    for datum in NER_dataset["validation"]:
        validation_data.append(
            {
                "text": (
                    f"Sentence: {' '.join(datum['tokens'])}.\n NER:\n {' '.join(datum['ner_tags'])}"
                )
            }
        )
    train_dataset = Dataset.from_list(train_data)
    validation_dataset = Dataset.from_list(validation_data)
    output_prefix = "qwen/base_finetuned"

    training_args = SFTConfig(
            output_dir=f"{output_prefix}/NER_en",
            eval_strategy="epoch",
            learning_rate=2e-5,
            num_train_epochs=3, 
            weight_decay=0.01,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            push_to_hub=False,
            save_strategy="no",
            fp16=False,
            max_length=512
    )
    trainer = SFTTrainer(
        model=model_checkpoint,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        compute_metrics=compute_metrics
    )    
    
    trainer.train()
    trainer.save_model(f"{output_prefix}/NER_en")



if __name__ == "__main__":
    roberta = "FacebookAI/xlm-roberta-base"
    bert = "google-bert/bert-base-multilingual-cased"
    qwen = "Qwen/Qwen3-0.6B"
    train_NER_model_causal(qwen)
