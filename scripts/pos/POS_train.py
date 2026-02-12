from transformers import (
    AutoModelForMaskedLM, 
    AutoTokenizer, 
    AutoModelForCausalLM,
    DataCollatorForTokenClassification,
    TrainingArguments, 
    AutoConfig,
    Trainer,
    BitsAndBytesConfig
)
import torch
from datasets import DatasetDict, Dataset
import numpy as np
from torch import nn

from seqeval.metrics import f1_score
import re
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig, get_peft_model,  prepare_model_for_kbit_training



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
    output_prefix = "bert-multilingual/base_finetuned" if is_bert else  "xlm-roberta/base_finetuned" 

    training_args = TrainingArguments(
            output_dir=f"{output_prefix}/POS_en",
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
    trainer.save_model(f"{output_prefix}/POS_en")

def train_POS_model_causal(model_checkpoint, GUM_folder: str = "GUM_en"):
    """
        parse dataset to be text-to-text
    """
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, trust_remote_code=True)

    model = AutoModelForCausalLM.from_pretrained(
        model_checkpoint,
        dtype=torch.float32,
        device_map="auto"
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id

    def extract_tags(text):
        # expects "... POS:\n TAG1 TAG2 ..."
        m = re.search(r"POS:\s*(.*)", text, re.S)
        return m.group(1).strip().split() if m else []
    
    def compute_metrics(eval_preds):
        # CURRENTLY NOT USING CUDA OOM
        preds, labels = eval_preds
        preds_text = tokenizer.batch_decode(preds, skip_special_tokens=True)
        labels_text = tokenizer.batch_decode(labels, skip_special_tokens=True)

        y_pred = [extract_tags(p) for p in preds_text]
        y_true = [extract_tags(l) for l in labels_text]
        return {"f1": f1_score(y_true, y_pred)}
    
    train_dataset = load_conllu_data(f"{GUM_folder}/en_gum-ud-train.conllu")
    dev_dataset = load_conllu_data(f"{GUM_folder}/en_gum-ud-dev.conllu")
    dataset = DatasetDict({"train": Dataset.from_pandas(train_dataset), "dev": Dataset.from_pandas(dev_dataset)})
    train_data = []
    for datum in dataset["train"]:
        train_data.append(
            {
                "text": (
                    f"Sentence: {' '.join(datum['tokens'])}.\n POS:\n {' '.join(datum['pos_tags'])}"
                )
            }
        )
    validation_data = []
    for datum in dataset["dev"]:
        validation_data.append(
            {
                "text": (
                    f"Sentence: {' '.join(datum['tokens'])}.\n POS:\n {' '.join(datum['pos_tags'])}"
                )
            }
        )
    train_dataset = Dataset.from_list(train_data)
    validation_dataset = Dataset.from_list(validation_data)
    output_prefix = "qwen/base_finetuned"

    training_args = SFTConfig(
            output_dir=f"{output_prefix}/POS_en",
            eval_strategy="epoch",
            learning_rate=2e-5,
            num_train_epochs=10, 
            weight_decay=0.01,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=4,
            push_to_hub=False,
            save_strategy="no",
            warmup_ratio=0.05,
            lr_scheduler_type="linear",
            max_grad_norm=1.0,
            fp16=True,
            max_length=512,
            report_to='wandb',
            project='xlt',
            run_name="POS_en"
    )
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset
    )    
    
    trainer.train()
    trainer.save_model(f"{output_prefix}/POS_en")

if __name__ == "__main__":
    roberta = "FacebookAI/xlm-roberta-base"
    bert = "google-bert/bert-base-multilingual-cased"
    qwen = "Qwen/Qwen3-0.6B"
    train_POS_model_causal(qwen)
