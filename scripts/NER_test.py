import torch
import numpy as np
from typing import List
from datasets import load_dataset
from transformers import AutoModelForMaskedLM, AutoTokenizer
from tqdm import tqdm
import math
from sklearn.metrics import precision_score, recall_score, f1_score
from safetensors.torch import load_model
from task_vectors import TaskVector
from finetune_tasks import TokenClassificationHead

def get_language_vector(base_model: str, saved_language: str):
    lang_vector = TaskVector(pretrained_checkpoint=AutoModelForMaskedLM.from_pretrained(base_model),
                             finetuned_checkpoint=AutoModelForMaskedLM.from_pretrained(saved_language, local_files_only=True))
    return lang_vector

def apply_language_vector_to_model(ner_model_checkpoint: str, language_vector:TaskVector, lambda_coef: float):
    ner_model = language_vector.apply_to(ner_model_checkpoint, scaling_coef=lambda_coef)
    return ner_model

def compute_metrics(predictions, labels):

    # Remove ignored labels (-100)
    y_pred = [
        [p for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    y_true = [
        [l for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    assert(len(y_true) == len(y_pred))
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    return precision, recall, f1

def lambda_search(eval_set, coef):
    ...

def get_label_mapping():
    NER_dataset = load_dataset("MultiCoNER/multiconer_v2", "English (EN)", trust_remote_code=True)
    # Build a dense mapping 0 … C‑1
    unique_tags = sorted({tag for ex in NER_dataset["test"] for tag in ex["ner_tags"]})
    label2id = {tag: i for i, tag in enumerate(unique_tags)}
    id2label = {i: tag for tag, i in label2id.items()}
    return id2label, label2id

def test_lang_ner(ner_model, language_model, pretrained_checkpoint, language_dataset, lambdas: List[float]= [0.0, 0.25, 0.5, 0.75, 1.0], batch_size:int=32):
    if language_dataset != "English (EN)":
        lv = get_language_vector(pretrained_checkpoint, language_model)
        best_lambda = 1.0
        ner = apply_language_vector_to_model(ner_model, lv, best_lambda) # TODO find best lambda
    else:
       ner = TokenClassificationHead()
       load_model(ner, language_model)

       ner = torch.load(pretrained_checkpoint) 
    NER_dataset = load_dataset("MultiCoNER/multiconer_v2", language_dataset, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(pretrained_checkpoint)
    id2label, label2id = get_label_mapping()
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
    tokenized_dataset= NER_dataset.map(
        tokenize_and_align_labels,
        batched=True,
    )
    def batches(it):
        for i in range(0, len(it), batch_size):
            yield it[i:i+batch_size]

    preds = []
    labels = []
    ner.eval()

    with torch.no_grad():
        for batch_sent in tqdm(batches(tokenized_dataset), total=math.ceil(len(tokenized_dataset)/batch_size), desc="Eval"):
            ps = ner(**batch_sent["input_ids"])
            ps = np.argmax(ps, axis=2)
            ls = batch_sent["labels"]
            preds += ps
            labels += ls
    p, r, f1 = compute_metrics(preds, labels, id2label)
    return p, r, f1

if __name__ == "__main__":
    datasets = ["English (EN)", "Spanish (ES)", "Hindi (HI)"]#, "German (DE)", "Chinese (ZH)"]
    language_models = ["language_en_done", "language_es_done", "language_hi_done"]
    with open("output/NER.txt", "w") as f:
        for idx, model in enumerate(language_models):
            p, r, f1 = test_lang_ner("NER_en", model, "language_en_done", datasets[idx])
            f.write(f"\n======language: {model.split('_')[1]}=======\n")
            f.write(f"precision: {p}\n")
            f.write(f"recall: {r}\n")
            f.write(f"f1: {f1}")
        f.close()





