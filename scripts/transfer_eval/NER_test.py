import torch
import numpy as np
from typing import List
from datasets import load_dataset
from task_vectors import TaskVector
from transformers import AutoModelForMaskedLM, AutoTokenizer
from tqdm import tqdm
import math

def get_language_vector(base_model: str, saved_language: str):
    lang_vector = TaskVector(pretrained_checkpoint=AutoModelForMaskedLM.from_pretrained(base_model),
                             finetuned_checkpoint=AutoModelForMaskedLM.from_pretrained(saved_language, local_files_only=True))
    return lang_vector

def apply_language_vector_to_model(ner_model_checkpoint: str, language_vector:TaskVector, lambda_coef: float):
    ner_model = language_vector.apply_to(ner_model_checkpoint, scaling_coef=lambda_coef)
    return ner_model

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

        #TODO P, R, F1
def lambda_search(eval_set, coef):
    ...

def test_lang_ner(ner_model, language_model, pretrained_checkpoint, language_dataset, lambdas: List[float]= [0.0, 0.25, 0.5, 0.75, 1.0], batch_size:int=32):
    lv = get_language_vector(pretrained_checkpoint, language_model)
    best_lambda = 1.0
    ner = apply_language_vector_to_model(ner_model, lv, best_lambda) # TODO find best lambda
    NER_dataset = load_dataset("MultiCoNER/multiconer_v2", language_dataset, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(pretrained_checkpoint)
    def tokenize_dataset(examples):
        tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)
        return tokenized_inputs
    tokenized_dataset= NER_dataset.map(
        tokenize_dataset,
        batched=True,
    )
    def batches(it):
        for i in range(0, len(it), batch_size):
            yield it[i:i+batch_size]

    ner.eval()
    with torch.no_grad():
        for batch_sent in tqdm(batches(tokenized_dataset), total=math.ceil(len(tokenized_dataset)/batch_size), desc="Eval"):
            predictions = ner(**batch_sent["input_ids"])
            labels = batch_sent["labels"]
