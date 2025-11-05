import torch
from torch.utils.data import DataLoader
from typing import List
from datasets import Dataset, DatasetDict
from transformers import AutoModelForMaskedLM, AutoTokenizer, DataCollatorForTokenClassification
from tqdm import tqdm
from safetensors.torch import load_model
from scripts.task_vectors import TaskVector
from scripts.task_utils import TokenClassificationHead, load_conllu_data
import gc

def get_language_vector(base_model: str, saved_language: str):
    lang_vector = TaskVector(pretrained_model=AutoModelForMaskedLM.from_pretrained(base_model),
                             finetuned_model=AutoModelForMaskedLM.from_pretrained(saved_language, local_files_only=True))
    return lang_vector

def apply_language_vector_to_model(pos_model_checkpoint: str, language_vector:TaskVector, lambda_coef: float):
    pos_model = language_vector.apply_to(pos_model_checkpoint, scaling_coef=lambda_coef)
    return pos_model

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
    # Simple accuracy calculation
    total = sum(len(pred) for pred in y_pred)
    correct = sum(1 for pred, lab in zip(y_pred, y_true) for p, l in zip(pred, lab) if p == l)
    accuracy = correct / total if total > 0 else 0
    return accuracy

def get_label_mapping():
    pos_dataset = load_conllu_data("GUM_en/en_gum-ud-train.conllu")
    pos_dataset = Dataset.from_pandas(pos_dataset)
    unique_tags = sorted({tag for ex in pos_dataset for tag in ex["pos_tags"]})
    label2id = {tag: i for i, tag in enumerate(unique_tags)}
    id2label = {i: tag for tag, i in label2id.items()}
    return id2label, label2id
    

def test_lang_pos(pos, language_model, pretrained_checkpoint, dataset, best_lambda:float=1.0, batch_size:int=32):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lv = get_language_vector(pretrained_checkpoint, language_model)
    pos = apply_language_vector_to_model(pos, lv, best_lambda)
    preds = []
    labels = []
    pos.to(device).eval()
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
    test_dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collator)
    with torch.no_grad():
        for batch in tqdm(test_dataloader):
            inputs = {k: v.to(device) for k, v in batch.items() if k != "label"}
            logits = pos(**inputs)["logits"]
            predictions = torch.argmax(logits, dim=-1).cpu().numpy()
            preds.extend(predictions)
            labels.extend(batch["labels"].cpu().numpy())
    accuracy = compute_metrics(preds, labels)
    pos.to("cpu")
    del pos
    gc.collect()
    return accuracy

if __name__ == "__main__":
    test_lambdas = [0.0, 0.1, 0.2, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9, 1.0]
    model_bases = ["lang_en_finetuned", "base_finetuned"]
    bert_values = [True, False]
    
    datasets = ["English (EN)", "Spanish (ES)", "Hindi (HI)", "German (DE)", "Chinese (ZH)"]
    id2label, label2id = get_label_mapping()
    val_datasets = [
        "GUM_en/en_gum-ud-dev.conllu", 
        "val_datasets_pos/es_gsd-ud-dev.conllu", 
        "val_datasets_pos/hi_hdtb-ud-dev.conllu",
        "val_datasets_pos/de_gsd-ud-dev.conllu",
        "val_datasets_pos/zh_gsd-ud-dev.conllu"
    ]
    test_datasets = [
        "UD_English-PUD/en_pud-ud-test.conllu",
        "UD_Spanish-PUD/es_pud-ud-test.conllu",
        "UD_Hindi-PUD/hi_pud-ud-test.conllu",
        "UD_German-PUD/de_pud-ud-test.conllu", 
        "UD_Chinese-PUD/zh_pud-ud-test.conllu"
        ]
    language_models = ["language_en_done", 
                    "language_es_done", 
                    "language_hi_done", 
                    "language_de_done", 
                    "language_zh_done"]
    for idx, model in enumerate(language_models):
        for bert in bert_values:
            for model_base in model_bases:
                if bert:
                    base_model = "google-bert/bert-base-multilingual-uncased"
                    prefix = "bert-multilingual"
                else:
                    base_model = "FacebookAI/xlm-roberta-base"
                    prefix = "xlm-roberta"
                # handle data
                val_dataset = load_conllu_data(val_datasets[idx])
                val_dataset = Dataset.from_pandas(val_dataset)
                test_dataset = load_conllu_data(test_datasets[idx])
                test_dataset = Dataset.from_pandas(test_dataset)
                POS_dataset = DatasetDict({
                    "validation": val_dataset,
                    "test": test_dataset
                })
                tokenizer = AutoTokenizer.from_pretrained(base_model)
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
                tokenized_dataset= POS_dataset.map(
                    tokenize_and_align_labels,
                    batched=True,
                )

                mlm_model = AutoModelForMaskedLM.from_pretrained(
                    base_model,
                    dtype=torch.float32,
                )
                if bert:
                    encoder = mlm_model.bert
                else:
                    encoder = mlm_model.roberta
                hyperparameter_results = {}
                for l in test_lambdas:
                    pos_model = TokenClassificationHead(encoder, num_labels=len(id2label), bert=bert)
                    load_model(pos_model, f"{prefix}/{model_base}/POS_en/model.safetensors", device="cpu")
                    accuracy = test_lang_pos(pos_model, f"{prefix}/{model}", base_model, tokenized_dataset["validation"], l)
                    hyperparameter_results[l] = accuracy
                print("hyperparamter serach results")
                print(hyperparameter_results)
                best_lambda = max(hyperparameter_results, key=hyperparameter_results.get)
                print(best_lambda)
                with open(f"output/{prefix}/{model_base}/POS.txt", "a") as f:
                    print("language model", model)
                    pos_model = TokenClassificationHead(encoder, num_labels=len(id2label), bert=bert)
                    load_model(pos_model, f"{prefix}/{model_base}/POS_en/model.safetensors", device="cpu")
                    accuracy= test_lang_pos(pos_model, f"{prefix}/{model}", base_model, tokenized_dataset["test"], best_lambda)
                    print(f"accuracy: {accuracy}")  
                    f.write(f"\n======language: {model.split('_')[1]}=======\n")
                    f.write(f"best lambda: {best_lambda}\n")
                    f.write(f"accuracy: {accuracy}\n")
                    f.close()





