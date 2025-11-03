import torch
from torch.utils.data import DataLoader
from typing import List
from datasets import load_dataset
from transformers import AutoModel,AutoModelForSequenceClassification , AutoModelForMaskedLM, AutoTokenizer, DataCollatorWithPadding, AutoConfig, DataCollatorForTokenClassification, BertForSequenceClassification
from tqdm import tqdm
from task_vectors import TaskVector
from safetensors.torch import load_model
import numpy as np
def get_language_vector(base_model: str, saved_language: str):
    lang_vector = TaskVector(pretrained_model=AutoModelForMaskedLM.from_pretrained(base_model),
                             finetuned_model=AutoModelForMaskedLM.from_pretrained(saved_language, local_files_only=True))
    return lang_vector

def apply_language_vector_to_model(nli_model_checkpoint: str, language_vector:TaskVector, lambda_coef: float):
    nli_model = language_vector.apply_to(nli_model_checkpoint, scaling_coef=lambda_coef)
    return nli_model

def lambda_search(eval_set, coef):
    ...

def get_label_mapping():
    NLI_dataset = load_dataset("facebook/xnli", "en", trust_remote_code=True)
    unique_tags = sorted({str(ex["label"]) for ex in NLI_dataset["train"]})
    id2label = {idx: lab for idx, lab in enumerate(unique_tags)}
    label2id = {v: k for k, v in id2label.items()}
    print(unique_tags, label2id, id2label)
    return id2label, label2id

def test_lang_nli(nli, language_model, pretrained_checkpoint, language_dataset, lambdas: List[float]= [0.0, 0.25, 0.5, 0.75, 1.0], batch_size:int=8):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("lang dataset ", language_dataset)
    if language_dataset != "en":
        print("getting lang vector and applying to nli model")
        lv = get_language_vector(pretrained_checkpoint, language_model)
        best_lambda = 1.0
        nli = apply_language_vector_to_model(nli, lv, best_lambda) # TODO find best lambda:
    else:
        print("english -- don't need to apply a language vector")
    NLI_dataset = load_dataset("facebook/xnli", language_dataset, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(pretrained_checkpoint)
    def preprocess(examples):
        inputs = [examples[column] for column in ["premise", "hypothesis"]]
        tokenized_examples = tokenizer(
            *inputs,
            padding="max_length",          # **important** – matches training
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        tokenized_examples['label'] = examples["label"]
        return tokenized_examples
    tokenized_dataset= NLI_dataset.map(
        preprocess,
        batched=True,
        remove_columns=NLI_dataset["test"].column_names
    )
    preds = []
    labels = []
    nli.to(device)
    test = tokenized_dataset["test"]
    #test.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    collator = DataCollatorWithPadding(tokenizer=tokenizer, padding="max_length", max_length=512)
    test_dataloader = DataLoader(test, batch_size=batch_size, collate_fn=collator, shuffle=False)
    with torch.no_grad():
        for batch in tqdm(test_dataloader):
            inputs = {k: v.to(device) for k, v in batch.items() if k != "label"}
            logits = model(**inputs).logits
            predictions = torch.argmax(logits, dim=-1).cpu().numpy()
            preds.extend(predictions)
            labels.extend(batch["labels"].cpu().numpy())
    total = len(preds)
    correct = sum(1 for pred, lab in zip(preds,labels) if pred == lab)
    accuracy = correct / total if total > 0 else 0
    nli.to("cpu")
    return accuracy

if __name__ == "__main__":
    bert = True
    if bert:
        base_model = "google-bert/bert-base-multilingual-uncased"
        prefix = "bert-multilingual"
    else:
        base_model = "FacebookAI/xlm-roberta-base"
        prefix = "xlm-roberta"
    datasets = ["en", "es", "hi", "de", "zh"]
    language_models = [f"{prefix}/language_en_done", 
                       f"{prefix}/language_es_done", 
                       f"{prefix}/language_hi_done", 
                       f"{prefix}/language_de_done", 
                       f"{prefix}/language_zh_done"]
    print(f"NLI_en")
    model = AutoModelForSequenceClassification.from_pretrained(
        f"{prefix}/NLI_en",
        device_map="auto",               # puts the model on GPU if available
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model)

    model.eval()                         # disables dropout / batch‑norm
    torch.set_grad_enabled(False)        # extra safety guard
    #load_model(model, f"{prefix}/NLI_en/model.safetensors", device="cpu")
    with open("output/NLI_1.0.txt", "w") as f:
        for idx, lang_model in enumerate(language_models):
            print("language model", datasets[idx])
            accuracy= test_lang_nli(model, lang_model, base_model, datasets[idx])
            print(f"accuracy: {accuracy}")  
            f.write(f"\n======language: {lang_model.split('_')[1]}=======\n")
            f.write(f"accuracy: {accuracy}\n")
            
        f.close()




