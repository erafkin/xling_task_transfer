import torch
from torch.utils.data import DataLoader
from typing import List
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoModelForMaskedLM, AutoTokenizer, DataCollatorWithPadding, AutoConfig, DataCollatorForTokenClassification
from tqdm import tqdm
from safetensors.torch import load_model
from task_vectors import TaskVector
from finetune_tasks import TokenClassificationHead

def get_language_vector(base_model: str, saved_language: str):
    lang_vector = TaskVector(pretrained_model=AutoModelForMaskedLM.from_pretrained(base_model),
                             finetuned_model=AutoModelForMaskedLM.from_pretrained(saved_language, local_files_only=True))
    return lang_vector

def apply_language_vector_to_model(nli_model_checkpoint: str, language_vector:TaskVector, lambda_coef: float):
    nli_model = language_vector.apply_to(nli_model_checkpoint, scaling_coef=lambda_coef)
    return nli_model

def compute_metrics(predictions, labels):
    # Simple accuracy calculation
    total = sum(len(pred) for pred in predictions)
    correct = sum(1 for pred, lab in zip(predictions,labels) for p, l in zip(pred, lab) if p == l)
    accuracy = correct / total if total > 0 else 0
    return accuracy

def lambda_search(eval_set, coef):
    ...

def get_label_mapping():
    NLI_dataset = load_dataset("facebook/xnli", "en", trust_remote_code=True)
    unique_tags = sorted({ex["label"] for ex in NLI_dataset["train"]})
    label2id = {tag: i for i, tag in enumerate(unique_tags)}
    id2label = {i: tag for tag, i in label2id.items()}
    return id2label, label2id

def test_lang_nli(nli, language_model, pretrained_checkpoint, language_dataset, label2id, lambdas: List[float]= [0.0, 0.25, 0.5, 0.75, 1.0], batch_size:int=32, ):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if language_dataset != "en":
        lv = get_language_vector(pretrained_checkpoint, language_model)
        best_lambda = 0.0
        nli = apply_language_vector_to_model(nli, lv, best_lambda) # TODO find best lambda:
    NLI_dataset = load_dataset("facebook/xnli", language_dataset, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(pretrained_checkpoint)
    def preprocess(examples):
        inputs = [examples[column] for column in ["premise", "hypothesis"]]
        tokenized_examples = tokenizer(
            *inputs,
            padding=True,
            max_length=512,
            truncation=True,
        )
        tokenized_examples['label'] = [
            label2id[label]
            for label in examples['label']
        ]
        return tokenized_examples
    tokenized_dataset= NLI_dataset.map(
        preprocess,
        batched=True,
    )
    preds = []
    labels = []
    nli.to(device).eval()
    test = tokenized_dataset["test"]
    test.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    collator = DataCollatorWithPadding(tokenizer=tokenizer)
    test_dataloader = DataLoader(test, batch_size=batch_size, collate_fn=collator)
    with torch.no_grad():
        for batch in tqdm(test_dataloader):
            input_ids = batch["input_ids"].to(device)
            ps = nli(input_ids)["logits"]
            ps = torch.argmax(ps, dim=-1).cpu().tolist()
            ls = batch["labels"].cpu().tolist()
            preds += ps
            labels += ls
    accuracy = compute_metrics(preds, labels)
    nli.to("cpu")
    return accuracy

if __name__ == "__main__":
    datasets = ["en", "es", "hi", "de", "zh"]
    language_models = ["bert-multilingual/language_en_done", 
                       "bert-multilingual/language_es_done", 
                       "bert-multilingual/language_hi_done", 
                       "bert-multilingual/language_de_done", 
                       "bert-multilingual/language_zh_done"]
    id2label, label2id = get_label_mapping()
    model = AutoModelForSequenceClassification.from_pretrained("NLI_en")
    with open("output/NLI_0.0.txt", "w") as f:
        for idx, lang_model in enumerate(language_models):
            print("language model", model)
            accuracy= test_lang_nli(model, lang_model, "bert-multilingual/language_en_done", datasets[idx], label2id)
            print(f"accuracy: {accuracy}")  
            f.write(f"\n======language: {lang_model.split('_')[1]}=======\n")
            f.write(f"accuracy: {accuracy}\n")
            
        f.close()





