import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoModelForMaskedLM, AutoTokenizer, DataCollatorWithPadding, AutoConfig, DataCollatorForTokenClassification, BertForSequenceClassification
from tqdm import tqdm
from task_vectors import TaskVector
import gc

def get_language_vector(base_model: str, saved_language: str):
    lang_vector = TaskVector(pretrained_model=AutoModelForMaskedLM.from_pretrained(base_model),
                             finetuned_model=AutoModelForMaskedLM.from_pretrained(saved_language, local_files_only=True))
    return lang_vector

def apply_language_vector_to_model(nli_model_checkpoint: str, language_vector:TaskVector, lambda_coef: float):
    nli_model = language_vector.apply_to(nli_model_checkpoint, scaling_coef=lambda_coef)
    return nli_model
    

def test_lang_nli(nli, language_model, pretrained_checkpoint, dataset, best_lambda:float=1.0, batch_size:int=32):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lv = get_language_vector(pretrained_checkpoint, language_model)
    nli = apply_language_vector_to_model(nli, lv, best_lambda) 
    preds = []
    labels = []
    nli.to(device).eval()
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    collator = DataCollatorWithPadding(tokenizer=tokenizer, padding="max_length", max_length=512)
    test_dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collator, shuffle=False)
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
    del nli
    gc.collet()
    return accuracy

if __name__ == "__main__":
    test_lambdas = [0.0, 0.1, 0.2, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9, 1.0]
    model_bases = ["lang_en_finetuned", "base_finetuned"]
    bert_values = [True, False]
    
    datasets = ["en", "es", "hi", "de", "zh"]
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
                NLI_dataset = load_dataset("facebook/xnli", datasets[idx], trust_remote_code=True)
                tokenizer = AutoTokenizer.from_pretrained(base_model)
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

                hyperparameter_results = {}
                for l in test_lambdas:
                    model = AutoModelForSequenceClassification.from_pretrained(
                        f"{prefix}/{model_base}/NLI_en",
                        device_map="auto",               # puts the model on GPU if available
                    )
                    model.eval()  
                    accuracy = test_lang_nli(model, f"{prefix}/{model}", base_model, tokenized_dataset["validation"], l)
                    hyperparameter_results[l] = accuracy
                print("hyperparamter serach results")
                print(hyperparameter_results)

                best_lambda = max(hyperparameter_results, key=hyperparameter_results.get)
                print(best_lambda)
                with open(f"output/{prefix}/{model_base}/NLI.txt", "a") as f:
                    print("language model", model)
                    model = AutoModelForSequenceClassification.from_pretrained(
                        f"{prefix}/NLI_en",
                        device_map="auto",               # puts the model on GPU if available
                    )

                    model.eval()                         # disables dropout / batch‑norm
                    torch.set_grad_enabled(False)  
                    accuracy= test_lang_nli(model, f"{prefix}/{model}", base_model, tokenized_dataset["test"], best_lambda)
                    print(f"accuracy: {accuracy}")  
                    f.write(f"\n======language: {model.split('_')[1]}=======\n")
                    f.write(f"best lambda: {best_lambda}\n")
                    f.write(f"accuracy: {accuracy}\n")
                    f.close()





