import json
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoModelForMaskedLM, AutoTokenizer, DataCollatorWithPadding, AutoConfig, DataCollatorForTokenClassification, BertForSequenceClassification, AutoModelForCausalLM
from tqdm import tqdm
from scripts.task_vectors import TaskVector
import gc

def get_language_vector(base_model: str, saved_language: str):
    lang_vector = TaskVector(pretrained_model=AutoModelForMaskedLM.from_pretrained(base_model),
                             finetuned_model=AutoModelForMaskedLM.from_pretrained(saved_language, local_files_only=True))
    return lang_vector

def get_language_vector_causal(base_model: str, saved_language: str):
    lang_vector = TaskVector(pretrained_model=AutoModelForCausalLM.from_pretrained(base_model),
                             finetuned_model=AutoModelForCausalLM.from_pretrained(saved_language, local_files_only=True))
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
    #dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

    collator = DataCollatorWithPadding(tokenizer=tokenizer, padding="max_length", max_length=512)
    test_dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collator, shuffle=False)
    with torch.no_grad():
        for batch in tqdm(test_dataloader):
            inputs = {k: v.to(device) for k, v in batch.items() if k != "label"}
            logits = nli(**inputs).logits
            predictions = torch.argmax(logits, dim=-1).cpu().numpy()
            preds.extend(predictions)
            labels.extend(batch["labels"].cpu().numpy())
    total = len(preds)
    correct = sum(1 for pred, lab in zip(preds,labels) if pred == lab)
    accuracy = correct / total if total > 0 else 0
    nli.to("cpu")
    del nli
    gc.collect()
    return accuracy




def test_lang_nli_causal(nli, language_model, pretrained_checkpoint, dataset, best_lambda:float=1.0, batch_size:int=8):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(pretrained_checkpoint, trust_remote_code=True, padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token
    lv = get_language_vector_causal(pretrained_checkpoint, language_model)

    nli = apply_language_vector_to_model(nli, lv, best_lambda)

    preds = []
    labels = []
    nli.to(device).eval()
    with torch.no_grad():
        for data in tqdm(dataset):
            prompt = f"Sentences: {' | '.join([data['premise'], data['hypothesis']])}.\n NLI:"
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            output_ids = nli.generate(
                **inputs,
                max_new_tokens=10,
                do_sample=False
            )
            text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
            pred_tag = text.split("NLI:")[-1].strip()
            preds.append(pred_tag)
            labels.append(data['label'])
    print("PREDICTIONS: ", preds[:5])
    print("LABELS: ", labels[:5])
    total = len(preds)
    correct = sum(1 for pred, lab in zip(preds,labels) if pred == str(lab))
    accuracy = correct / total if total > 0 else 0
    print("accuracy: ", accuracy)
    nli.to("cpu")
    del nli
    gc.collect()
    return accuracy

if __name__ == "__main__":
    test_lambdas = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    model_base = "base_finetuned"
    base_models = ["bert", "roberta", "qwen"]
    datasets = ["en", "es", "hi", "de", "zh", "fr","ru"]
    language_models = ["language_en_done", 
                    "language_es_done", 
                    "language_hi_done", 
                    "language_de_done", 
                    "language_zh_done", 
                    "language_fr_done",
                    "language_ru_done"
                    ]
    overall_hyperparameter_results = {}
    torch.set_grad_enabled(False)  
    for idx, model in enumerate(language_models):
        overall_hyperparameter_results[model] = {}
        for base_model_str in base_models:
            overall_hyperparameter_results[base_model_str] = {}
            if base_model_str == "bert":
                base_model = "google-bert/bert-base-multilingual-cased"
                prefix = "bert-multilingual"
            elif base_model_str == "roberta":
                base_model = "FacebookAI/xlm-roberta-base"
                prefix = "xlm-roberta"
            else:
                base_model = "Qwen/Qwen3-0.6B"
                prefix = "qwen"
            # handle data
            NLI_dataset = load_dataset("facebook/xnli", datasets[idx], trust_remote_code=True)
            tokenizer = AutoTokenizer.from_pretrained(base_model)

            if base_model_str == "qwen":
                hyperparameter_results = {}
                nli =  AutoModelForCausalLM.from_pretrained(f"{prefix}/{model_base}/NLI_en")
                for l in test_lambdas:
                    print("lambda: ", l)
                    accuracy = test_lang_nli_causal(nli, f"{prefix}/{model}", base_model, NLI_dataset["validation"].select(range(1000)), l)
                    hyperparameter_results[l] = accuracy
                print("hyperparamter search results")
                print(hyperparameter_results)
                overall_hyperparameter_results[model][base_model_str] = hyperparameter_results
                
                best_lambda = max(hyperparameter_results, key=hyperparameter_results.get)
                if model.split("_")[1] == "en":
                    print("lang en, best lambda 0")
                    best_lambda = 0.0
                print(best_lambda)
                nli =  AutoModelForCausalLM.from_pretrained(f"{prefix}/{model_base}/NLI_en")
                with open(f"output/{prefix}/{model_base}/NLI.txt", "a") as f:
                    print("language model", model)
                    accuracy= test_lang_nli_causal(nli, f"{prefix}/{model}", base_model, NLI_dataset["test"], best_lambda)
                    print(f"accuracy: {accuracy}")  
                    f.write(f"\n======language: {model.split('_')[1]}=======\n")
                    f.write(f"best lambda: {best_lambda}\n")
                    f.write(f"accuracy: {accuracy}\n")
                    f.close()

            else:
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
                batch_size = 32
                torch.set_grad_enabled(False)
                for l in test_lambdas:
                    nli = AutoModelForSequenceClassification.from_pretrained(
                        f"{prefix}/{model_base}/NLI_en",
                        local_files_only=True,
                    )
                    nli.eval()
                    accuracy = test_lang_nli(nli, f"{prefix}/{model}", base_model, tokenized_dataset["validation"], l, batch_size)
                    hyperparameter_results[l] = accuracy
                print("hyperparamter serach results")
                print(hyperparameter_results)
                overall_hyperparameter_results[model][base_model_str] = hyperparameter_results
                best_lambda = max(hyperparameter_results, key=hyperparameter_results.get)
                if model.split("_")[1] == "en":
                    print("lang en, best lambda 0")
                    best_lambda = 0.0
                print(best_lambda)
                with open(f"output/{prefix}/{model_base}/NLI.txt", "a") as f:
                    print("language model", model)
                    nli = AutoModelForSequenceClassification.from_pretrained(
                        f"{prefix}/{model_base}/NLI_en",
                        local_files_only=True
                    )

                    nli.eval()                         # disables dropout / batch‑norm  
                    accuracy= test_lang_nli(nli, f"{prefix}/{model}", base_model, tokenized_dataset["test"], best_lambda, batch_size)
                    print(f"accuracy: {accuracy}")  
                    f.write(f"\n======language: {model.split('_')[1]}=======\n")
                    f.write(f"best lambda: {best_lambda}\n")
                    f.write(f"accuracy: {accuracy}\n")
                    f.close()
    with open(f"output/NLI_pretrained_hyperparameter_search.json", "w") as f:
            json.dump(overall_hyperparameter_results, f, indent=4)
            f.close()





