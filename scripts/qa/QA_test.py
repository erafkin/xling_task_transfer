import json
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoModelForQuestionAnswering, AutoModelForMaskedLM, AutoTokenizer, DefaultDataCollator, AutoConfig, DataCollatorForTokenClassification, BertForSequenceClassification, AutoModelForCausalLM
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

def apply_language_vector_to_model(qa_model_checkpoint: str, language_vector:TaskVector, lambda_coef: float):
    qa_model = language_vector.apply_to(qa_model_checkpoint, scaling_coef=lambda_coef)
    return qa_model
    

def test_lang_qa(qa, language_model, pretrained_checkpoint, dataset, best_lambda:float=1.0, batch_size:int=32):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lv = get_language_vector(pretrained_checkpoint, language_model)
    qa = apply_language_vector_to_model(qa, lv, best_lambda) 
    preds = []
    labels = []
    qa.to(device).eval()

    collator = DefaultDataCollator()
    test_dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collator, shuffle=False)
    with torch.no_grad():
        for batch in tqdm(test_dataloader):
            inputs = {k: v.to(device) for k, v in batch.items() if k != "label"}
            logits = qa(**inputs).logits
            predictions = torch.argmax(logits, dim=-1).cpu().numpy()
            preds.extend(predictions)
            labels.extend(batch["labels"].cpu().numpy())
    total = len(preds)
    correct = sum(1 for pred, lab in zip(preds,labels) if pred == lab)
    accuracy = correct / total if total > 0 else 0
    qa.to("cpu")
    del qa
    gc.collect()
    return accuracy




def test_lang_qa_causal(qa, language_model, pretrained_checkpoint, dataset, best_lambda:float=1.0, batch_size:int=8):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(pretrained_checkpoint, trust_remote_code=True, padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token
    lv = get_language_vector_causal(pretrained_checkpoint, language_model)

    qa = apply_language_vector_to_model(qa, lv, best_lambda)

    preds = []
    labels = []
    qa.to(device).eval()
    with torch.no_grad():
        for data in tqdm(dataset):
            prompt = f"Context: {data['context']}\n. Question: {data['question']}\n Answer:"
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            output_ids = qa.generate(
                **inputs,
                max_new_tokens=10,
                do_sample=False
            )
            text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
            pred_tag = text.split("Answer:")[-1].strip()
            preds.append(pred_tag)
            labels.append(data['label'])
    print("PREDICTIONS: ", preds[:5])
    print("LABELS: ", labels[:5])
    total = len(preds)
    correct = sum(1 for pred, lab in zip(preds,labels) if pred == str(lab))
    accuracy = correct / total if total > 0 else 0
    print("accuracy: ", accuracy)
    qa.to("cpu")
    del qa
    gc.collect()
    return accuracy

if __name__ == "__main__":
    test_lambdas = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    model_base = "base_finetuned"
    base_models = ["bert", "roberta", "qwen", "granite"]
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
            elif base_model_str == "granite":
                base_model = "ibm-granite/granite-4.0-350m"
                prefix = "granite"
            else:
                base_model = "Qwen/Qwen3-0.6B"
                prefix = "qwen"
            # handle data
            QA_dataset = load_dataset("google/xquad", f"xquad.{datasets[idx]}", trust_remote_code=True)
            QA_dataset = QA_dataset["validation"] # the only split that exists is validation, split into 2
            QA_dataset = QA_dataset.train_test_split(test_size=0.1)

            tokenizer = AutoTokenizer.from_pretrained(base_model)

            if base_model_str == "qwen" or base_model_str == "granite":
                hyperparameter_results = {}
                qa =  AutoModelForCausalLM.from_pretrained(f"{prefix}/{model_base}/QA_en")
                if base_model_str == "granite":
                    qa.config.use_cache = False
                for l in test_lambdas:
                    print("lambda: ", l)
                    accuracy = test_lang_qa_causal(qa, f"{prefix}/{model}", base_model, QA_dataset["test"], l)
                    hyperparameter_results[l] = accuracy
                print("hyperparamter search results")
                print(hyperparameter_results)
                overall_hyperparameter_results[model][base_model_str] = hyperparameter_results
                
                best_lambda = max(hyperparameter_results, key=hyperparameter_results.get)
                if model.split("_")[1] == "en":
                    print("lang en, best lambda 0")
                    best_lambda = 0.0
                print(best_lambda)
                qa =  AutoModelForCausalLM.from_pretrained(f"{prefix}/{model_base}/QA_en")
                with open(f"output/{prefix}/{model_base}/QA.txt", "a") as f:
                    print("language model", model)
                    accuracy= test_lang_qa_causal(qa, f"{prefix}/{model}", base_model, QA_dataset["train"], best_lambda)
                    print(f"accuracy: {accuracy}")  
                    f.write(f"\n======language: {model.split('_')[1]}=======\n")
                    f.write(f"best lambda: {best_lambda}\n")
                    f.write(f"accuracy: {accuracy}\n")
                    f.close()

            else:
                def preprocess(examples):
                    questions = [q.strip() for q in examples["question"]]
                    inputs = tokenizer(
                        questions,
                        examples["context"],
                        max_length=384,
                        truncation="only_second",
                        return_offsets_mapping=True,
                        padding="max_length",
                    )

                    offset_mapping = inputs.pop("offset_mapping")
                    answers = examples["answers"]
                    start_positions = []
                    end_positions = []

                    for i, offset in enumerate(offset_mapping):
                        answer = answers[i]
                        start_char = answer["answer_start"][0]
                        end_char = answer["answer_start"][0] + len(answer["text"][0])
                        sequence_ids = inputs.sequence_ids(i)

                        # Find the start and end of the context
                        idx = 0
                        while sequence_ids[idx] != 1:
                            idx += 1
                        context_start = idx
                        while sequence_ids[idx] == 1:
                            idx += 1
                        context_end = idx - 1

                        # If the answer is not fully inside the context, label it (0, 0)
                        if offset[context_start][0] > end_char or offset[context_end][1] < start_char:
                            start_positions.append(0)
                            end_positions.append(0)
                        else:
                            # Otherwise it's the start and end token positions
                            idx = context_start
                            while idx <= context_end and offset[idx][0] <= start_char:
                                idx += 1
                            start_positions.append(idx - 1)

                            idx = context_end
                            while idx >= context_start and offset[idx][1] >= end_char:
                                idx -= 1
                            end_positions.append(idx + 1)

                    inputs["start_positions"] = start_positions
                    inputs["end_positions"] = end_positions
                    return inputs
                tokenized_dataset= QA_dataset.map(
                    preprocess,
                    batched=True,
                    remove_columns=QA_dataset["test"].column_names
                )

                hyperparameter_results = {}
                batch_size = 32
                torch.set_grad_enabled(False)
                for l in test_lambdas:
                    qa = AutoModelForQuestionAnswering.from_pretrained(
                        f"{prefix}/{model_base}/QA_en",
                        local_files_only=True,
                    )
                    qa.eval()
                    accuracy = test_lang_qa(qa, f"{prefix}/{model}", base_model, tokenized_dataset["test"], l, batch_size)
                    hyperparameter_results[l] = accuracy
                print("hyperparamter serach results")
                print(hyperparameter_results)
                overall_hyperparameter_results[model][base_model_str] = hyperparameter_results
                best_lambda = max(hyperparameter_results, key=hyperparameter_results.get)
                if model.split("_")[1] == "en":
                    print("lang en, best lambda 0")
                    best_lambda = 0.0
                print(best_lambda)
                with open(f"output/{prefix}/{model_base}/QA.txt", "a") as f:
                    print("language model", model)
                    qa = AutoModelForQuestionAnswering.from_pretrained(
                        f"{prefix}/{model_base}/QA_en",
                        local_files_only=True
                    )

                    qa.eval()                         # disables dropout / batch‑norm  
                    accuracy= test_lang_qa(qa, f"{prefix}/{model}", base_model, tokenized_dataset["train"], best_lambda, batch_size)
                    print(f"accuracy: {accuracy}")  
                    f.write(f"\n======language: {model.split('_')[1]}=======\n")
                    f.write(f"best lambda: {best_lambda}\n")
                    f.write(f"accuracy: {accuracy}\n")
                    f.close()
    with open(f"output/QA_pretrained_hyperparameter_search.json", "w") as f:
            json.dump(overall_hyperparameter_results, f, indent=4)
            f.close()





