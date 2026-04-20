import json
import torch
from transformers import default_data_collator
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoModelForQuestionAnswering, AutoModelForMaskedLM, AutoTokenizer, DefaultDataCollator, AutoConfig, DataCollatorForTokenClassification, BertForSequenceClassification, AutoModelForCausalLM
from tqdm import tqdm
from scripts.task_vectors import TaskVector
import gc
import evaluate
import collections
import numpy as np

metric = evaluate.load("squad")

def compute_metrics(start_logits, end_logits, features, examples):
    # from https://huggingface.co/learn/llm-course/en/chapter7/7#training-loop
    max_answer_length = 30
    n_best = 20

    example_to_features = collections.defaultdict(list)
    for idx, feature in enumerate(features):
        example_to_features[feature["example_id"]].append(idx)

    predicted_answers = []
    for example in tqdm(examples):
        example_id = example["id"]
        context = example["context"]
        answers = []

        # Loop through all features associated with that example
        for feature_index in example_to_features[example_id]:
            start_logit = start_logits[feature_index]
            end_logit = end_logits[feature_index]
            offsets = features[feature_index]["offset_mapping"]

            start_indexes = np.argsort(start_logit)[-1 : -n_best - 1 : -1].tolist()
            end_indexes = np.argsort(end_logit)[-1 : -n_best - 1 : -1].tolist()
            for start_index in start_indexes:
                for end_index in end_indexes:
                    # Skip answers that are not fully in the context
                    if offsets[start_index] is None or offsets[end_index] is None:
                        continue
                    # Skip answers with a length that is either < 0 or > max_answer_length
                    if (
                        end_index < start_index
                        or end_index - start_index + 1 > max_answer_length
                    ):
                        continue

                    answer = {
                        "text": context[offsets[start_index][0] : offsets[end_index][1]],
                        "logit_score": start_logit[start_index] + end_logit[end_index],
                    }
                    answers.append(answer)

        # Select the answer with the best score
        if len(answers) > 0:
            best_answer = max(answers, key=lambda x: x["logit_score"])
            predicted_answers.append(
                {"id": example_id, "prediction_text": best_answer["text"]}
            )
        else:
            predicted_answers.append({"id": example_id, "prediction_text": ""})

    theoretical_answers = [{"id": ex["id"], "answers": ex["answers"]} for ex in examples]
    return metric.compute(predictions=predicted_answers, references=theoretical_answers)

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
    

def test_lang_qa(qa, language_model, pretrained_checkpoint, dataset,raw_dataset, best_lambda:float=1.0, batch_size:int=32):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = dataset.remove_columns(["example_id", "offset_mapping"])
    dataset.set_format("torch")

    lv = get_language_vector(pretrained_checkpoint, language_model)
    qa = apply_language_vector_to_model(qa, lv, best_lambda) 
    start_logits = []
    end_logits = []
    qa.to(device).eval()

    test_dataloader = DataLoader(ds, batch_size=batch_size, collate_fn=default_data_collator, shuffle=False)
    
    with torch.no_grad():
        for batch in tqdm(test_dataloader):
            inputs = {k: v.to(device) for k, v in batch.items() if k != "label"}
            
            outputs = qa(**inputs)
            start_logits.append(outputs.start_logits.cpu().numpy())
            end_logits.append(outputs.end_logits.cpu().numpy())
    start_logits = np.concatenate(start_logits)
    end_logits = np.concatenate(end_logits)
    start_logits = start_logits[: len(dataset)]
    end_logits = end_logits[: len(dataset)]

    metrics = compute_metrics(
        start_logits, end_logits, dataset, raw_dataset
    )
    print(metrics)
    qa.to("cpu")
    del qa
    gc.collect()
    return metrics["exact_match"]




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
                max_new_tokens=100,
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
    base_models = ["bert", "roberta"]#, "qwen", "granite"]
    base_models = ["qwen", "granite"]
    datasets = ["en", "es", "hi", "de", "zh", "fr", "ru",] # THERE IS NO FRENCH FOR XQUAD
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
            if datasets[idx] == "fr":
                QA_dataset = load_dataset("lincoln/newsquadfr", trust_remote_code=True)
                QA_dataset = QA_dataset["train"] # train is the main split.
                QA_dataset = QA_dataset.train_test_split(test_size=0.1)
            else:
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
                    max_length = 384
                    stride = 128
                    questions = [q.strip() for q in examples["question"]]
                    inputs = tokenizer(
                        questions,
                        examples["context"],
                        max_length=max_length,
                        truncation="only_second",
                        stride=stride,
                        return_overflowing_tokens=True,
                        return_offsets_mapping=True,
                        padding="max_length",
                    )

                    sample_map = inputs.pop("overflow_to_sample_mapping")
                    example_ids = []

                    for i in range(len(inputs["input_ids"])):
                        sample_idx = sample_map[i]
                        example_ids.append(examples["id"][sample_idx])

                        sequence_ids = inputs.sequence_ids(i)
                        offset = inputs["offset_mapping"][i]
                        inputs["offset_mapping"][i] = [
                            o if sequence_ids[k] == 1 else None for k, o in enumerate(offset)
                        ]

                    inputs["example_id"] = example_ids
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
                    accuracy = test_lang_qa(qa, f"{prefix}/{model}", base_model, tokenized_dataset["test"], QA_dataset["test"],l, batch_size)
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
                    accuracy= test_lang_qa(qa, f"{prefix}/{model}", base_model, tokenized_dataset["train"], QA_dataset["train"], best_lambda, batch_size)
                    print(f"accuracy: {accuracy}")  
                    f.write(f"\n======language: {model.split('_')[1]}=======\n")
                    f.write(f"best lambda: {best_lambda}\n")
                    f.write(f"accuracy: {accuracy}\n")
                    f.close()
    with open(f"output/QA_pretrained_hyperparameter_search.json", "w") as f:
            json.dump(overall_hyperparameter_results, f, indent=4)
            f.close()





