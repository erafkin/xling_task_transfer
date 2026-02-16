import torch
from torch.utils.data import DataLoader
from typing import List
from datasets import Dataset, DatasetDict
from transformers import AutoModelForMaskedLM, AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
from safetensors.torch import load_model
from scripts.task_vectors import TaskVector
from scripts.task_utils import load_conllu_data
from scripts.dp.dp_model import TransformerForBiaffineParsing, DataCollatorForDependencyParsing
from scripts.dp.DP_train import UD_HEAD_LABELS
import gc
import json
import numpy as np

def get_language_vector(base_model: str, saved_language: str):
    lang_vector = TaskVector(pretrained_model=AutoModelForMaskedLM.from_pretrained(base_model),
                             finetuned_model=AutoModelForMaskedLM.from_pretrained(saved_language, local_files_only=True))
    return lang_vector

def get_language_vector_causal(base_model: str, saved_language: str):
    lang_vector = TaskVector(pretrained_model=AutoModelForCausalLM.from_pretrained(base_model),
                             finetuned_model=AutoModelForCausalLM.from_pretrained(saved_language, local_files_only=True))
    return lang_vector


def apply_language_vector_to_model(dp_model_checkpoint: str, language_vector:TaskVector, lambda_coef: float):
    dp_model = language_vector.apply_to(dp_model_checkpoint, scaling_coef=lambda_coef)
    return dp_model

def get_label_mapping():
    unique_tags = sorted(UD_HEAD_LABELS)
    label2id = {tag: i for i, tag in enumerate(unique_tags)}
    id2label = {i: tag for tag, i in label2id.items()}
    return id2label, label2id

def compute_metrics(eval_pred):
    # adapted from https://github.com/cambridgeltl/composable-sft/blob/main/examples/dependency-parsing/dp/utils_udp.py#L87
    
    predicted_head, predicted_arc, head_labels, arc_labels = eval_pred
    predicted_head = torch.Tensor(predicted_head)
    predicted_arc = torch.Tensor(predicted_arc)
    head_labels = torch.Tensor(head_labels)
    arc_labels = torch.Tensor(arc_labels)
    correct_indices = predicted_head.eq(head_labels)
    correct_labels = predicted_arc.eq(arc_labels)
    correct_labels_and_indices = correct_indices * correct_labels

    unlabeled_correct = correct_indices.sum().item()
    labeled_correct = correct_labels_and_indices.sum().item()
    total_words = correct_indices.numel()

    if total_words > 0.0:
        unlabeled_attachment_score = unlabeled_correct / total_words
        labeled_attachment_score = labeled_correct / total_words
    else:
        unlabeled_attachment_score = 0.0
        labeled_attachment_score = 0.0
    return {
        "uas": unlabeled_attachment_score * 100,
        "las": labeled_attachment_score * 100,
    }


def compute_metrics_causal(eval_pred):    
    predicted_head, predicted_arc, head_labels, arc_labels = eval_pred
    print("PREDICTIONS: ")
    print("HEADS:", predicted_head[:])
    print("ARCS:", predicted_arc[:2])
    print("LABELS: ")
    print("HEADS:", head_labels[:2])
    print("ARCS:", arc_labels[:2])
    correct_heads = []
    for pred, lab in zip(predicted_head, head_labels):
        correct_heads.append([1 if p == l else 0 for p, l in zip(pred, lab)])
    correct_rels = []
    for pred, lab in zip(predicted_arc, arc_labels):
        correct_rels.append([1 if p == l else 0  for p, l in zip(pred, lab)])
    correct_labels_and_indices = []
    for head, rel in zip(correct_heads, correct_rels):
        correct_labels_and_indices.append([1 if (h ==1 and r == 1) else 0 for h, r in zip(head, rel)]) 
    unlabeled_correct = sum([sum(c) for c in correct_heads])
    labeled_correct = sum([sum(c) for c in correct_labels_and_indices])
    total_words = sum(len(h) for h in predicted_head)

    if total_words > 0.0:
        unlabeled_attachment_score = unlabeled_correct / total_words
        labeled_attachment_score = labeled_correct / total_words
    else:
        unlabeled_attachment_score = 0.0
        labeled_attachment_score = 0.0
    print({
        "uas": unlabeled_attachment_score * 100,
        "las": labeled_attachment_score * 100,
    })
    return {
        "uas": unlabeled_attachment_score * 100,
        "las": labeled_attachment_score * 100,
    }

def test_lang_dp(dp, language_model, pretrained_checkpoint, dataset, best_lambda:float=1.0, batch_size:int=32):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lv = get_language_vector(pretrained_checkpoint, language_model)
    dp = apply_language_vector_to_model(dp, lv, best_lambda)
    dp.to(device).eval()
    if "token_type_ids" in dataset:
        dataset.set_format(type="torch", columns=["input_ids", 
                                              "attention_mask", 
                                              "token_type_ids", 
                                              "word_starts", 
                                              'labels_arcs', 
                                              'labels_rels'])
    else:
        dataset.set_format(type="torch", columns=["input_ids", 
                                              "attention_mask", 
                                              "word_starts", 
                                              'labels_arcs', 
                                              'labels_rels'])

    collator = DataCollatorForDependencyParsing(tokenizer=tokenizer)
    test_dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collator)
    preds_arcs = []
    preds_rels = []
    lab_arcs = []
    lab_rels = []
    with torch.no_grad():
        for batch in tqdm(test_dataloader):
            inputs = {k: v.to(device) for k, v in batch.items()}
            _, rel_preds, arc_preds = dp(**inputs)
            mask = inputs["labels_arcs"].ne(dp.pad_token_id)
            predictions_arcs = torch.argmax(arc_preds, dim=-1)[mask]

            labels_arcs = inputs["labels_arcs"][mask]

            predictions_rels, labels_rels = rel_preds[mask], inputs["labels_rels"][mask]
            predictions_rels = predictions_rels[torch.arange(len(labels_arcs)), labels_arcs]
            predictions_rels = torch.argmax(predictions_rels, dim=-1)
            preds_arcs.extend(predictions_arcs)
            preds_rels.extend(predictions_rels)
            lab_arcs.extend(labels_arcs)
            lab_rels.extend(labels_rels)
    results = compute_metrics((preds_arcs, preds_rels, lab_arcs, lab_rels))
    dp.to("cpu")
    del dp
    gc.collect()
    return results

def test_lang_dp_causal(dp, language_model, pretrained_checkpoint, dataset, best_lambda:float=1.0, batch_size:int=8):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(pretrained_checkpoint, trust_remote_code=True, padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token
    lv = get_language_vector_causal(pretrained_checkpoint, language_model)

    dp = apply_language_vector_to_model(dp, lv, best_lambda)
    dp.to(device).eval()
    pred_heads = []
    pred_rels = []
    label_heads = []
    label_rels = []
    with torch.no_grad():
        for data in tqdm(dataset):
            prompt = f"Sentence: {' '.join(data['tokens'])}.\n DP:"
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            output_ids = dp.generate(
                **inputs,
                max_new_tokens=64,
                do_sample=False
            )
            text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
            output = text.split("DP:")[-1].strip().split()
            label_heads.append(data["dep_head"])
            label_rels.append(data["dep_rel"])
            p_heads = [p.split(":")[0] for p in output]
            p_rels = [p.split(":")[1] if len(p.split(":")) > 1 else "ERR" for p in output]

            pred_heads.append(p_heads)
            pred_rels.append(p_rels)

    results = compute_metrics_causal((pred_heads, pred_rels, label_heads, label_rels))
    dp.to("cpu")
    del dp
    gc.collect()
    return results

if __name__ == "__main__":
    test_lambdas = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    model_base = "base_finetuned"
    base_models = ["bert", "roberta", "qwen"]
    id2label, label2id = get_label_mapping()
    val_datasets = [
        "GUM_en/en_gum-ud-dev.conllu", 
        "val_datasets_pos/es_gsd-ud-dev.conllu", 
        "val_datasets_pos/hi_hdtb-ud-dev.conllu",
        "val_datasets_pos/de_gsd-ud-dev.conllu",
        "val_datasets_pos/zh_gsd-ud-dev.conllu",
        "val_datasets_pos/fr_gsd-ud-dev.conllu",
        "val_datasets_pos/ru_gsd-ud-dev.conllu"
    ]
    test_datasets = [
        "UD_English-PUD/en_pud-ud-test.conllu",
        "UD_Spanish-PUD/es_pud-ud-test.conllu",
        "UD_Hindi-PUD/hi_pud-ud-test.conllu",
        "UD_German-PUD/de_pud-ud-test.conllu", 
        "UD_Chinese-PUD/zh_pud-ud-test.conllu",
        "UD_French-PUD/fr_pud-ud-test.conllu",
        "UD_Russian-PUD/ru_pud-ud-test.conllu"
        ]
    language_models = ["language_en_done", 
                    "language_es_done", 
                    "language_hi_done", 
                    "language_de_done", 
                    "language_zh_done", 
                    "language_fr_done",
                    "language_ru_done"]
    overall_hyperparameter_results = {}
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
            val_dataset = load_conllu_data(val_datasets[idx])
            val_dataset = Dataset.from_pandas(val_dataset)
            test_dataset = load_conllu_data(test_datasets[idx])
            test_dataset = Dataset.from_pandas(test_dataset)
            DP_dataset = DatasetDict({
                "validation": val_dataset,
                "test": test_dataset
            })
            tokenizer = AutoTokenizer.from_pretrained(base_model)
            if base_model_str == "qwen":
                hyperparameter_results = {}
                dp =  AutoModelForCausalLM.from_pretrained(f"{prefix}/{model_base}/DP_en")
                for l in test_lambdas:
                    print("lambda: ", l)
                    accuracy = test_lang_dp_causal(dp, f"{prefix}/{model}", base_model, DP_dataset["validation"].select(range(100)), l)
                    hyperparameter_results[l] = accuracy
                print("hyperparamter search results")
                print(hyperparameter_results)
                overall_hyperparameter_results[model][base_model_str] = hyperparameter_results
                
                best_lambda = max(hyperparameter_results, key=lambda k: hyperparameter_results[k]["uas"])
                if model.split("_")[1] == "en":
                    print("lang en, best lambda 0")
                    best_lambda = 0.0
                print(best_lambda)
                dp =  AutoModelForCausalLM.from_pretrained(f"{prefix}/{model_base}/DP_en")
                with open(f"output/{prefix}/{model_base}/DP.txt", "a") as f:
                    print("language model", model)
                    accuracy= test_lang_dp_causal(dp, f"{prefix}/{model}", base_model, DP_dataset["test"], best_lambda)
                    print(f"accuracy: {accuracy}")  
                    f.write(f"\n======language: {model.split('_')[1]}=======\n")
                    f.write(f"best lambda: {best_lambda}\n")
                    f.write(f"accuracy: {accuracy}\n")
                    f.close()
            else:
                def preprocess(examples):
                    # Credit: https://github.com/cambridgeltl/composable-sft/blob/main/examples/dependency-parsing/run_dp.py
                    features = {}
                    for idx in range(len(examples['tokens'])):
                        invalid_indices = set(i for i, head in enumerate(examples['dep_head'][idx]) if head in ['_', 'None'])
                        for col in ['tokens', 'dep_head', 'dep_rel']:
                            examples[col][idx] = [v for i, v in enumerate(examples[col][idx]) if i not in invalid_indices]

                        tokens = [tokenizer.tokenize(w) for w in examples['tokens'][idx]]
                        word_lengths = [len(w) for w in tokens]

                        tokenized_inputs = tokenizer(
                            examples['tokens'][idx],
                            padding="max_length",
                            max_length=tokenizer.model_max_length,
                            truncation=True,
                            is_split_into_words=True,
                        )

                        tokenized_inputs['labels_arcs'] = [int(x) for x in examples['dep_head'][idx]]
                        tokenized_inputs['labels_rels'] = [label2id[x.split(':')[0]] for x in examples['dep_rel'][idx]]

                        # determine start indices of words
                        tokenized_inputs['word_starts'] = np.cumsum([1] + word_lengths).tolist()

                        for k, v in tokenized_inputs.items():
                            features.setdefault(k, []).append(v)

                    return features
                tokenized_dataset= DP_dataset.map(
                    preprocess,
                    batched=True,
                )

                mlm_model = AutoModelForMaskedLM.from_pretrained(
                    base_model,
                    dtype=torch.float32,
                )
                if base_model_str == "bert":
                    encoder = mlm_model.bert
                else:
                    encoder = mlm_model.roberta
                hyperparameter_results = {}
                torch.set_grad_enabled(False)
                for l in test_lambdas:
                    dp_model = TransformerForBiaffineParsing(encoder, num_labels=len(id2label), bert=base_model_str == "bert")
                    load_model(dp_model, f"{prefix}/{model_base}/DP_en/model.safetensors", device="cpu")
                    accuracy = test_lang_dp(dp_model, f"{prefix}/{model}", base_model, tokenized_dataset["validation"], l)
                    hyperparameter_results[l] = accuracy
                print("hyperparamter search results")
                print(hyperparameter_results)
                overall_hyperparameter_results[model]["bert" if base_model_str == "bert" else "roberta"] = hyperparameter_results
                
                best_lambda = max(hyperparameter_results, key=lambda k: hyperparameter_results[k]["uas"])
                if model.split("_")[1] == "en":
                    print("lang en, best lambda 0")
                    best_lambda = 0.0
                print(best_lambda)
                with open(f"output/{prefix}/{model_base}/DP.txt", "a") as f:
                    print("language model", model)
                    dp_model = TransformerForBiaffineParsing(encoder, num_labels=len(id2label), bert=base_model_str == "bert")
                    load_model(dp_model, f"{prefix}/{model_base}/DP_en/model.safetensors", device="cpu")
                    accuracy= test_lang_dp(dp_model, f"{prefix}/{model}", base_model, tokenized_dataset["test"], best_lambda)
                    print(f"accuracy: {accuracy}")  
                    f.write(f"\n======language: {model.split('_')[1]}=======\n")
                    f.write(f"best lambda: {best_lambda}\n")
                    f.write(f"accuracy: {accuracy}\n")
                    f.close()
    with open(f"output/DP_pretrained_hyperparameter_search.json", "w") as f:
        json.dump(overall_hyperparameter_results, f, indent=4)
        f.close()




