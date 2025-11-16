import torch
from torch.utils.data import DataLoader
from typing import List
from datasets import Dataset, DatasetDict
from transformers import AutoModelForMaskedLM, AutoTokenizer, DataCollatorForTokenClassification
from tqdm import tqdm
from safetensors.torch import load_model
from scripts.task_vectors import TaskVector
from scripts.task_utils import load_conllu_data
from scripts.dp.dp_model import TransformerForBiaffineParsing
import gc
import json
import numpy as np

def get_language_vector(base_model: str, saved_language: str):
    lang_vector = TaskVector(pretrained_model=AutoModelForMaskedLM.from_pretrained(base_model),
                             finetuned_model=AutoModelForMaskedLM.from_pretrained(saved_language, local_files_only=True))
    return lang_vector

def apply_language_vector_to_model(dp_model_checkpoint: str, language_vector:TaskVector, lambda_coef: float):
    dp_model = language_vector.apply_to(dp_model_checkpoint, scaling_coef=lambda_coef)
    return dp_model

def compute_metrics(predictions, labels):

    predicted_head, predicted_arc = predictions
    head_labels, arc_labels = labels
    predicted_indices = predicted_head.long()
    predicted_labels = predicted_arc.long()
    gold_indices = head_labels.long()
    gold_labels = arc_labels.long()

    correct_indices = predicted_indices.eq(gold_indices).long()
    correct_labels = predicted_labels.eq(gold_labels).long()
    correct_labels_and_indices = correct_indices * correct_labels

    unlabeled_correct += correct_indices.sum().item()
    labeled_correct += correct_labels_and_indices.sum().item()
    total_words += correct_indices.numel()

    if total_words > 0.0:
        unlabeled_attachment_score = unlabeled_correct / total_words
        labeled_attachment_score = labeled_correct / total_words
    return {
        "uas": unlabeled_attachment_score * 100,
        "las": labeled_attachment_score * 100,
    }

def get_label_mapping():
    dp_dataset = load_conllu_data("GUM_en/en_gum-ud-train.conllu")
    dp_dataset = Dataset.from_pandas(dp_dataset)
    unique_tags = sorted({tag for ex in dp_dataset for tag in ex["dep_rel"]})
    label2id = {tag: i for i, tag in enumerate(unique_tags)}
    id2label = {i: tag for tag, i in label2id.items()}
    return id2label, label2id

def compute_metrics(eval_pred):
    # adapted from https://github.com/cambridgeltl/composable-sft/blob/main/examples/dependency-parsing/dp/utils_udp.py#L87
    
    predicted_head, predicted_arc, head_labels, arc_labels = eval_pred
    predicted_indices = predicted_head.long()
    predicted_labels = predicted_arc.long()
    gold_indices = head_labels.long()
    gold_labels = arc_labels.long()

    correct_indices = predicted_indices.eq(gold_indices).long()
    correct_labels = predicted_labels.eq(gold_labels).long()
    correct_labels_and_indices = correct_indices * correct_labels

    unlabeled_correct += correct_indices.sum().item()
    labeled_correct += correct_labels_and_indices.sum().item()
    total_words += correct_indices.numel()

    if total_words > 0.0:
        unlabeled_attachment_score = unlabeled_correct / total_words
        labeled_attachment_score = labeled_correct / total_words
    return {
        "uas": unlabeled_attachment_score * 100,
        "las": labeled_attachment_score * 100,
    }

def test_lang_dp(dp, language_model, pretrained_checkpoint, dataset, best_lambda:float=1.0, batch_size:int=32):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lv = get_language_vector(pretrained_checkpoint, language_model)
    dp = apply_language_vector_to_model(dp, lv, best_lambda)
    dp.to(device).eval()
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
    test_dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collator)
    preds_arcs = []
    preds_rels = []
    lab_arcs = []
    lab_rels = []
    with torch.no_grad():
        for batch in tqdm(test_dataloader):
            inputs = {k: v.to(device) for k, v in batch.items() if k != "label"}
            _, (rel_preds, arc_preds), _ = dp(**inputs)
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

if __name__ == "__main__":
    test_lambdas = [0.0, 0.1, 0.2, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9, 1.0]
    model_base = "base_finetuned"
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
    overall_hyperparameter_results = {}
    for idx, model in enumerate(language_models):
        overall_hyperparameter_results[model] = {}
        for bert in bert_values:
            overall_hyperparameter_results[model]["bert" if bert else "roberta"] = {}
            if bert:
                base_model = "google-bert/bert-base-multilingual-cased"
                prefix = "bert-multilingual"
            else:
                base_model = "FacebookAI/xlm-roberta-base"
                prefix = "xlm-roberta"
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
            if bert:
                encoder = mlm_model.bert
            else:
                encoder = mlm_model.roberta
            hyperparameter_results = {}
            for l in test_lambdas:
                dp_model = TransformerForBiaffineParsing(encoder, num_labels=len(id2label), bert=bert)
                load_model(dp_model, f"{prefix}/{model_base}/DP_en/model.safetensors", device="cpu")
                accuracy = test_lang_dp(dp_model, f"{prefix}/{model}", base_model, tokenized_dataset["validation"], l)
                hyperparameter_results[l] = accuracy
            print("hyperparamter search results")
            print(hyperparameter_results)
            overall_hyperparameter_results[model]["bert" if bert else "roberta"] = hyperparameter_results
            best_lambda = max(hyperparameter_results, key=lambda k: hyperparameter_results[k]["uas"])
            print(best_lambda)
            with open(f"output/{prefix}/{model_base}/DP.txt", "a") as f:
                print("language model", model)
                dp_model = TransformerForBiaffineParsing(encoder, num_labels=len(id2label), bert=bert)
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




