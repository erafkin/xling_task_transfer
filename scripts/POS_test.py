import torch
from torch.utils.data import DataLoader
from typing import List
from datasets import load_dataset, Dataset
from transformers import AutoModelForMaskedLM, AutoTokenizer, AutoConfig, DataCollatorForTokenClassification
from tqdm import tqdm
from safetensors.torch import load_model
from task_vectors import TaskVector
from finetune_tasks import TokenClassificationHead, load_conllu_data

def get_language_vector(base_model: str, saved_language: str):
    lang_vector = TaskVector(pretrained_model=AutoModelForMaskedLM.from_pretrained(base_model),
                             finetuned_model=AutoModelForMaskedLM.from_pretrained(saved_language, local_files_only=True))
    return lang_vector

def apply_language_vector_to_model(pos_model_checkpoint: str, language_vector:TaskVector, lambda_coef: float):
    ner_model = language_vector.apply_to(pos_model_checkpoint, scaling_coef=lambda_coef)
    return ner_model

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
def lambda_search(eval_set, coef):
    ...

def get_label_mapping():
    pos_dataset = load_conllu_data("GUM_en/en_gum-ud-train.conllu")
    pos_dataset = Dataset.from_pandas(pos_dataset)
    unique_tags = sorted({tag for ex in pos_dataset for tag in ex["pos_tags"]})
    label2id = {tag: i for i, tag in enumerate(unique_tags)}
    id2label = {i: tag for tag, i in label2id.items()}
    return id2label, label2id

def test_lang_pos(ner, language_model, pretrained_checkpoint, language_folder, label2id, lambdas: List[float]= [0.0, 0.25, 0.5, 0.75, 1.0], batch_size:int=32):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if language_model != "language_en_done":
        lv = get_language_vector(pretrained_checkpoint, language_model)
        best_lambda = 1.0
        ner = apply_language_vector_to_model(ner, lv, best_lambda) # TODO find best lambda:
    test_dataset = load_conllu_data(language_folder)
    test_dataset = Dataset.from_pandas(test_dataset)

    tokenizer = AutoTokenizer.from_pretrained(pretrained_checkpoint)
    def tokenize_and_align_labels(examples):
        # from https://reybahl.medium.com/token-classification-in-python-with-huggingface-3fab73a6a20e
        tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)
        labels = []
        for i, label in enumerate(examples[f"pos_tags"]):
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
    test_dataset= test_dataset.map(
        tokenize_and_align_labels,
        batched=True,
    )
    preds = []
    labels = []
    ner.to(device).eval()
    test_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=collator)
    with torch.no_grad():
        for batch in tqdm(test_dataloader):
            input_ids = batch["input_ids"].to(device)
            ps = ner(input_ids)["logits"]
            ps = torch.argmax(ps, dim=-1).cpu().tolist()
            ls = batch["labels"].cpu().tolist()
            preds += ps
            labels += ls
    accuracy = compute_metrics(preds, labels)
    ner.to("cpu")
    return accuracy

if __name__ == "__main__":
    datasets = [
        "UD_English-PUD/en_pud-ud-test.conllu",
        "UD_Spanish-PUD/es_pud-ud-test.conllu",
        "UD_Hindi-PUD/hi_pud-ud-test.conllu",
        "UD_German-PUD/de_pud-ud-test.conllu", 
        "UD_Chinese-PUD/zh_pud-ud-test.conllu"
        ]
    language_models = ["language_en_done", "language_es_done", "language_hi_done", "language_de_done", "language_zh_done"]
    id2label, label2id = get_label_mapping()
    encoder_checkpoint = "language_en_done"
    config = AutoConfig.from_pretrained(encoder_checkpoint)
    mlm_model = AutoModelForMaskedLM.from_pretrained(
        "language_en_done",
        config=config,
        dtype=torch.float32,
    )
    bert_encoder = mlm_model.bert
    ner_model = TokenClassificationHead(bert_encoder, num_labels=len(id2label))
    load_model(ner_model, "POS_en/model.safetensors", device="cpu")
    print('pos model loaded')
    with open("output/POS_1.0.txt", "w") as f:
        for idx, model in enumerate(language_models):
            print("language model", model)
            accuracy= test_lang_pos(ner_model, model, "language_en_done", datasets[idx], label2id)
            print(f"accuracy: {accuracy}")  
            f.write(f"\n======language: {model.split('_')[1]}=======\n")
            f.write(f"accuracy: {accuracy}\n")
            
        f.close()





