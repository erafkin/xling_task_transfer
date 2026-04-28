import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForMaskedLM, AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
from safetensors.torch import load_model
from scripts.task_vectors import TaskVector
from scripts.morph_reinflection.morph_train import BertDecoderReinflector, get_unimorph_data
import gc
import json

def get_language_vector(base_model: str, saved_language: str):
    lang_vector = TaskVector(pretrained_model=AutoModelForMaskedLM.from_pretrained(base_model),
                             finetuned_model=AutoModelForMaskedLM.from_pretrained(saved_language, local_files_only=True))
    return lang_vector

def get_language_vector_causal(base_model: str, saved_language: str):
    lang_vector = TaskVector(pretrained_model=AutoModelForCausalLM.from_pretrained(base_model),
                             finetuned_model=AutoModelForCausalLM.from_pretrained(saved_language, local_files_only=True))
    return lang_vector


def apply_language_vector_to_model(morph_model_checkpoint: str, language_vector:TaskVector, lambda_coef: float):
    model = language_vector.apply_to(morph_model_checkpoint, scaling_coef=lambda_coef)
    return model

def test_lang_morph(
    morph_model,
    language_model,
    pretrained_checkpoint,
    dataset,
    tokenizer,
    best_lambda: float = 1.0,
    batch_size: int = 64,
    max_len: int = 5,
):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    lv = get_language_vector(pretrained_checkpoint, language_model)
    model = apply_language_vector_to_model(morph_model, lv, best_lambda)
    tokenizer = AutoTokenizer.from_pretrained(pretrained_checkpoint)
    model.to(device).eval()
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    dataloader = DataLoader(dataset, batch_size=batch_size)

    total, correct = 0, 0
    all_preds, all_targets = [], []

    with torch.inference_mode():
        for batch in tqdm(dataloader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            B = input_ids.size(0)

            # ---- ENCODE ONCE ----
            enc = model.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask
            ).last_hidden_state
            enc = model.enc_proj(enc).mean(dim=1)  # (B, H)

            # ---- INIT ----
            start_id = tokenizer.pad_token_id
            prev = torch.full((B, 1), start_id, dtype=torch.long, device=device)
            hidden = None

            generated = []

            # ---- FAST DECODING ----
            for _ in range(max_len):
                x = model.embedding(prev) + enc.unsqueeze(1)  # (B,1,H)
                out, hidden = model.decoder(x, hidden)
                logits = model.out(out)  # (B,1,V)

                next_tokens = logits[:, -1, :].argmax(dim=-1, keepdim=True)
                generated.append(next_tokens)

                prev = next_tokens

                if (next_tokens == tokenizer.eos_token_id):
                    break

            generated = torch.cat(generated, dim=1)

            # ---- DECODE ----
            preds = tokenizer.batch_decode(generated, skip_special_tokens=True)

            labels_clean = labels.clone()
            labels_clean[labels_clean == -100] = tokenizer.pad_token_id
            targets = tokenizer.batch_decode(labels_clean, skip_special_tokens=True)
            print("preds", preds[0:5])
            print("targets", targets[0:5])
            # ---- METRICS ----
            for p, t in zip(preds, targets):
                all_preds.append(p)
                all_targets.append(t)

                if p.strip() == t.strip():
                    correct += 1
                total += 1

    acc = correct / total
    print("accuracy:", acc)

    model.to("cpu")
    del model
    gc.collect()
    return acc


def test_lang_morph_causal(morph, language_model, pretrained_checkpoint, dataset, best_lambda:float=1.0, batch_size:int=32):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_checkpoint, trust_remote_code=True, padding_side="left"
    )
    tokenizer.pad_token = tokenizer.eos_token

    lv = get_language_vector_causal(pretrained_checkpoint, language_model)
    model = apply_language_vector_to_model(morph, lv, best_lambda)

    def collate_fn(batch):
        prompts = [
            f"Root: {d['root']}\n Features: {d['features']}\n Inflection: "
            for d in batch
        ]
        labels = [d["inflection"] for d in batch]
        return prompts, labels

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=1,
        pin_memory=True
    )

    model.to(device).eval()
    preds, labels = [], []

    with torch.no_grad():
        for prompts, batch_labels in tqdm(loader):
            inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(device)

            output_ids = model.generate(
                **inputs,
                max_new_tokens=5,
                do_sample=False
            )

            texts = tokenizer.batch_decode(output_ids, skip_special_tokens=True)

            for text, lab in zip(texts, batch_labels):
                pred = text.split("Inflection:")[-1].strip().split()
                if len(pred) > 0:
                    preds.append(pred[0])
                else:
                    preds.append("")
                labels.append(lab)

    print("preds", preds[:5])
    print("labs", labels[:5])
    print(len(preds), len(labels))
    acc = sum(p == l for p, l in zip(preds, labels)) / len(preds)
    print("accuracy:", acc)

    model.to("cpu")
    del model
    gc.collect()
    return acc

if __name__ == "__main__":
    test_lambdas = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    model_base = "base_finetuned"
    base_models = ["bert", "roberta", "qwen", "granite"]
    base_models = ["qwen", "granite"]
    language_models = ["language_en_done", 
                    "language_es_done", 
                    "language_hi_done", 
                    "language_de_done", 
                    "language_fr_done",
                    "language_ru_done"]
    language_code_mapping = ["eng", "spa", "hin", "deu", "fra", "rus"]
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
            elif base_model_str == "granite":
                base_model = "ibm-granite/granite-4.0-350m"
                prefix = "granite"
            else:
                base_model = "Qwen/Qwen3-0.6B"
                prefix = "qwen"
            # handle data
            unimorph_ds = get_unimorph_data(language_code_mapping[idx])
            val_dataset = unimorph_ds["validation"]
            test_dataset = unimorph_ds["test"]
            
            tokenizer = AutoTokenizer.from_pretrained(base_model)
            if base_model_str == "qwen" or base_model_str == "granite":
                hyperparameter_results = {}
                morph =  AutoModelForCausalLM.from_pretrained(f"{prefix}/{model_base}/morph_en")
                if base_model_str == "granite":
                    morph.config.use_cache = False
                for l in test_lambdas:
                    print("lambda: ", l)
                    accuracy = test_lang_morph_causal(morph, f"{prefix}/{model}", base_model, val_dataset.take(5000), l)
                    hyperparameter_results[l] = accuracy
                print("hyperparamter search results")
                print(hyperparameter_results)
                overall_hyperparameter_results[model][base_model_str] = hyperparameter_results
                
                best_lambda = max(hyperparameter_results, key=hyperparameter_results.get)
                if model.split("_")[1] == "en":
                    print("lang en, best lambda 0")
                    best_lambda = 0.0
                print(best_lambda)
                morph =  AutoModelForCausalLM.from_pretrained(f"{prefix}/{model_base}/morph_en")
                with open(f"output/{prefix}/{model_base}/morph.txt", "a") as f:
                    print("language model", model)
                    accuracy= test_lang_morph_causal(morph, f"{prefix}/{model}", base_model, test_dataset, best_lambda)
                    print(f"accuracy: {accuracy}")  
                    f.write(f"\n======language: {model.split('_')[1]}=======\n")
                    f.write(f"best lambda: {best_lambda}\n")
                    f.write(f"accuracy: {accuracy}\n")
                    f.close()
            else:
                def preprocess(ex):
                    src = [f"Root {r} Features {f}" for r, f in zip(ex["root"], ex["features"])]
                    tgt = ex["inflection"]

                    model_in = tokenizer(src, truncation=True, padding="max_length", max_length=128)
                    tgt_tok = tokenizer(tgt, truncation=True, padding="max_length", max_length=32)

                    decoder_input_ids = [ids[:-1] for ids in tgt_tok["input_ids"]]
                    labels = [[t if t != tokenizer.pad_token_id else -100 for t in ids[1:]] for ids in tgt_tok["input_ids"]]

                    model_in["decoder_input_ids"] = decoder_input_ids
                    model_in["labels"] = labels
                    return model_in
                tokenizer.pad_token = tokenizer.sep_token or tokenizer.eos_token
                val_tokenized= val_dataset.map(
                    preprocess,
                    batched=True,
                )
                test_tokenized= test_dataset.map(
                    preprocess,
                    batched=True,
                )

                hyperparameter_results = {}
                torch.set_grad_enabled(False)
                for l in test_lambdas:
                    morph_model = BertDecoderReinflector(base_model, vocab_size=len(tokenizer))
                    load_model(morph_model, f"{prefix}/{model_base}/morph_en/model.safetensors", device="cpu")
                    accuracy = test_lang_morph(morph_model, f"{prefix}/{model}", base_model, val_tokenized, l)
                    hyperparameter_results[l] = accuracy
                print("hyperparamter search results")
                print(hyperparameter_results)
                overall_hyperparameter_results[model]["bert" if base_model_str == "bert" else "roberta"] = hyperparameter_results
                
                best_lambda = max(hyperparameter_results, key=hyperparameter_results.get)
                if model.split("_")[1] == "en":
                    print("lang en, best lambda 0")
                    best_lambda = 0.0
                print(best_lambda)
                with open(f"output/{prefix}/{model_base}/morph.txt", "a") as f:
                    print("language model", model)
                    morph_model = BertDecoderReinflector(base_model, vocab_size=len(tokenizer))
                    load_model(morph_model, f"{prefix}/{model_base}/morph_en/model.safetensors", device="cpu")
                    accuracy= test_lang_morph(morph_model, f"{prefix}/{model}", base_model, test_tokenized, best_lambda)
                    print(f"accuracy: {accuracy}")  
                    f.write(f"\n======language: {model.split('_')[1]}=======\n")
                    f.write(f"best lambda: {best_lambda}\n")
                    f.write(f"accuracy: {accuracy}\n")
                    f.close()
    with open(f"output/morph_pretrained_hyperparameter_search.json", "w") as f:
        json.dump(overall_hyperparameter_results, f, indent=4)
        f.close()




