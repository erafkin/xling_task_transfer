import math
from tqdm.auto import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM, DataCollatorForLanguageModeling
from datasets import load_dataset
"""
    Method for calculating perplexity and loss to make sure that the gradients didnt vanish/explode. 
    Scaffolded using ChatGPT.
"""

def eval_model(model_dir, sentences, batch_size: int = 32, max_len: int = 512):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ------------------- Load model & tokenizer -------------------
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForMaskedLM.from_pretrained(
        model_dir,
        torch_dtype=torch.float32,
    ).to(device).eval()
    # ------------------- MLM collator -------------------
    collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=0.15,
    )

    # ------------------- Evaluation loop -------------------
    total_loss, total_tokens = 0.0, 0
    batch_sz = batch_size

    def batches(it):
        for i in range(0, len(it), batch_sz):
            yield it[i:i+batch_sz]

    model.eval()
    with torch.no_grad():
        for batch_sent in tqdm(batches(sentences), total=math.ceil(len(sentences)/batch_sz), desc="Eval"):
            tokenized = tokenizer(batch_sent,
                            padding=False,
                            truncation=True,
                            max_length=max_len)

            tokenized_list = [{k: v[i] for k, v in tokenized.items()} for i in range(len(tokenized['input_ids']))]
            batch = collator(tokenized_list)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            loss = model(input_ids=input_ids,
                         attention_mask=attention_mask,
                         labels=labels).loss

            valid = (labels != -100).sum().item()
            total_loss += loss.item() * valid
            total_tokens += valid

    avg_loss = total_loss / total_tokens
    ppl = math.exp(avg_loss)

    print("\n=== Perplexity result ===")
    print(f"Model: {model_dir}")
    print(f"Average token‑wise loss : {avg_loss:.4f}")
    print(f"Perplexity (exp(loss)) : {ppl:.2f}")

    # ------------------- Quick health check -------------------
    nan = [n for n, p in model.named_parameters() if torch.isnan(p).any()]
    inf = [n for n, p in model.named_parameters() if torch.isinf(p).any()]
    if nan:
        print(f" NaNs in: {', '.join(nan)}")
    else:
        print("No NaNs")
    if inf:
        print(f" Infs in: {', '.join(inf)}")
    else:
        print("No Infs")

    total_norm = sum(p.norm().item() ** 2 for p in model.parameters()) ** 0.5
    print(f"🔎 Total L2 norm of parameters: {total_norm:.2f}")
    return avg_loss, ppl, nan, inf, total_norm

if __name__ == "__main__":
    models = ["language_en", "language_es", "language_hi", "language_de", "language_zh"]

    base_model = "google-bert/bert-base-multilingual-cased"
    with open("output/languages/perplexity/summary_mbert.txt", "w") as file:
        for model in models:
            language = model.split("_")[1]
            dataset = load_dataset("uonlp/CulturaX", language, streaming=True, split="train")#, cache_dir="/home/scratch/epr41")
            dataset = dataset.take(100)
            sentences = []
            for d in dataset:
                if language == "zh":
                    sentences += d["text"].split("。")
                elif language == "hi":
                    sentences += d["text"].split("\n")
                else: 
                    sentences += d["text"].split(".")
            avg_loss, ppl, nan, inf, total_norm = eval_model(base_model, sentences=sentences)
            file.write(f"\n=== LANGUAGE {language} ===\n")
            file.write("\n=== Perplexity ===\n")
            file.write(f"Model: base\n")
            file.write(f"Average token‑wise loss : {avg_loss:.4f}\n")
            file.write(f"Perplexity (exp(loss)) : {ppl:.2f}\n")
            if nan:
                file.write(f" NaNs in: {', '.join(nan)}\n")
            else:
                file.write("No NaNs\n")
            if inf:
                file.write(f" Infs in: {', '.join(inf)}\n")
            else:
                file.write("No Infs\n")
            file.write(f"🔎 Total L2 norm of parameters: {total_norm:.2f}\n")
            avg_loss, ppl, nan, inf, total_norm = eval_model(f"bert-multilingual/{model}_done", sentences=sentences)
            file.write("\n=== Perplexity result ===\n")
            file.write(f"Model: {model}\n")
            file.write(f"Average token‑wise loss : {avg_loss:.4f}\n")
            file.write(f"Perplexity (exp(loss)) : {ppl:.2f}\n")
            if nan:
                file.write(f" NaNs in: {', '.join(nan)}\n")
            else:
                file.write("No NaNs\n")
            if inf:
                file.write(f" Infs in: {', '.join(inf)}\n")
            else:
                file.write("No Infs")
            file.write(f"Total L2 norm of parameters: {total_norm:.2f}\n")
    file.close()
