#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse, math, sys
from tqdm.auto import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM, DataCollatorForLanguageModeling
from datasets import load_dataset

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_dir", required=True, help="Path to the local MLM checkpoint")
    p.add_argument("--dataset", default="wikitext", help="HF dataset name (e.g. wikitext)")
    p.add_argument("--subset", default="wikitext-2-raw-v1", help="Dataset subset")
    p.add_argument("--split", default="test", help="Dataset split")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--max_len", type=int, default=512, help="Maximum sequence length")
    return p.parse_args()

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ------------------- Load model & tokenizer -------------------
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForMaskedLM.from_pretrained(
        args.model_dir,
        torch_dtype=torch.float32,
    ).to(device).eval()

    # ------------------- Load evaluation sentences -------------------
    raw = load_dataset(args.dataset, args.subset, split=args.split)
    sentences = [s for s in raw["text"] if s.strip() != ""]
    print(f"Loaded {len(sentences)} non‑empty sentences from {args.dataset}/{args.subset}:{args.split}")

    # ------------------- MLM collator -------------------
    collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=0.15,
    )

    # ------------------- Evaluation loop -------------------
    total_loss, total_tokens = 0.0, 0
    batch_sz = args.batch_size

    def batches(it):
        for i in range(0, len(it), batch_sz):
            yield it[i:i+batch_sz]

    model.eval()
    with torch.no_grad():
        for batch_sent in tqdm(batches(sentences), total=math.ceil(len(sentences)/batch_sz), desc="Eval"):
            tokenized = tokenizer(batch_sent,
                            padding=False,
                            truncation=True,
                            max_length=args.max_len)

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
    print(f"Average token‑wise loss : {avg_loss:.4f}")
    print(f"Perplexity (exp(loss)) : {ppl:.2f}")

    # ------------------- Quick health check -------------------
    nan = [n for n, p in model.named_parameters() if torch.isnan(p).any()]
    inf = [n for n, p in model.named_parameters() if torch.isinf(p).any()]
    if nan:
        print(f"⚠️  NaNs in: {', '.join(nan)}")
    else:
        print("✅ No NaNs")
    if inf:
        print(f"⚠️  Infs in: {', '.join(inf)}")
    else:
        print("✅ No Infs")

    total_norm = sum(p.norm().item() ** 2 for p in model.parameters()) ** 0.5
    print(f"🔎 Total L2 norm of parameters: {total_norm:.2f}")

if __name__ == "__main__":
    main()
