from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, AutoModel
from trl import SFTTrainer, SFTConfig
from datasets import Dataset, DatasetDict

import wandb
import torch
import torch.nn as nn

import requests


class EncoderGRUReinflector(nn.Module):
    def __init__(self, model_name, vocab_size, hidden=768):
        super().__init__()

        self.encoder = AutoModel.from_pretrained(model_name)

        # project encoder → small space
        self.enc_proj = nn.Linear(
            self.encoder.config.hidden_size, hidden
        )

        # shared embeddings
        self.embedding = nn.Embedding(vocab_size, hidden)

        # GRU decoder
        self.decoder = nn.GRU(
            input_size=hidden,
            hidden_size=hidden,
            batch_first=True
        )

        # output layer (tied idea optional)
        self.out = nn.Linear(hidden, vocab_size)

    def forward(self, input_ids, attention_mask, decoder_input_ids, labels=None):

        enc = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        ).last_hidden_state

        # compress encoder output
        enc = self.enc_proj(enc).mean(dim=1).unsqueeze(1)

        # decoder inputs
        dec_in = self.embedding(decoder_input_ids)

        # prepend context
        dec_in = dec_in + enc

        dec_out, _ = self.decoder(dec_in)

        logits = self.out(dec_out)

        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fn(
                logits.view(-1, logits.size(-1)),
                labels.view(-1)
            )

        return {"loss": loss, "logits": logits}

def get_unimorph_data(lang:str = "eng") -> DatasetDict:
    base = "https://raw.githubusercontent.com/unimorph/"
    url = f"{base}{lang}/master/{lang}"
    text = requests.get(url).text.strip().split("\n")
    dataset_as_list = [
        {
            "root": l[0],
            "inflection": l[1],
            "features": l[2] if len(l) > 2 else "None"
        }
        for line in text
        for l in [line.split("\t")]
        if len(l) >= 2
    ]
    dataset = Dataset.from_list(dataset_as_list)
    dataset_dict_temp = dataset.train_test_split(test_size=0.2, seed=42)
    dataset_dict_temp1 = dataset_dict_temp["test"].train_test_split(test_size=0.5, seed=42)
    dataset_dict = DatasetDict({"train": dataset_dict_temp["train"], "validation": dataset_dict_temp1["train"], "test": dataset_dict_temp1["test"]})
    return dataset_dict

def train_model_causal(model_checkpoint):
    """
        parse dataset to be text-to-text
    """
    train_data = []
    dataset = get_unimorph_data()
    
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, trust_remote_code=True)
    if "roberta" in model_checkpoint:
        output_prefix = "xlm-roberta/base_finetuned"
        run_prefix = "roberta"
    elif "bert" in model_checkpoint:
        output_prefix = "bert-multilingual/base_finetuned"
        run_prefix = "bert"
    elif "granite" in model_checkpoint:
        output_prefix = "granite/base_finetuned"
        run_prefix = "granite"
    else:
        output_prefix = "qwen/base_finetuned"
        run_prefix = "qwen"
    if "bert" in model_checkpoint:
        def preprocess(ex, tokenizer):
            src = f"Root {ex['root']} Features {ex['features']}"
            tgt = ex["inflection"]

            model_in = tokenizer(
                src,
                truncation=True,
                padding="max_length",
                max_length=128
            )
            tgt_tok = tokenizer(
                tgt,
                truncation=True,
                padding="max_length",
                max_length=32
            )

            decoder_input = tgt_tok["input_ids"][:-1]
            labels = tgt_tok["input_ids"][1:]

            labels = [
                t if t != tokenizer.pad_token_id else -100
                for t in labels
            ]

            model_in["decoder_input_ids"] = decoder_input
            model_in["labels"] = labels
            return model_in
        tokenizer.pad_token = tokenizer.sep_token or tokenizer.eos_token

        model = EncoderGRUReinflector(model_checkpoint, vocab_size=len(tokenizer))

        train_ds = dataset["train"].map(lambda x: preprocess(x, tokenizer))
        val_ds = dataset["validation"].map(lambda x: preprocess(x, tokenizer))

        train_ds.set_format("torch")
        val_ds.set_format("torch")

        args = TrainingArguments(
            output_dir=f"{output_prefix}/morph_en",
            per_device_train_batch_size=32,
            num_train_epochs=3,
            save_strategy="no",
            eval_strategy="epoch",
            learning_rate=1e-5,
            weight_decay=0.01,
            logging_steps=1000,
            report_to='wandb',
            project='xlt',
            run_name=f"{run_prefix}_morph_en",
        )

        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=train_ds,
            eval_dataset=val_ds
        )

    else:
    
        model = AutoModelForCausalLM.from_pretrained(
            model_checkpoint,
            dtype=torch.float32,
            device_map="auto",
        )
        for datum in dataset["train"]:
            train_data.append(
                {
                    "text": (
                        f"Root: {datum['root']}\n Features: {datum['features']}\n Inflection: {datum['inflection']}"
                    )
                }
            )
        validation_data = []
        for datum in dataset["validation"]:
            validation_data.append(
                {
                    "text": (
                        f"Root: {datum['root']}\n Features: {datum['features']}\n Inflection: {datum['inflection']}"
                    )
                }
            )
        train_dataset = Dataset.from_list(train_data)
        validation_dataset = Dataset.from_list(validation_data)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            model.config.pad_token_id = tokenizer.eos_token_id

        training_args = SFTConfig(
                output_dir=f"{output_prefix}/morph_en",
                eval_strategy="epoch",
                learning_rate=1e-5,
                num_train_epochs=3, 
                weight_decay=0.01,
                per_device_train_batch_size=16 if "granite" in model_checkpoint else 8,
                per_device_eval_batch_size=8 if "granite" in model_checkpoint else 4,
                push_to_hub=False,
                save_strategy="no",
                warmup_ratio=0.01,
                lr_scheduler_type="linear",
                max_grad_norm=1.0,
                bf16=True,
                max_length=512,
                report_to='wandb',
                project='xlt',
                run_name=f"{run_prefix}_morph_en",
        )
        if "granite" in model_checkpoint:
            model.config.use_cache = False
        trainer = SFTTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=validation_dataset
            )    
    
    trainer.train()
    trainer.save_model(f"{output_prefix}/morph_en")
    wandb.finish()

if __name__ == "__main__":
    roberta = "FacebookAI/xlm-roberta-base"
    bert = "google-bert/bert-base-multilingual-cased"
    qwen = "Qwen/Qwen3-0.6B"
    granite = "ibm-granite/granite-4.0-350m"
    for m in [granite]:
        train_model_causal(m)
