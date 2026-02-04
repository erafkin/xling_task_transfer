import torch
from torch import nn
import numpy as np
from transformers import AutoTokenizer, AutoConfig, AutoModelForMaskedLM, Trainer, TrainingArguments, AutoModelForCausalLM, BitsAndBytesConfig
from datasets import Dataset, DatasetDict
from peft import LoraConfig, get_peft_model,  prepare_model_for_kbit_training

from scripts.dp.dp_model import TransformerForBiaffineParsing, DataCollatorForDependencyParsing
from scripts.task_utils import load_conllu_data

from seqeval.metrics import f1_score
import re
from trl import SFTTrainer, SFTConfig

UD_HEAD_LABELS = [
    "_",
    "acl",
    "advcl",
    "advmod",
    "amod",
    "appos",
    "aux",
    "case",
    "cc",
    "ccomp",
    "clf",
    "compound",
    "conj",
    "cop",
    "csubj",
    "dep",
    "det",
    "discourse",
    "dislocated",
    "expl",
    "fixed",
    "flat",
    "goeswith",
    "iobj",
    "list",
    "mark",
    "nmod",
    "nsubj",
    "nummod",
    "obj",
    "obl",
    "orphan",
    "parataxis",
    "punct",
    "reparandum",
    "root",
    "vocative",
    "xcomp",
]

def train_DP_model(model_checkpoint, GUM_folder: str = "GUM_en"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dataset = load_conllu_data(f"{GUM_folder}/en_gum-ud-train.conllu")
    dev_dataset = load_conllu_data(f"{GUM_folder}/en_gum-ud-dev.conllu")
    dataset = DatasetDict({"train": Dataset.from_pandas(train_dataset), "dev": Dataset.from_pandas(dev_dataset)})
    dep_rel_tags = sorted(UD_HEAD_LABELS)
    label2id = {tag: i for i, tag in enumerate(dep_rel_tags)}
    id2label = {i: tag for tag, i in label2id.items()}
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    data_collator = DataCollatorForDependencyParsing(tokenizer=tokenizer, max_length=tokenizer.model_max_length)
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

    tokenized_dataset= dataset.map(
        preprocess,
        batched=True,
        remove_columns=["tokens", "pos_tags", "dep_rel", "dep_head"]
    )
    config = AutoConfig.from_pretrained(model_checkpoint)
    mlm_model = AutoModelForMaskedLM.from_pretrained(
        model_checkpoint,
        config=config,
        dtype=torch.float32,
    ).to(device)

    if "roberta" in model_checkpoint:
        encoder = mlm_model.roberta
        is_bert = False
    else:
        encoder = mlm_model.bert
        is_bert = True
    model = TransformerForBiaffineParsing(encoder=encoder, num_labels=len(id2label), bert=is_bert)

    def set_trainable(mod: nn.Module, train_encoder: bool = False):
        for n, p in mod.named_parameters():
            if "classifier" in n:
                p.requires_grad = True
            else:
                p.requires_grad = train_encoder
        mod.train()
    set_trainable(model, train_encoder=True) 
    output_prefix = "bert-multilingual/base_finetuned" if is_bert else  "xlm-roberta/base_finetuned" 

    training_args = TrainingArguments(
            output_dir=f"{output_prefix}/DP_en",
            eval_strategy="no",
            learning_rate=2e-5,
            num_train_epochs=3, 
            weight_decay=0.01,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            push_to_hub=False,
            save_strategy="no",
            fp16=True            )    
    trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset["train"],
            eval_dataset=tokenized_dataset["dev"],
            data_collator=data_collator,
            tokenizer=tokenizer, # no training results because it was crashing the GPU
        )

    trainer.train()
    trainer.save_model(f"{output_prefix}/DP_en")

def train_DP_model_causal(model_checkpoint, GUM_folder: str = "GUM_en"):
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_checkpoint,
        dtype=torch.float32,
        device_map="auto"
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id

    train_dataset = load_conllu_data(f"{GUM_folder}/en_gum-ud-train.conllu")
    dev_dataset = load_conllu_data(f"{GUM_folder}/en_gum-ud-dev.conllu")
    dataset = DatasetDict({"train": Dataset.from_pandas(train_dataset), "dev": Dataset.from_pandas(dev_dataset)})
    train_data = []
    for datum in dataset["train"]:
        train_data.append(
            {
                "text": (
                    f"Sentence: {' '.join(datum['tokens'])}.\n DP:\n {' '.join([f'{head}:{rel}' for head, rel in zip(datum['dep_head'], datum['dep_rel'])])}"
                )
            }
        )
    validation_data = []
    for datum in dataset["dev"]:
        validation_data.append(
            {
                "text": (
                    f"Sentence: {' '.join(datum['tokens'])}.\n DP:\n {' '.join([f'{head}:{rel}' for head, rel in zip(datum['dep_head'], datum['dep_rel'])])}"
                )
            }
        )
    train_dataset = Dataset.from_list(train_data)
    validation_dataset = Dataset.from_list(validation_data)
    output_prefix = "qwen/base_finetuned"

    training_args = SFTConfig(
            output_dir=f"{output_prefix}/DP_en",
            eval_strategy="epoch",
            learning_rate=2e-5,
            num_train_epochs=10, 
            weight_decay=0.01,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=4,
            push_to_hub=False,
            save_strategy="no",
            fp16=False,
            max_length=512
    )
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset
    )    
    
    trainer.train()
    trainer.save_model(f"{output_prefix}/DP_en")

if __name__ == "__main__":
    roberta = "FacebookAI/xlm-roberta-base"
    bert = "google-bert/bert-base-multilingual-cased"
    qwen = "Qwen/Qwen3-0.6B"
    train_DP_model_causal(qwen)


