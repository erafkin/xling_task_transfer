from transformers import (
    AutoModelForTokenClassification, 
    AutoTokenizer, 
    DataCollatorForTokenClassification,
    TrainingArguments, 
    Trainer
)
import torch
from datasets import load_dataset
import argparse

def train_NER_model(model_checkpoint):
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

    def tokenize_and_align_labels(examples):
        # from https://reybahl.medium.com/token-classification-in-python-with-huggingface-3fab73a6a20e
        tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)

        labels = []
        for i, label in enumerate(examples[f"ner_tags_index"]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)  # Map tokens to their respective word.
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:  # Set the special tokens to -100.
                if word_idx is None:
                    label_ids.append(-100)
                elif word_idx != previous_word_idx:  # Only label the first token of a given word.
                    label_ids.append(label[word_idx])
                else:
                    label_ids.append(-100)
                previous_word_idx = word_idx
            labels.append(label_ids)

        tokenized_inputs["labels"] = labels
        return tokenized_inputs
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
    NER_dataset = load_dataset("MultiCoNER/multiconer_v2", "English (EN)", trust_remote_code=True)
    print("done")
    tokenized_dataset= NER_dataset.map(
        tokenize_and_align_labels,
        batched=True,
        remove_columns=NER_dataset["train"].column_names
    )
    id2label = {}
    label2id = {}
    for row in NER_dataset["train"]:
        for idx, tag in enumerate(row["ner_tags"]):
            label2id[tag] = row["ner_tags_index"][idx]
            id2label[row["ner_tags_index"][idx]] = tag
    all_labels = []
    for row in tokenized_dataset["train"]:
        for l in row["labels"]:
            if l != -100:
                all_labels.append(l)
    
    print(f"[DEBUG] Labels min: {min(all_labels)}, max: {max(all_labels)}")
    model = AutoModelForTokenClassification.from_pretrained(
        model_checkpoint,
        dtype=torch.float16,
        device_map="auto",
        local_files_only=True,
        id2label=id2label, 
        label2id=label2id,
        num_labels=max(all_labels)
    )
    training_args = TrainingArguments(
            output_dir=f"NER_en",
            eval_strategy="no",
            learning_rate=2e-5,
            num_train_epochs=3, 
            weight_decay=0.01,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            push_to_hub=False,
            save_strategy="no",
            fp16=True
        )    
    trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset["train"],
            eval_dataset=tokenized_dataset["validation"],
            data_collator=data_collator,
            tokenizer=tokenizer,
        )

    trainer.train()
    trainer.save_model(f"NER_en")


# def train_POS_model(model_checkpoint):

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A simple script to demonstrate argument parsing.")
    parser.add_argument("task", help="the task to train an english model on")
    args = parser.parse_args()
    if args.task == "ner":
       train_NER_model("language_en")
    else:
        print("no task: ", args.task)
