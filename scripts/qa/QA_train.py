from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer, 
    AutoModelForQuestionAnswering,
    DefaultDataCollator,
    TrainingArguments, 
    Trainer
)
from trl import SFTTrainer, SFTConfig

import torch
from datasets import load_dataset, Dataset
import numpy as np
from torch import nn
import wandb

def train_QA_model(model_checkpoint):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, local_files_only=True)
    data_collator = DefaultDataCollator()
    # TODO train test split to get train, validation set, SET A SEED
    # We will test on XQuAD which comes from the validation set of SQuAD. Therefore, we run training and validation on the 
    # train set only of SQuAD to avoid data bleed.
    QA_dataset = load_dataset("rajpurkar/squad", "train", trust_remote_code=True)
    QA_dataset.train_test_split(test_size=0.1, seed=42)


    def preprocess(examples):
        questions = [q.strip() for q in examples["question"]]
        inputs = tokenizer(
            questions,
            examples["context"],
            max_length=384,
            truncation="only_second",
            return_offsets_mapping=True,
            padding="max_length",
        )

        offset_mapping = inputs.pop("offset_mapping")
        answers = examples["answers"]
        start_positions = []
        end_positions = []

        for i, offset in enumerate(offset_mapping):
            answer = answers[i]
            start_char = answer["answer_start"][0]
            end_char = answer["answer_start"][0] + len(answer["text"][0])
            sequence_ids = inputs.sequence_ids(i)

            # Find the start and end of the context
            idx = 0
            while sequence_ids[idx] != 1:
                idx += 1
            context_start = idx
            while sequence_ids[idx] == 1:
                idx += 1
            context_end = idx - 1

            # If the answer is not fully inside the context, label it (0, 0)
            if offset[context_start][0] > end_char or offset[context_end][1] < start_char:
                start_positions.append(0)
                end_positions.append(0)
            else:
                # Otherwise it's the start and end token positions
                idx = context_start
                while idx <= context_end and offset[idx][0] <= start_char:
                    idx += 1
                start_positions.append(idx - 1)

                idx = context_end
                while idx >= context_start and offset[idx][1] >= end_char:
                    idx -= 1
                end_positions.append(idx + 1)

        inputs["start_positions"] = start_positions
        inputs["end_positions"] = end_positions
        return inputs

    tokenized_dataset= QA_dataset.map(
        preprocess,
        batched=True,
        remove_columns=QA_dataset["train"].column_names
    )

    model = AutoModelForQuestionAnswering.from_pretrained(
        model_checkpoint,
        dtype=torch.float32,
        local_files_only=True,
    ).to(device)

    output_prefix = "xlm-roberta/base_finetuned" if "roberta" in model_checkpoint else "bert-multilingual/base_finetuned"
    training_args = TrainingArguments(
            output_dir=f"{output_prefix}/QA_en",
            eval_strategy="epoch",
            learning_rate=2e-5,
            num_train_epochs=3, 
            weight_decay=0.01,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            push_to_hub=False,
            save_strategy="epoch",
            report_to='wandb',
            project='xlt',
            run_name=f"{'roberta_' if 'roberta' in model_checkpoint else 'bert_'}QA_en",
            fp16=False
        )    
    trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset["train"],
            eval_dataset=tokenized_dataset["test"],
            data_collator=data_collator,
            tokenizer=tokenizer,
        )

    trainer.train()
    trainer.save_model(f"{output_prefix}/QA_en")
    wandb.finish()
def train_QA_model_causal(model_checkpoint):
    """
        parse dataset to be text-to-text
    """
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, trust_remote_code=True)
    
    model = AutoModelForCausalLM.from_pretrained(
        model_checkpoint,
        dtype=torch.float32,
        device_map="auto"
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id
   
    QA_dataset = load_dataset("rajpurkar/squad", "train", trust_remote_code=True)
    QA_dataset.train_test_split(test_size=0.1, seed=42)   
    train_data = []
    #TODO EMMA HERE:
    for datum in QA_dataset["train"]:
        train_data.append(
            {
                "text": (
                    f"Context: {datum['context']}\n. Question: {datum['question']}\n Answer: {datum['answers']['text'][0]}"
                )
            }
        )
    validation_data = []
    for datum in QA_dataset["test"]:
        validation_data.append(
            {
                "text": (
                    f"Context: {datum['context']}\n. Question: {datum['question']}\n Answer: {datum['answers']['text'][0]}"
                )
            }
        )
    train_dataset = Dataset.from_list(train_data)
    validation_dataset = Dataset.from_list(validation_data)
    if "granite" in model_checkpoint:
        output_prefix = "granite/base_finetuned"
    else:
        output_prefix = "qwen/base_finetuned"

    training_args = SFTConfig(
            output_dir=f"{output_prefix}/QA_en",
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
            run_name=f"{'qwen_' if 'qwen' in model_checkpoint else 'granite_'}QA_en",
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
    trainer.save_model(f"{output_prefix}/NLI_en")
    wandb.finish()

if __name__ == "__main__":
    roberta = "FacebookAI/xlm-roberta-base"
    bert = "google-bert/bert-base-multilingual-cased"
    qwen = "Qwen/Qwen3-0.6B"
    granite = "ibm-granite/granite-4.0-350m"
    for i, model in enumerate([roberta, bert, qwen, granite]):
        if i < 2:
            train_QA_model(model)
        else:
            train_QA_model_causal(model)
