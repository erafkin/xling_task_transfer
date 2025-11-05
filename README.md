# Crosslingual Task Transfer
This code explores the success of cross lingual task transfer using [Task Vectors](https://arxiv.org/abs/2212.04089) across different linguistic tasks and different languages. 

Multilingual transformers were trained on each target language and then the base transformer was trained to perform a task (NER, POS tagging, NLI, or Dependency Parsing). Success was evaluated across tasks and crosslinguistically. It was hypothesized that task vectors would perform better for semantic tasks than syntactic tasks, and would perform better on languages more similar to the source language. 

The base transformers tested were [`xlm-roberta-base`](https://huggingface.co/FacebookAI/xlm-roberta-base) and [`bert-base-multilingual-uncased`](https://huggingface.co/google-bert/bert-base-multilingual-uncased). These were either finetuned or trained from scratch on [Wikipedia data](https://huggingface.co/datasets/wikimedia/wikipedia). Source language was always English, target languages were Spanish, German, Hindi, and Chinese.

## Setup
Developed in Python 3.11
```sh
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Repository Structure 
Code is broken down for training and testing each task (NLI, NER, POS, and DP (tbd)) and also for language finetuning. Each script also has a slurm script for running on an HPC. 

## Datasets: 
| Task | Dataset | 
|----------|----------|
| Language Finetuning| [Wikipedia Dump](https://huggingface.co/datasets/wikimedia/wikipedia)
| NER | [MultiCoNER V2](https://huggingface.co/datasets/MultiCoNER/multiconer_v2) | Data 3A  |
| NLI  | [XNLI](https://huggingface.co/datasets/facebook/xnli)  | Data 3B  |
|POS tagging| [Universal Dependencies](https://universaldependencies.org/)
|Dependency Parsing| [Universal Dependencies](https://universaldependencies.org/)

## Author
Emma Rafkin
epr41@georgetown.edu
