# Crosslingual Task Transfer
This code explores the success of cross lingual task transfer using [Task Vectors](https://arxiv.org/abs/2212.04089) across different linguistic tasks and different languages. 

Multilingual transformers were trained on each target language and then the base transformer was trained to perform a task (NER, POS tagging, NLI, Dependency Parsing, Question-Answering, and Morphological Inflection). Success was evaluated across tasks and crosslinguistically. It was hypothesized that task vectors would perform better for semantic tasks than syntactic tasks, and would perform better on languages more similar to the source language. 

The base transformers tested were [`xlm-roberta-base`](https://huggingface.co/FacebookAI/xlm-roberta-base), [`bert-base-multilingual-cased`](https://huggingface.co/google-bert/bert-base-multilingual-cased), [`qwen3`](https://huggingface.co/Qwen/Qwen3-0.6B), and [`granite4`](https://huggingface.co/ibm-granite/granite-4.0-350m). These were finetuned on [Wikipedia data](https://huggingface.co/datasets/wikimedia/wikipedia). Source language was always English, target languages were Spanish, German, Hindi, Chinese, Russian, and French.

Through this experiment I found that XLT is more successful across semantic tasks vs syntactic tasks. Additionally, I found that there is a correlation between source and target language distance and XLT accuracy loss--the transfer degrades as the languages become more distant. Finally, I find that the cosine similarity between the target language vectors and the source language vector (English) is correlated to theoretically-based metrics of language distance. 

## Setup
Developed in Python 3.11
```sh
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Repository Structure 
Code is broken down for training and testing each task (NLI, NER, POS, DP, QA, and MI) and also for language finetuning. Each script also has a slurm script for running on an HPC. 

Results for each model as well as the visualizations used in the paper are in the `output` folder.
## Datasets: 
| Task | Dataset | 
|----------|----------|
| Language Finetuning | [Wikipedia Dump](https://huggingface.co/datasets/wikimedia/wikipedia)|
| NER | [MultiCoNER V2](https://huggingface.co/datasets/MultiCoNER/multiconer_v2) | 
| NER | [UniversalNER](https://www.universalner.org/) | 
| NLI | [XNLI](https://huggingface.co/datasets/facebook/xnli)  | 
|POS tagging| [Universal Dependencies](https://universaldependencies.org/)
|Dependency Parsing| [Universal Dependencies](https://universaldependencies.org/)
|Question Answering| [SQuAD](https://huggingface.co/datasets/rajpurkar/squad)
|Question Answering| [xQuAD](https://huggingface.co/datasets/google/xquad)
|Question Answering| [newsQuADfr](https://huggingface.co/datasets/lincoln/newsquadfr)
|Morphological Inflection| [Unimorph](https://unimorph.github.io/)



## Author
Emma Rafkin
epr41@georgetown.edu
