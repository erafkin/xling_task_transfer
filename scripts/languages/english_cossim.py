from scripts.task_vectors import TaskVector
from transformers import AutoModelForMaskedLM
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd

rows = []

bert_model = "google-bert/bert-base-multilingual-cased"
roberta_model = "FacebookAI/xlm-roberta-base"

bert_vectors = []
roberta_vectors = []
lang = ["en", "es", "de", "hi", "zh", "fr", "ru"]
for l in lang:
    bert_vectors.append(TaskVector(pretrained_model=AutoModelForMaskedLM.from_pretrained(bert_model),                           
                       finetuned_model=AutoModelForMaskedLM.from_pretrained(f"bert/language_{l}_done", local_files_only=True)))
    roberta_vectors.append(TaskVector(pretrained_model=AutoModelForMaskedLM.from_pretrained(roberta_model),                           
                       finetuned_model=AutoModelForMaskedLM.from_pretrained(f"xlm/language_{l}_done", local_files_only=True)))

for idx, vec in enumerate(bert_vectors[1:]):
    cossim = cosine_similarity(bert_vectors[0].tv_to_vector().reshape(1, -1), vec.tv_to_vector().reshape(1, -1))[0][0]
    rows.append(["bert", lang[idx + 1], cossim])

for idx, vec in enumerate(roberta_vectors[1:]):
    cossim = cosine_similarity(roberta_vectors[0].tv_to_vector().reshape(1, -1), vec.tv_to_vector().reshape(1, -1))[0][0]
    rows.append(["roberta", lang[idx + 1], cossim])

df = pd.DataFrame(rows, columns=["model", "langauge", "en_cossim"])
df.to_csv("output/languages/english_cosims.csv", index=False)