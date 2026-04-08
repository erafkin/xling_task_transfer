from scripts.task_vectors import TaskVector
from transformers import AutoModelForMaskedLM, AutoModelForCausalLM
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd

rows = []

bert_model = "google-bert/bert-base-multilingual-cased"
roberta_model = "FacebookAI/xlm-roberta-base"
qwen_model = "Qwen/Qwen3-0.6B"
granite_model = "ibm-granite/granite-4.0-350m"

bert_vectors = []
roberta_vectors = []
qwen_vectors = []
granite_vectors = []
lang = ["en", "es", "de", "hi", "zh", "fr", "ru"]
old_df = pd.read_csv("output/languages/english_cosims.csv")
for l in lang:
    # bert_vectors.append(TaskVector(pretrained_model=AutoModelForMaskedLM.from_pretrained(bert_model),                           
    #                    finetuned_model=AutoModelForMaskedLM.from_pretrained(f"bert/language_{l}_done", local_files_only=True)))
    # roberta_vectors.append(TaskVector(pretrained_model=AutoModelForMaskedLM.from_pretrained(roberta_model),                           
    #                    finetuned_model=AutoModelForMaskedLM.from_pretrained(f"xlm/language_{l}_done", local_files_only=True)))
    # qwen_vectors.append(TaskVector(pretrained_model=AutoModelForCausalLM.from_pretrained(qwen_model),                           
    #                    finetuned_model=AutoModelForCausalLM.from_pretrained(f"qwen/language_{l}_done", local_files_only=True)))
    granite_vectors.append(TaskVector(pretrained_model=AutoModelForCausalLM.from_pretrained(granite_model),                           
                       finetuned_model=AutoModelForCausalLM.from_pretrained(f"granite/language_{l}_done", local_files_only=True)))

# for idx, vec in enumerate(bert_vectors[1:]):
#     cossim = cosine_similarity(bert_vectors[0].tv_to_vector().reshape(1, -1), vec.tv_to_vector().reshape(1, -1))[0][0]
#     rows.append(["bert", lang[idx + 1], cossim])

# for idx, vec in enumerate(roberta_vectors[1:]):
#     cossim = cosine_similarity(roberta_vectors[0].tv_to_vector().reshape(1, -1), vec.tv_to_vector().reshape(1, -1))[0][0]
#     rows.append(["roberta", lang[idx + 1], cossim])

# for idx, vec in enumerate(qwen_vectors[1:]):
#     cossim = cosine_similarity(qwen_vectors[0].tv_to_vector().reshape(1, -1), vec.tv_to_vector().reshape(1, -1))[0][0]
#     rows.append(["qwen", lang[idx + 1], cossim])


for idx, vec in enumerate(granite_vectors[1:]):
    cossim = cosine_similarity(granite_vectors[0].tv_to_vector().reshape(1, -1), vec.tv_to_vector().reshape(1, -1))[0][0]
    rows.append(["granite", lang[idx + 1], cossim])

new_df = pd.DataFrame(rows, columns=["model", "langauge", "en_cossim"])
df_combined = pd.concat([old_df, new_df], ignore_index=True)
# df = pd.DataFrame(rows, columns=["model", "langauge", "en_cossim"])
df_combined.to_csv("output/languages/english_cosims_new.csv", index=False)