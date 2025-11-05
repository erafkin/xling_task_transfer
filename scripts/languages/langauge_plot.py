from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np
import umap
from scripts.task_vectors import TaskVector
from transformers import AutoModelForMaskedLM
import matplotlib.pyplot as plt

scaler = StandardScaler()

pca = PCA(n_components=2)
bert_model = "google-bert/bert-base-multilingual-uncased"
roberta_model = "FacebookAI/xlm-roberta-base"

bert_vectors = []
roberta_vectors = []
lang = ["en", "es", "de", "hi", "zh"]
for l in lang:
    bert_vectors.append(TaskVector(pretrained_model=AutoModelForMaskedLM.from_pretrained(bert_model),                           
                       finetuned_model=AutoModelForMaskedLM.from_pretrained(f"bert-multilingual/language_{l}_done", local_files_only=True)))

data = [v.tv_to_vector() for v in bert_vectors]
reducer = umap.UMAP(n_neighbors=2, min_dist=0.1, n_components=2, random_state=42)
embedding = reducer.fit_transform(data)
plt.figure(figsize=(10, 8))
fig, ax = plt.subplots()
ax.scatter(embedding[:, 0], embedding[:, 1]) 
for i, txt in enumerate(lang):
    ax.annotate(txt, (embedding[:, 0][i], embedding[:, 1][i]))
plt.title('BERT Language Vectors')
plt.savefig('output/languages/umap_plot.png', dpi=300, bbox_inches='tight') # Saves as PNG, 300 DPI, tight bounding box
plt.close()