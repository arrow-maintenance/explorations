import pandas as pd
from qdrant_client import models, QdrantClient
from sentence_transformers import SentenceTransformer
import re
from sentence_transformers import SentenceTransformer
import itertools


# load data
with gzip.open("test_data/issues_min.json.gz", "rt", encoding="utf-8") as f:
    df = json.load(f)
  
df = pd.DataFrame(data)

data = df.to_dict('records')

just_open = [x for x in data if x['state'] == "open"]

non_prs = [x for x in just_open if (len(x["pull_request"]) == 0 and (not x["url"].startswith("https://github.com/apache/arrow/pull/")))]

len(non_prs)

just_titles = [x['title'] for x in non_prs]
len(just_titles)

model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(just_titles, convert_to_tensor=True)

# Work out cosine similarity

import faiss
import numpy as np

# Convert to numpy float32
emb_np = embeddings.cpu().numpy().astype('float32')
faiss.normalize_L2(emb_np)

index = faiss.IndexFlatIP(emb_np.shape[1])  # Inner product = cosine if normalized
index.add(emb_np)

D, I = index.search(emb_np, 5)  # top 5 neighbors per issue

duplicates = []


for idx, (neighbors, sims) in enumerate(zip(I, D)):
    for j, sim in zip(neighbors[1:], sims[1:]):  # skip self
        if sim > 0.85:
            duplicates.append((idx, j, sim))

duplicates = sorted(duplicates, key=lambda x: -x[2])

seen = set()
html_snippets = []

for idx1, idx2, sim in duplicates:
    # Skip if either issue already used
    if idx1 in seen or idx2 in seen:
        continue

    issue1 = non_prs[int(idx1)]
    issue2 = non_prs[int(idx2)]

    block = f"""
<br>
<b>Issue:</b> <a target="blank" href="{issue1['url']}">{issue1['title']}</a><br>
<b>Duplicate:</b> <a target="blank" href="{issue2['url']}">{issue2['title']}</a><br>
<b>Score:</b> {sim:.3f}<br>
<br>
"""
    html_snippets.append(block)

    # Mark both as seen so we don’t include them again
    seen.add(idx1)
    seen.add(idx2)

html_output = "\n".join(html_snippets)

with open("issue_duplication_detection/duplicates.html", "w", encoding="utf-8") as f:
    f.write(html_output)

    
