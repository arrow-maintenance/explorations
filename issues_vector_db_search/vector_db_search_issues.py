import pandas as pd
from qdrant_client import models, QdrantClient
from sentence_transformers import SentenceTransformer
import re

import gzip
import json
import pandas as pd

# Here, we load the data. In future we should pre-filter properly on PR versus issue or just fetch from API instead 
# of filtering here, but this is working with our issues object for now

with gzip.open("test_data/issues_min.json.gz", "rt", encoding="utf-8") as f:
    df = json.load(f)
    
df = pd.DataFrame(df)
data = df.to_dict('records')

# Data transformation and cleaning
non_empty = [x for x in data if len(x['body']) > 1]
no_prs = [x for x in non_empty if "### What changes are included in this PR?" not in x.get('body', '')]

phrases_to_remove = [
    "### Describe the enhancement requested",
    "### Describe the bug, including details regarding any error messages, version, and platform.",
    "### Component(s)",
    "### Describe the usage question you have. Please include as many useful details as  possible."
]

def remove_code_chunks(text):
    # Remove fenced code blocks (```...```)
    text = re.sub(r"```.*?\n.*?```", "", text, flags=re.DOTALL)
    
    # Remove inline code (`...`)
    text = re.sub(r"`[^`]*`", "", text)
    
    # Remove indented code blocks (lines starting with 4+ spaces or a tab)
    text = re.sub(r"^(?: {4,}|\t).*\n?", "", text, flags=re.MULTILINE)

    return text

def remove_urls(text):
    return re.sub(r'(https?://\S+|www\.\S+|ftp://\S+)', '', text)




for x in no_prs:
    body = x.get('body', '')
    if not isinstance(body, str):
        continue  # or set x['body'] = "" if you prefer
    for phrase in phrases_to_remove:
        body = body.replace(phrase, '')
    x['body'] = "\n".join(line for line in body.splitlines() if line.strip())
    x['body'] = remove_code_chunks(x['body'])
    x['body'] = remove_urls(x.get('body', ''))


for x in no_prs[11:50]:
    print(x['body'])
    print("*************************************************")


# Vector DB

encoder = SentenceTransformer('all-MiniLM-L6-v2', device='cpu') # Model to create embeddings

qdrant = QdrantClient(":memory:") # Create in-memory Qdrant instance

# Create collection to store books
qdrant.recreate_collection(
    collection_name="arrow_issues",
    vectors_config=models.VectorParams(
        size=encoder.get_sentence_embedding_dimension(), # Vector size is defined by used model
        distance=models.Distance.COSINE
    )
)


qdrant.upload_points(
    collection_name="arrow_issues",
    points=[
        models.PointStruct(
            id=idx,
            vector=encoder.encode(doc["body"]).tolist(),
            payload=doc
        ) for idx, doc in enumerate(non_empty) 
    ]
)

hits = qdrant.search(
    collection_name="arrow_issues",
    query_vector=encoder.encode("decimal").tolist(),
    limit=10
)

for hit in hits:
  print(hit.payload['body'], "\nscore:", hit.score, "\n", hit.payload['url'])
  print("\n")


