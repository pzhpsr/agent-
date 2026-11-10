from math import sqrt
from pathlib import Path

from langchain_huggingface import HuggingFaceEmbeddings


ROOT = Path(__file__).resolve().parents[1]
MODEL = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
REVISION = 'e8f8c211226b894fcb81acc59f3b34ba3efd5f42'


def cosine_similarity(left, right):
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = sqrt(sum(value * value for value in left))
    right_norm = sqrt(sum(value * value for value in right))
    return dot / (left_norm * right_norm)


embeddings = HuggingFaceEmbeddings(
    model_name=MODEL,
    cache_folder=str(ROOT / 'models'),
    model_kwargs={'device': 'cpu', 'revision': REVISION},
    encode_kwargs={'normalize_embeddings': True},
)

query = '投影仪最多借用几天？'
texts = [
    '教学规则中投影仪最长连续借用 3 天，延期需人工确认。',
    '差旅报销需要发票与行程说明。',
]

query_vector = embeddings.embed_query(query)
text_vectors = embeddings.embed_documents(texts)

print(f'查询向量长度：{len(query_vector)}')
print(f'查询向量前 8 个数字：{[round(value, 4) for value in query_vector[:8]]}')
print()

for text, vector in zip(texts, text_vectors):
    score = cosine_similarity(query_vector, vector)
    print(f'相似度：{score:.4f} | 文字：{text}')
