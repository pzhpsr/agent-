"""D4：看得见的分块、嵌入、向量存储和检索。"""

import hashlib
import json
import os
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import ROOT

EMBEDDING_MODEL = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
EMBEDDING_REVISION = 'e8f8c211226b894fcb81acc59f3b34ba3efd5f42'


class CachedEmbeddings(Embeddings):
    def __init__(self, backend, cache):
        self.backend = backend
        self.cache = Path(cache)
        self.cache.mkdir(parents=True, exist_ok=True)

    def embed_documents(self, texts):
        results = []
        for text in texts:
            path = self.cache / (hashlib.sha256(text.encode()).hexdigest() + '.json')
            if path.is_file():
                vector = json.loads(path.read_text(encoding='utf-8'))
            else:
                vector = self.backend.embed_documents([text])[0]
                path.write_text(json.dumps(vector), encoding='utf-8')
            results.append(vector)
        return results

    def embed_query(self, text):
        # 当前后端对 query/document 不加不同前缀，因此可以共用缓存。
        return self.embed_documents([text])[0]


class KnowledgeBase:
    def __init__(self, root=ROOT, *, embeddings=None, chunk_size=240, chunk_overlap=40):
        self.root = Path(root)
        base = self.root / 'data' / 'knowledge'
        paths = sorted(p for p in base.glob('*') if p.is_file() and p.suffix in {'.md', '.txt'})
        if not paths:
            raise ValueError('知识目录中没有 Markdown/TXT 文档。')
        documents = [Document(page_content=p.read_text(encoding='utf-8-sig'),
                              metadata={'source': p.name}) for p in paths]
        identity = json.dumps([(d.metadata['source'], d.page_content) for d in documents], ensure_ascii=False)
        self.fingerprint = hashlib.sha256((identity + str((chunk_size, chunk_overlap)) + EMBEDDING_MODEL + EMBEDDING_REVISION).encode()).hexdigest()
        splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap,
                                                  separators=['\n\n', '\n', '。', '；', ' ', ''])
        self.chunks = splitter.split_documents(documents)
        for index, document in enumerate(self.chunks):
            document.metadata['chunk'] = index
        if embeddings is None:
            # 使用普通 HTTPS 下载同一固定版本的权重，避免额外 Xet 缓存及传输层。
            os.environ.setdefault('HF_HUB_DISABLE_XET', '1')
            os.environ.setdefault('HF_XET_CACHE', str(self.root / 'models' / 'xet'))
            from langchain_huggingface import HuggingFaceEmbeddings
            backend = HuggingFaceEmbeddings(
                model_name=EMBEDDING_MODEL, cache_folder=str(self.root / 'models'),
                model_kwargs={'device': 'cpu', 'revision': EMBEDDING_REVISION},
                encode_kwargs={'normalize_embeddings': True})
            model_id = hashlib.sha256((EMBEDDING_MODEL + EMBEDDING_REVISION).encode()).hexdigest()[:16]
            embeddings = CachedEmbeddings(backend, self.root / 'runs' / 'embedding_cache' / model_id)
        self.store = InMemoryVectorStore.from_documents(self.chunks, embeddings)

    def search(self, query, k=3):
        if not isinstance(query, str) or not query.strip() or not 1 <= k <= 6:
            return {'ok': False, 'error': {'code': 'invalid_query', 'message': 'query 非空，k 在 1–6 之间。'}}
        matches = self.store.similarity_search_with_score(query, k=k)
        return {'ok': True, 'fingerprint': self.fingerprint, 'matches': [
            {'source': document.metadata['source'], 'chunk': document.metadata['chunk'],
             'score': round(float(score), 5), 'text': document.page_content}
            for document, score in matches],
            'note': '相似度只是排序依据，不是答案正确率；必须阅读证据判断是否支持问题。'}
