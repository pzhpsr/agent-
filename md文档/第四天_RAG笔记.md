# 第四天：RAG 检索增强生成

## 1. RAG 的完整流程

RAG（Retrieval-Augmented Generation，检索增强生成）让模型先从外部知识中检索相关文本，再根据检索到的证据生成回答。它不等于模型已经“学会”了这些文档内容。

### 1.1 文档处理：分块、嵌入与向量存储

知识库流程：

```text
文档 → 分块 → 嵌入向量 → 向量存储
```

- **分块（chunking）**：把文档拆成较短片段，并保留来源和片段编号。
- **嵌入（embedding）**：把文本转换为数字向量，供语义检索比较。
- **向量存储**：保存文本块及其向量，供后续搜索。

分块器按空行、换行、句号等分隔符依次尝试切分。`chunk_size` 是长度上限，`chunk_overlap` 控制相邻块的重叠，不保证每块都刚好达到上限。创建分块和内存向量库的核心代码：

```python
splitter = RecursiveCharacterTextSplitter(
    chunk_size=chunk_size,
    chunk_overlap=chunk_overlap,
    separators=['\n\n', '\n', '。', '；', ' ', ''],
)
self.chunks = splitter.split_documents(documents)
self.store = InMemoryVectorStore.from_documents(self.chunks, embeddings)
```

### 1.2 问题处理：检索证据并生成回答

查询流程：

```text
问题 → 查询向量 → 相似片段 → 模型读取证据 → 回答并引用来源
```

查询向量与文档向量用于排序相似片段。检索结果是候选证据，模型仍须判断原文是否真正支持答案。

## 2. 知识库与检索

### 2.1 chunk_size、缓存和 fingerprint

本次对照中，6 份原有知识文档在 `chunk_size=240` 时共切成 **6 块**，改为 `120` 后共 **15 块**。较短的上限让原先完整的一份文档被拆成多个片段；实际块长还受分隔符影响。

项目的向量缓存位于 `runs/embedding_cache/<模型编号>/`。缓存键由文本计算哈希：文本已有缓存时读取对应 JSON，文本未缓存时才调用嵌入模型并保存。因此，改变分块可能产生新缓存；旧缓存仍留在磁盘，但不等于被加入当前向量库。每次运行会重新构建内存向量库。

`fingerprint` 概括知识文档内容、分块参数及嵌入模型版本。新增教学文档后，检索输出中的 fingerprint 从 `812826…` 变为 `e84c72…`，来源列表也出现了新文档。

`fingerprint` 的核心是对文档内容、分块参数和嵌入模型版本计算哈希：

```python
self.fingerprint = hashlib.sha256(
    (identity + str((chunk_size, chunk_overlap)) + EMBEDDING_MODEL + EMBEDDING_REVISION).encode()
).hexdigest()
```

缓存按文本哈希命名；命中时读 JSON，未命中时才调用嵌入模型并保存向量：

```python
for text in texts:
    path = self.cache / (hashlib.sha256(text.encode()).hexdigest() + '.json')
    if path.is_file():
        vector = json.loads(path.read_text(encoding='utf-8'))
    else:
        vector = self.backend.embed_documents([text])[0]
        path.write_text(json.dumps(vector), encoding='utf-8')
    results.append(vector)
```

### 2.2 相似度排序、k 与检索结果的局限

`k` 指返回排名前几的片段；例如 `k=3` 返回至多 3 个候选片段。相似度分数用于排序，**不是概率，也不是答案正确率**。检索会给出最相似的片段，即使它们没有答案。

新增的 `07课程安排.md` 说明 RAG-204 的实验主题是文本分块。查询“rag是什么？”时，该文档排第一，但内容并没有定义 RAG。这说明“排名第一”或“主题有些相关”不等于“证据充分”。

## 3. LangChain 中的 RAG 实现

### 3.1 KnowledgeBase 如何构建和检索知识库

`KnowledgeBase` 读取 `data/knowledge` 下的 Markdown/TXT 文件，生成文档对象和来源元数据，完成分块、嵌入与内存向量存储。`search()` 调用 LangChain 向量库的相似度搜索，返回来源、片段编号、分数和原文：

```python
matches = self.store.similarity_search_with_score(query, k=k)
return {
    'ok': True,
    'fingerprint': self.fingerprint,
    'matches': [
        {
            'source': document.metadata['source'],
            'chunk': document.metadata['chunk'],
            'score': round(float(score), 5),
            'text': document.page_content,
        }
        for document, score in matches
    ],
}
```

命令行的 `search` 会建立本地知识库并执行检索；它不调用对话模型。`day4` 则进入 RAG 模式，由运行器建立知识库、注册工具并调用模型。

### 3.2 search_knowledge 与 read_document 如何协作

`search_knowledge` 把问题交给 `KnowledgeBase.search()`；`read_document` 根据文件名读取知识目录中的文档全文。注册这两个工具的核心代码：

```python
@tool('read_document')
def document(name: str) -> dict:
    return read_document(name, root=root)

@tool('search_knowledge')
def search(query: str, k: int = 3) -> dict:
    return knowledge.search(query, k=k)
```

运行器创建 Agent 并调用 `invoke`；模型请求工具后，LangChain 执行工具并将结果放回消息，再继续调用模型：

```python
agent = create_agent(selected_model, tools, system_prompt=prompt + '\n' + extra, name=stage)
result = agent.invoke(
    {'messages': [{'role': 'user', 'content': task}]},
    config={'callbacks': [evidence], 'recursion_limit': 24},
)
```

创建 Agent 本身不是一次模型请求；Trace 中的 `model_request` 才记录实际请求。

## 4. 根据证据判断回答是否可靠

### 4.1 投影仪采购价格：无答案案例

手动搜索“投影仪采购价是多少？”时，前三个结果都没有采购价格。`day4` 内部实际使用的问题文字和 `k` 不同：它检索“投影仪采购价格”，取 5 个候选片段；其中最高分是《02_报销材料.md》的标题（`0.43406`），而《03_设备借用.md》排第三（`0.33279`），其中提到了投影仪借用规则。

Trace 显示模型随后调用 `read_document` 读取《03_设备借用.md》全文，最终回答“知识库未提供投影仪采购价格”。这次运行共 **3 次模型请求、2 次工具调用**：先 `search_knowledge`，再 `read_document`，最后生成答案。全文没有价格证据，因此回答没有编造价格。

Trace 能证明调用顺序和工具输入输出，但不能证明模型内部具体如何权衡候选片段。较稳妥的说法是：模型在检索后读取了包含“投影仪”的文档，并据得到的证据作答。

### 4.2 区分检索相关、证据充分与回答正确

- **检索相关**：片段与问题主题有联系，例如设备借用片段提到了投影仪。
- **证据充分**：片段明确包含问题所需事实；借用天数不能证明采购价格。
- **回答正确**：最终结论不超出证据支持范围；缺少价格依据时应说知识库未提供。

排查 RAG 答案时，应阅读实际命中的 `text`，核对 `source`、`chunk` 和问题需要的事实，而不能只看分数或最终答案。
