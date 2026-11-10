from pathlib import Path

from xuexi_lab.rag import KnowledgeBase


ROOT = Path(__file__).resolve().parents[1]
knowledge = KnowledgeBase(ROOT)

print(f'向量库中的记录数：{len(knowledge.store.store)}')
print()

for item in knowledge.store.store.values():
    metadata = item['metadata']
    print(f"来源：{metadata['source']}")
    print(f"片段编号：{metadata['chunk']}")
    print(f"向量维度：{len(item['vector'])}")
    print(f"原文：{item['text']}")
    print('-' * 40)
