"""D6：固定案例、透明的自动断言，以及单独的人工语义核对。"""

import hashlib
import json
import sqlite3
import uuid
from contextlib import closing
from pathlib import Path

from .config import ROOT


def load_cases(root=ROOT, split='dev'):
    cases = json.loads((Path(root) / 'data' / 'eval_cases.json').read_text(encoding='utf-8'))
    return [case for case in cases if split == 'all' or case['split'] == split]


def data_fingerprint(root, knowledge=None):
    """记录实际数据内容；不读取 .env，也不依赖数据库文件的修改时间。"""
    root = Path(root)
    documents = [(path.name, path.read_text(encoding='utf-8-sig')) for path in
                 sorted((root / 'data' / 'knowledge').glob('*'))
                 if path.is_file() and path.suffix in {'.md', '.txt'}]
    database = root / 'data' / 'expenses.sqlite'
    rows = None
    if database.is_file():
        with closing(sqlite3.connect(database.resolve().as_uri() + '?mode=ro', uri=True)) as connection:
            rows = connection.execute('SELECT id,category,cents,note FROM expenses ORDER BY id').fetchall()
    identity = {'documents': documents, 'expenses': rows,
                'rag_fingerprint': knowledge.fingerprint if knowledge is not None else None}
    return hashlib.sha256(json.dumps(identity, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def grade(case, report):
    missing_text = [value for value in case.get('expected', []) if value not in report.get('answer', '')]
    missing_tools = sorted(set(case.get('required_tools', [])) - set(report.get('tool_names', [])))
    missing_stages = sorted(set(case.get('required_stages', [])) - set(report.get('stages', [])))
    return {'id': case['id'], 'auto_pass': report.get('status') == 'completed' and not (missing_text or missing_tools or missing_stages),
            'missing_text': missing_text, 'missing_tools': missing_tools, 'missing_stages': missing_stages,
            'manual_review': 'pending', 'note': '自动断言只检查基本条件，不等于答案语义正确或证据充分。'}


def evaluate(settings, *, split='dev', variant='baseline', limit=None):
    from .runtime import run_task, resolved_style
    cases = load_cases(settings.root, split)
    if limit is not None:
        cases = cases[:limit]
    knowledge = None
    if any(case['mode'] in {'rag', 'collaborate', 'final'} for case in cases):
        from .rag import KnowledgeBase
        knowledge = KnowledgeBase(settings.root)
    data_hash = data_fingerprint(settings.root, knowledge)
    style = resolved_style(settings)
    rows = []
    for case in cases:
        report = run_task(settings, case['question'], mode=case['mode'], knowledge=knowledge, variant=variant)
        rows.append({'case': case, 'report': report, 'grade': grade(case, report)})
    directory = settings.root / 'runs' / 'evaluations'
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / (variant + '-' + uuid.uuid4().hex[:10] + '.json')
    dataset_hash = hashlib.sha256((settings.root / 'data' / 'eval_cases.json').read_bytes()).hexdigest()
    payload = {'variant': variant, 'split': split, 'dataset_sha256': dataset_hash,
               'model_fingerprint': settings.fingerprint, 'api_style': style,
               'data_fingerprint': data_hash, 'count': len(rows),
               'auto_pass': sum(row['grade']['auto_pass'] for row in rows), 'manual_review': 'pending', 'rows': rows}
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    return {'path': str(path), 'count': payload['count'], 'auto_pass': payload['auto_pass'], 'manual_review': 'pending'}


def compare(before, after):
    records = [json.loads(Path(path).read_text(encoding='utf-8')) for path in (before, after)]
    left, right = records
    identity = lambda record: [row['case']['id'] for row in record['rows']]
    if (left['dataset_sha256'] != right['dataset_sha256'] or identity(left) != identity(right)
            or left['model_fingerprint'] != right['model_fingerprint']
            or not left.get('data_fingerprint') or left['data_fingerprint'] != right.get('data_fingerprint')
            or not left.get('api_style') or left['api_style'] != right.get('api_style')):
        raise ValueError('对照必须使用相同题目及顺序、相同实际数据/分块设置、相同模型配置和接口风格。')
    return {'before_auto_pass': left['auto_pass'], 'after_auto_pass': right['auto_pass'],
            'count': left['count'], 'manual_review': '仍需逐题核对语义与证据，不能仅看自动得分。'}
