"""所有命令以 python -m xuexi_lab 开始，适用于 Windows PowerShell。"""

import argparse
import json
import sys
from pathlib import Path

from .config import ROOT, Settings
from .tools import calculate, get_weather, initialize_data, list_documents, query_expenses, read_document

DAYS = {
    'day1': ('basic', '请必须调用 calculate 计算 (18+24)/2，并说明使用了什么工具和返回值。'),
    'day2': ('tools', '分别查询成都今天的日预报、差旅支出总额，并读取 02_报销材料.md。按三个部分回答并给出来源。'),
    'day3': ('plan', '读取 01_统计规则.md，查询可报销类别的实际金额，计算增加10%预留后的金额，给出CNY及依据。'),
    'day4': ('rag', '根据知识库，投影仪最长连续借用多少天？回答时引用实际文档名和 chunk 编号。'),
    'day5': ('collaborate', '根据知识库说明报销材料不完整时如何处理，引用文档名，不要编造审批或到账结果。'),
    'day7': ('final', '检索知识库中的可报销统计规则，查询各类别实际金额，计算加10%预留后的金额，给出CNY、原始数据及文档来源。'),
}


def emit(value):
    print(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))


def parser():
    root = argparse.ArgumentParser(description='LangChain 七天互动实战课')
    commands = root.add_subparsers(dest='command', required=True)
    commands.add_parser('init', help='创建虚构教学 SQLite 数据，不调用模型')
    commands.add_parser('doctor', help='验证真实模型工具循环，少量 API 调用')
    commands.add_parser('rag-build', help='下载/加载 CPU 多语言模型，建立教学知识库')
    search = commands.add_parser('search', help='独立运行本地语义检索，不调用对话模型')
    search.add_argument('--query', required=True)
    search.add_argument('--k', type=int, default=3, choices=range(1, 7))
    for day, (mode, _) in DAYS.items():
        item = commands.add_parser(day, help=f'{day} 默认练习，模式 {mode}')
        item.add_argument('--question')
        item.add_argument('--max-calls', type=int, default=8, choices=range(1, 9))
        item.add_argument('--variant', choices=['baseline', 'evidence_first'], default='baseline')
    run = commands.add_parser('run', help='指定模式与问题进行对照')
    run.add_argument('--mode', choices=['basic', 'tools', 'plan', 'rag', 'collaborate', 'final'], default='tools')
    run.add_argument('--question', required=True)
    run.add_argument('--max-calls', type=int, default=8, choices=range(1, 9))
    run.add_argument('--variant', choices=['baseline', 'evidence_first'], default='baseline')
    tool = commands.add_parser('tool', help='独立调用工具，不调用对话模型')
    tool.add_argument('name', choices=['calculate', 'weather', 'expenses', 'document', 'documents'])
    tool.add_argument('--expression', default='(18+24)/2')
    tool.add_argument('--city', default='成都')
    tool.add_argument('--category', default='all')
    tool.add_argument('--name', dest='document_name', default='01_统计规则.md')
    for name in ('eval', 'day6'):
        evaluation = commands.add_parser(name, help='固定题评测；默认只运行2题，--preview 不调用模型')
        evaluation.add_argument('--split', choices=['dev', 'holdout', 'all'], default='dev')
        evaluation.add_argument('--variant', choices=['baseline', 'evidence_first'], default='baseline')
        evaluation.add_argument('--limit', type=int, default=2, choices=range(1, 13))
        evaluation.add_argument('--preview', action='store_true')
    compare = commands.add_parser('compare', help='比较同数据、同模型、同题的两次评测')
    compare.add_argument('--before', required=True)
    compare.add_argument('--after', required=True)
    trace = commands.add_parser('trace', help='查看一次运行的本地证据')
    trace.add_argument('--path', required=True)
    return root


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        if args.command == 'init':
            initialize_data(ROOT)
            emit({'ok': True, 'teaching_data': True, 'database': str(ROOT / 'data' / 'expenses.sqlite'),
                  'note': '仅初始化虚构数据，没有调用模型；已有记录不覆盖。'})
        elif args.command == 'tool':
            actions = {'calculate': lambda: calculate(args.expression), 'weather': lambda: get_weather(args.city),
                       'expenses': lambda: query_expenses(args.category), 'document': lambda: read_document(args.document_name),
                       'documents': list_documents}
            result = actions[args.name]()
            emit(result)
            return 0 if result['ok'] else 1
        elif args.command in {'rag-build', 'search'}:
            from .rag import KnowledgeBase
            knowledge = KnowledgeBase(ROOT)
            emit(knowledge.search(args.query, k=args.k) if args.command == 'search' else
                 {'ok': True, 'chunks': len(knowledge.chunks), 'fingerprint': knowledge.fingerprint,
                  'note': '本地向量索引已建立；尚未验证对话模型问答。'})
        elif args.command in {'eval', 'day6'}:
            from .evaluation import evaluate, load_cases
            if args.preview:
                cases = load_cases(ROOT, args.split)
                preview = [case if case['split'] == 'dev' else
                           {key: case[key] for key in ('id', 'split', 'group')} for case in cases]
                emit({'calls_model': False, 'cases': preview,
                      'note': '保留题只显示编号和类别，问题和答案条件留到最终检查。'})
            else:
                result = evaluate(Settings.load(), split=args.split, variant=args.variant, limit=args.limit)
                emit(result)
                return 0 if result['auto_pass'] == result['count'] else 1
        elif args.command == 'compare':
            from .evaluation import compare
            emit(compare(args.before, args.after))
        elif args.command == 'trace':
            for line in Path(args.path).read_text(encoding='utf-8').splitlines():
                emit(json.loads(line))
        elif args.command == 'doctor':
            from .runtime import doctor
            result = doctor(Settings.load())
            emit(result)
            return 0 if result['ok'] else 1
        else:
            from .runtime import run_task
            if args.command in DAYS:
                mode, question = DAYS[args.command]
                question = args.question or question
            else:
                mode, question = args.mode, args.question
            report = run_task(Settings.load(), question, mode=mode, max_calls=args.max_calls, variant=args.variant)
            emit(report)
            return 0 if report['status'] == 'completed' else 1
        return 0
    except (ValueError, OSError, ImportError) as exc:
        # 配置错误消息由本项目生成；外部错误只公开类型，避免打印请求或凭据。
        message = str(exc) if isinstance(exc, ValueError) and not isinstance(exc, json.JSONDecodeError) else type(exc).__name__
        emit({'ok': False, 'error_type': type(exc).__name__, 'message': message})
        return 1


if __name__ == '__main__':
    sys.exit(main())
