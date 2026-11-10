"""课程行为测试；模型与 HTTP 的替身仅用于离线验证，不冒充真实调用。"""

import importlib
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


def course(name):
    try:
        return importlib.import_module('xuexi_lab.' + name)
    except ModuleNotFoundError as exc:
        raise AssertionError(f'课程模块尚未实现或依赖缺失：{name}') from exc


class ToolsTests(unittest.TestCase):
    def test_calculator_numeric_result_and_reject_code(self):
        tools = course('tools')
        self.assertEqual(tools.calculate('(18 + 24) / 2')['value'], 21)
        for expression in ('__import__("os")', 'True + 2', '2 ** 100000', 'float("inf")'):
            self.assertFalse(tools.calculate(expression)['ok'])
        self.assertEqual(tools.calculate('1 / 0')['error']['code'], 'division_by_zero')

    def test_database_exact_total_empty_and_injection(self):
        tools = course('tools')
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            tools.initialize_data(root)
            result = tools.query_expenses('all', root=root)
            self.assertEqual(result['amount'], '860.00')
            self.assertEqual(result['count'], 6)
            self.assertEqual(tools.query_expenses('不存在', root=root)['count'], 0)
            self.assertEqual(tools.query_expenses("差旅' OR 1=1 --", root=root)['count'], 0)
            self.assertEqual(tools.query_expenses('差旅', root=root)['amount'], '550.00')
            tools.initialize_data(root)
            self.assertEqual(tools.query_expenses('all', root=root)['count'], 6)

    def test_document_path_boundary_missing_and_utf8(self):
        tools = course('tools')
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            docs = root / 'data' / 'knowledge'
            docs.mkdir(parents=True)
            (docs / '规则.md').write_text('虚构教学规则', encoding='utf-8')
            (root / '.env').write_text('SECRET=never-read', encoding='utf-8')
            self.assertEqual(tools.read_document('规则.md', root=root)['text'], '虚构教学规则')
            self.assertFalse(tools.read_document('../../.env', root=root)['ok'])
            self.assertFalse(tools.read_document(str(root / '.env'), root=root)['ok'])
            self.assertEqual(tools.read_document('missing.md', root=root)['error']['code'], 'not_found')

    def test_weather_service_error_is_not_a_forecast(self):
        import httpx
        tools = course('tools')
        transport = httpx.MockTransport(lambda request: httpx.Response(503))
        result = tools.get_weather('成都', transport=transport)
        self.assertFalse(result['ok'])
        self.assertEqual(result['error']['code'], 'weather_http_error')
        self.assertFalse(tools.get_weather('不存在的城市')['ok'])

    def test_weather_timeout_is_reported_explicitly(self):
        import httpx
        tools = course('tools')

        def timeout(request):
            raise httpx.ReadTimeout('test timeout', request=request)

        result = tools.get_weather('成都', transport=httpx.MockTransport(timeout))
        self.assertFalse(result['ok'])
        self.assertEqual(result['error']['code'], 'weather_timeout')

    def test_weather_units_date_and_source(self):
        import httpx
        tools = course('tools')
        payload = {'timezone': 'Asia/Shanghai', 'daily_units': {
            'temperature_2m_min': '°C', 'temperature_2m_max': '°C'},
            'daily': {'time': ['2026-11-04'], 'temperature_2m_min': [10.0],
                      'temperature_2m_max': [20.0]}}
        transport = httpx.MockTransport(lambda request: httpx.Response(200, json=payload))
        result = tools.get_weather('成都', transport=transport)
        self.assertTrue(result['ok'])
        self.assertEqual(result['date'], '2026-11-04')
        self.assertEqual(result['unit'], '°C')
        self.assertEqual(result['source'], 'https://api.open-meteo.com/v1/forecast')


class ConfigAndTraceTests(unittest.TestCase):
    def test_shared_config_project_override_environment_and_redaction(self):
        config = course('config')
        with tempfile.TemporaryDirectory() as folder, patch.dict(os.environ):
            # 只隔离课程配置，不能清空 Windows/Conda 的 TLS 运行环境。
            for name in ('AGENT_BASE_URL', 'AGENT_MODEL', 'AGENT_API_KEY', 'AGENT_API_STYLE'):
                os.environ.pop(name, None)
            parent = Path(folder)
            root = parent / 'xuexi'
            root.mkdir()
            (parent / '.env').write_text('AGENT_BASE_URL=https://example.test/v1\nAGENT_MODEL=m\nAGENT_API_KEY=secret-test\n', encoding='utf-8')
            self.assertEqual(config.Settings.load(root).model, 'm')
            (root / '.env').write_text('AGENT_MODEL=project-model\n', encoding='utf-8')
            self.assertEqual(config.Settings.load(root).model, 'project-model')
            os.environ['AGENT_MODEL'] = 'env-model'
            settings = config.Settings.load(root)
            self.assertEqual(settings.model, 'env-model')
            self.assertNotIn('secret-test', repr(settings))
            trace = course('runtime').Trace(root, secrets=(settings.api_key,))
            trace.event('failure', message='a secret-test token')
            self.assertNotIn('secret-test', trace.path.read_text(encoding='utf-8'))

    def test_invalid_configuration_fails_before_network(self):
        config = course('config')
        with tempfile.TemporaryDirectory() as folder, patch.dict(os.environ):
            for name in ('AGENT_BASE_URL', 'AGENT_MODEL', 'AGENT_API_KEY', 'AGENT_API_STYLE'):
                os.environ.pop(name, None)
            with self.assertRaises(ValueError):
                config.Settings.load(Path(folder))


class RagAndEvalTests(unittest.TestCase):
    def test_preview_keeps_holdout_questions_and_answers_hidden(self):
        cli = course('cli')
        output = io.StringIO()
        with patch('sys.stdout', output):
            self.assertEqual(cli.main(['eval', '--preview', '--split', 'all']), 0)
        payload = json.loads(output.getvalue())
        holdout = [case for case in payload['cases'] if case['split'] == 'holdout']
        self.assertEqual(len(holdout), 4)
        self.assertTrue(all('question' not in case and 'expected' not in case for case in holdout))

    def test_retrieval_preserves_evidence_and_detects_changed_documents(self):
        rag = course('rag')
        from langchain_core.embeddings import Embeddings

        class KeywordEmbeddings(Embeddings):
            def embed_documents(self, texts):
                return [self.embed_query(text) for text in texts]

            def embed_query(self, text):
                return [float('报销' in text), float('网络' in text), 0.01]

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            docs = root / 'data' / 'knowledge'
            docs.mkdir(parents=True)
            (docs / '报销.md').write_text('报销必须提供发票。' * 20, encoding='utf-8')
            (docs / '网络.md').write_text('网络故障先检查路由器。' * 20, encoding='utf-8')
            first = rag.KnowledgeBase(root, embeddings=KeywordEmbeddings())
            result = first.search('报销发票', k=1)
            self.assertTrue(result['ok'])
            self.assertEqual(result['matches'][0]['source'], '报销.md')
            self.assertIn('chunk', result['matches'][0])
            fingerprint = first.fingerprint
            (docs / '报销.md').write_text('报销必须附上行程单。', encoding='utf-8')
            second = rag.KnowledgeBase(root, embeddings=KeywordEmbeddings())
            self.assertNotEqual(fingerprint, second.fingerprint)

    def test_missing_corpus_is_an_error(self):
        rag = course('rag')
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(ValueError):
                rag.KnowledgeBase(Path(folder), embeddings=object())

    def test_evaluation_requires_success_and_tool_evidence(self):
        evaluation = course('evaluation')
        case = {'id': 't', 'expected': ['21'], 'required_tools': ['calculate']}
        report = {'status': 'completed', 'answer': '21', 'tool_names': []}
        self.assertFalse(evaluation.grade(case, report)['auto_pass'])
        report['tool_names'] = ['calculate']
        self.assertTrue(evaluation.grade(case, report)['auto_pass'])
        report['status'] = 'api_error'
        self.assertFalse(evaluation.grade(case, report)['auto_pass'])

    def test_holdout_selection_and_unchanged_case_identity(self):
        evaluation = course('evaluation')
        root = Path(__file__).resolve().parents[1]
        dev = evaluation.load_cases(root, split='dev')
        holdout = evaluation.load_cases(root, split='holdout')
        self.assertEqual(len(dev), 8)
        self.assertEqual(len(holdout), 4)
        self.assertFalse({c['id'] for c in dev} & {c['id'] for c in holdout})
        self.assertEqual(len({c['id'] for c in dev + holdout}), 12)

    def test_compare_rejects_changed_business_data(self):
        evaluation = course('evaluation')
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            base = {'dataset_sha256': 'same-questions', 'model_fingerprint': 'same-model',
                    'api_style': 'chat_completions', 'data_fingerprint': 'original-data',
                    'rows': [{'case': {'id': 't'}}], 'auto_pass': 1, 'count': 1}
            before, after = root / 'before.json', root / 'after.json'
            before.write_text(json.dumps(base), encoding='utf-8')
            after.write_text(json.dumps({**base, 'data_fingerprint': 'modified-data'}), encoding='utf-8')
            with self.assertRaises(ValueError):
                evaluation.compare(before, after)


if __name__ == '__main__':
    unittest.main()
