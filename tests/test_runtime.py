"""真实 LangChain 执行循环的离线测试，使用显式标记的脚本模型。"""

import tempfile
import unittest
from pathlib import Path

from test_course import course


class RuntimeTests(unittest.TestCase):
    def make_model(self, messages):
        from langchain_core.language_models.chat_models import BaseChatModel
        from langchain_core.outputs import ChatGeneration, ChatResult
        from pydantic import Field

        class ScriptedModel(BaseChatModel):
            responses: list = Field(default_factory=list)
            cursor: int = 0

            @property
            def _llm_type(self):
                return 'offline-scripted-test-only'

            def bind_tools(self, tools, **kwargs):
                return self

            def _generate(self, messages, stop=None, run_manager=None, **kwargs):
                if self.cursor >= len(self.responses):
                    raise AssertionError('脚本模型响应已耗尽')
                response = self.responses[self.cursor]
                self.cursor += 1
                return ChatResult(generations=[ChatGeneration(message=response)])

        return ScriptedModel(responses=messages)

    def settings(self, root):
        return course('config').Settings('https://example.test/v1', 'test-model', 'test-secret',
                                         'chat_completions', Path(root))

    def test_real_langchain_loop_executes_tool_and_records_evidence(self):
        runtime = course('runtime')
        from langchain_core.messages import AIMessage
        messages = [AIMessage(content='', tool_calls=[{'name': 'calculate', 'args': {'expression': '18+24'}, 'id': 'c1'}]),
                    AIMessage(content='结果是 42。')]
        with tempfile.TemporaryDirectory() as root:
            model = self.make_model(messages)
            report = runtime.run_task(self.settings(root), '计算18+24', model=model)
            self.assertEqual(report['status'], 'completed')
            self.assertEqual(report['model_calls'], 2)
            self.assertIn('calculate', report['tool_names'])
            evidence = Path(report['trace_path']).read_text(encoding='utf-8')
            self.assertIn('42', evidence)
            self.assertNotIn('test-secret', evidence)

    def test_model_budget_stops_before_second_network_equivalent_call(self):
        runtime = course('runtime')
        from langchain_core.messages import AIMessage
        messages = [AIMessage(content='', tool_calls=[{'name': 'calculate', 'args': {'expression': '1+1'}, 'id': 'c1'}]),
                    AIMessage(content='2')]
        with tempfile.TemporaryDirectory() as root:
            model = self.make_model(messages)
            report = runtime.run_task(self.settings(root), '计算', model=model, max_calls=1)
            self.assertEqual(report['status'], 'model_limit')
            self.assertEqual(model.cursor, 1)
            self.assertEqual(report['model_calls'], 1)

    def test_collaboration_has_real_two_agents_and_single_shared_budget(self):
        runtime = course('runtime')
        from langchain_core.messages import AIMessage
        model = self.make_model([AIMessage(content='已有证据的答复'), AIMessage(content='{"approved":true,"feedback":"依据充分"}')])
        with tempfile.TemporaryDirectory() as root:
            report = runtime.run_task(self.settings(root), '测试协作', mode='collaborate', model=model, knowledge=False)
            self.assertEqual(report['status'], 'completed')
            self.assertEqual(report['model_calls'], 2)
            self.assertEqual(report['stages'], ['researcher', 'reviewer'])

    def test_failed_tool_stops_planned_workflow(self):
        runtime = course('runtime')
        from langchain_core.messages import AIMessage
        messages = [AIMessage(content='{"steps":["读取缺失文档","继续统计"]}'),
                    AIMessage(content='', tool_calls=[{'name': 'read_document', 'args': {'name': 'missing.md'}, 'id': 'c1'}]),
                    AIMessage(content='文档缺失，停止。')]
        with tempfile.TemporaryDirectory() as root:
            model = self.make_model(messages)
            report = runtime.run_task(self.settings(root), '先读取再统计', mode='plan', model=model)
            self.assertEqual(report['status'], 'step_failed')
            self.assertEqual(model.cursor, 3)

    def test_invalid_tool_arguments_stop_planned_workflow(self):
        runtime = course('runtime')
        from langchain_core.messages import AIMessage
        messages = [AIMessage(content='{"steps":["计算","继续执行"]}'),
                    AIMessage(content='', tool_calls=[{'name': 'calculate', 'args': {}, 'id': 'c1'}]),
                    AIMessage(content='工具参数缺失，无法执行。')]
        with tempfile.TemporaryDirectory() as root:
            model = self.make_model(messages)
            report = runtime.run_task(self.settings(root), '先计算再继续', mode='plan', model=model)
            self.assertEqual(report['status'], 'step_failed')
            self.assertEqual(model.cursor, 3)

    def test_collaboration_budget_applies_across_both_agents(self):
        runtime = course('runtime')
        from langchain_core.messages import AIMessage
        model = self.make_model([AIMessage(content='草稿'), AIMessage(content='{"approved":true,"feedback":"可接受"}')])
        with tempfile.TemporaryDirectory() as root:
            report = runtime.run_task(self.settings(root), '测试协作', mode='collaborate', model=model,
                                      knowledge=False, max_calls=1)
            self.assertEqual(report['status'], 'model_limit')
            self.assertEqual(model.cursor, 1)


if __name__ == '__main__':
    unittest.main()
