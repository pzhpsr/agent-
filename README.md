# LangChain 七天互动实战课

每天 3–4 小时，围绕规划、工具调用、多 Agent、API/数据库/文档对接、RAG 和评测学习。
**课程代码准备完成与否、助手验证结果、你的学习进度分别记录，不互相替代。**

## 从这里开始

- [文件说明：每个课程文件的用途](文件说明.md)
- [七天安排](plans/七天学习计划.md)
- [第一天：看清工具调用循环](第1天_工具调用/学习指南.md)
- [个人学习进度](学习进度.md)
- [实际验证记录](docs/验证记录.md)

所有命令都在 `H:/论文学习/agent/xuexi` 内运行。下面是 Windows PowerShell 命令，不必先激活环境：

```powershell
Set-Location 'H:/论文学习/agent/xuexi'
./.venv/Scripts/python.exe -X utf8 -m xuexi_lab init
./.venv/Scripts/python.exe -X utf8 -m xuexi_lab doctor
./.venv/Scripts/python.exe -X utf8 -m xuexi_lab day1
```

`doctor` 会产生少量真实模型调用，必须实际完成计算工具调用和结果回传才算通过。
`day1` 输出中的 `completed` 只代表运行结束，仍需读 trace 核对参数、返回值和回答。
若接口失败，保留错误记录并修复；不把脚本模型或离线测试的响应当作 API 已接通。

## 环境与配置

当前机器复用 `D:/Miniconda3/envs/psr` 的 Python 3.12 和 PyTorch。
课程 `.venv` 使用 `--system-site-packages` 继承已有 PyTorch，其新增依赖安装在本目录，不修改 psr 的包。
这也意味着课程环境依赖 psr 仍位于原路径；跨机器复现应使用独立环境。

复建本机环境：

```powershell
& 'D:/Miniconda3/envs/psr/python.exe' -m venv --system-site-packages .venv
./.venv/Scripts/python.exe -m pip install -r requirements-lock.txt
./.venv/Scripts/python.exe -m pip install --no-deps -e .
```

`requirements-lock.txt` 锁定课程环境本地新增包；继承环境的版本记录在 `environment-base.json`。
跨机器需先安装合适的 PyTorch，并按本机平台重新验证。课程依赖的范围声明在 `pyproject.toml`。

配置优先级：系统环境变量 > 本目录 `.env` > 上一层 `agent/.env`。
默认复用已有 `AGENT_BASE_URL`、`AGENT_MODEL`、`AGENT_API_KEY`、`AGENT_API_STYLE`。
如需本课单独覆盖配置，参照 `.env.example`；无需将真实密钥发到聊天里。
LangChain 统一使用已验证的接口风格，`auto` 由 doctor 验证后记录，地址或模型变化需要重跑。
课程关闭 LangSmith 云追踪，只保留本地执行证据。超时 90 秒、自动重试关闭，单任务最多 8 次模型调用。

## 每天一个学习文件夹

每天打开对应文件夹的 `学习指南.md`，把练习与疑问记在同目录 `学习笔记.md`。公共代码在 `xuexi_lab/`，数据和环境共用。
即使打开了某一天的文件夹，终端也先切回 `H:/论文学习/agent/xuexi`，再运行指南中的命令。

## 七个入口

| 入口 | 内容 | 讲义 |
|---|---|---|
| `day1` | 计算工具调用 | [D1](第1天_工具调用/学习指南.md) |
| `day2` | 天气 API、SQLite、文档工具 | [D2](第2天_工具封装/学习指南.md) |
| `day3` | 先规划、再按依赖执行 | [D3](第3天_规划执行/学习指南.md) |
| `day4` | 本地文档 RAG 问答 | [D4](第4天_RAG/学习指南.md) |
| `day5` | 资料助手与检查助手 | [D5](第5天_多Agent协作/学习指南.md) |
| `day6` | 固定案例评测，默认仅 2 题 | [D6](第6天_评测迭代/学习指南.md) |
| `day7` | 知识库＋数据＋计算＋复核 | [D7](第7天_综合实践/学习指南.md) |

`day1` 至 `day5` 和 `day7` 均支持 `--question "自定义问题"` 与 `--max-calls 8`。
`run --mode tools --question "..."` 用于指定模式；`--variant evidence_first` 用于单因素提示词对照。

## 先独立验证工具

```powershell
./.venv/Scripts/python.exe -X utf8 -m xuexi_lab tool calculate --expression '(18+24)/2'
./.venv/Scripts/python.exe -X utf8 -m xuexi_lab tool expenses --category '差旅'
./.venv/Scripts/python.exe -X utf8 -m xuexi_lab tool document --name '01_统计规则.md'
./.venv/Scripts/python.exe -X utf8 -m xuexi_lab tool weather --city '成都'
```

SQLite 六条记录、知识库六篇文档均为虚构教学资料，不能当作真实企业数据。
天气通过 Open-Meteo 获取当日日预报，只支持成都、北京和上海，明确区分日最低/最高温度与实时温度。
工具不提交真实工单，不发起审批或付款。

## RAG 首次运行

```powershell
./.venv/Scripts/python.exe -X utf8 -m xuexi_lab rag-build
./.venv/Scripts/python.exe -X utf8 -m xuexi_lab search --query '投影仪最多借用几天'
./.venv/Scripts/python.exe -X utf8 -m xuexi_lab day4
```

首次运行会从 Hugging Face 下载 `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`，使用 CPU。
模型在 `models/`，嵌入缓存位于 `runs/embedding_cache/`；不调用付费嵌入 API。
每次运行重建内存索引，文档变化时内容哈希会更新，未变化的片段复用缓存。
检索相似度不是答案正确率。必须检查返回的原文，回答引用文档名和 chunk；没有证据时说明知识库未提供。
模型下载失败时如实报错，不切换到关键词匹配冒充语义检索。

## 评测和运行记录

```powershell
./.venv/Scripts/python.exe -X utf8 -m xuexi_lab eval --preview --split dev
./.venv/Scripts/python.exe -X utf8 -m xuexi_lab eval --split dev --limit 2
./.venv/Scripts/python.exe -X utf8 -m xuexi_lab eval --split dev --limit 8 --variant baseline
./.venv/Scripts/python.exe -X utf8 -m xuexi_lab eval --split dev --limit 8 --variant evidence_first
./.venv/Scripts/python.exe -X utf8 -m xuexi_lab eval --split holdout --limit 4
./.venv/Scripts/python.exe -X utf8 -m pytest
```

先预览、再小规模验证，完整评测留到第六天。12 题中 8 题用于开发、4 题保留到最终检查。
自动检查工具调用、阶段、运行状态和必要答案片段；`auto_pass` 不等于人工语义核对通过。
对照命令：`compare --before '第一次结果.json' --after '第二次结果.json'`，两次题目及模型配置必须一致。
`runs/traces/` 保存模型和工具事件；`trace --path '实际 trace 路径'` 可查看。不要使用虚构日志路径。
模型用量未提供时记录为 null，不当作 0；没有单价依据时不估算人民币费用。

## 参考与代码复用

- [LangChain 官方仓库](https://github.com/langchain-ai/langchain)：核对 Agent、消息、工具、向量存储等实际接口；MIT。
- [LangChain Academy](https://github.com/langchain-ai/langchain-academy)：参考逐步练习与运行观察方式；MIT。
- [Hello-Agents](https://github.com/datawhalechina/hello-agents)：选读第 4/6/7/8/12 章，不复制其自研框架；仅作学习链接。
- [All-in-RAG](https://github.com/datawhalechina/all-in-rag)：补充加载、分块、嵌入和检索知识；仅作学习链接。
- `math_tools.py` 复用当前工作区原有 Agent 项目的计算器，保留 AST 白名单与数值限制。

项目中未复制外部教程正文或整套仓库；保留原创教学代码和本地已有工具的复用说明。
