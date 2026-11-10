# LangChain 核心知识点：工具调用与规划执行

D1–D3 精简复习版。重点分清：模型决定调用什么，LangChain 执行工具循环，业务函数计算或取数据，Python 编排计划步骤。代码为核心节选，省略部分日志和校验，需要结合课程环境使用。

## 1. 工具调用：模型提出请求，程序执行函数

### 1.1 调用循环与工具定义

用户问题 → 模型返回工具名和参数 → LangChain 执行函数 → 工具结果加入消息 → 模型继续判断或回答。

```python
@tool('calculate')
def calculator(expression: str) -> dict:
    """计算数字四则运算。支持 + - * / 和括号，不支持代码、变量、乘方；请传纯数值表达式。"""
    return calculate(expression)
```

- `@tool` 包装函数：工具名、说明和参数类型组成模型可使用的接口。
- `calculator` 是包装函数；真正计算的是 `calculate(expression)`。模型不直接执行 Python。
- 工具说明影响模型行为；输入是否合法、除零如何处理等限制必须由业务代码落实。

实际观察：计算 `27*13-51` 得到 300。原先是 **2 次模型请求、1 次工具调用**；说明要求拆成两次后，变为 **3 次模型请求、2 次工具调用**。说明影响调用方式，但不保证以后每次相同。

### 1.2 创建、运行与消息传递

```python
agent = create_agent(selected_model, tools, system_prompt=prompt + '\n' + extra, name=stage)
result = agent.invoke(
    {'messages': [{'role': 'user', 'content': task}]},
    config={'callbacks': [evidence], 'recursion_limit': 24},
)
```

`create_agent` 组装流程；`agent.invoke()` 启动执行，内部可以产生多次模型请求。`recursion_limit` 限制执行图步数，不是模型请求次数；课程另设 `max_calls` 统计模型调用上限。

| 消息 | 作用 |
|---|---|
| System | 提示词与约束 |
| Human/User | 用户任务 |
| AI/Assistant | 模型回答或工具调用请求 |
| Tool | 实际工具结果，通过调用 ID 关联请求 |

一次模型响应可以请求多个工具，因此工具调用数不等于模型请求数。模型能使用第一次工具结果，是因为 LangChain 把结果放回后续消息。

## 2. 工具封装：接口清楚，业务结果可信

### 2.1 包装层与业务层

```python
@tool('query_expenses')
def expenses(category: str = 'all') -> dict:
    """查询虚构教学支出，返回明细、笔数及总金额。category 为 all、差旅、办公、餐饮或精确类别。"""
    return query_expenses(category, root=root)
```

模型选择类别；包装层接入 LangChain；`query_expenses` 实际查询数据库。`root` 由程序绑定。参数 schema 检查类型，不代表参数业务含义一定有效，例如不存在的类别仍是字符串。

计算器用 AST 限制表达式语法；数据库使用只读连接和参数化查询；文档读取检查目录范围。工具返回的 `ok`、金额、币种、来源都是业务代码定义的字段。

### 2.2 复用查询与区分返回状态

平均支出的关键逻辑如下，省略了工具装饰器和返回值中的部分来源字段：

```python
result = query_expenses(category, root=root)
if not result['ok']:
    return result

count = result['count']
if count == 0:
    return {'ok': True, 'count': 0, 'average': None}

total_cents = sum(record['cents'] for record in result['records'])
return {'ok': True, 'count': count, 'average': f'{total_cents / count / 100:.2f}'}
```

数据库金额以整数分保存。差旅两笔为 32000 分、23000 分，平均为 `(32000+23000)/2/100 = 275.00` 元。工具内部调用普通 Python 函数不会再请求模型，也不另算一次 Agent 工具调用。

| 返回结果 | 含义与处理 |
|---|---|
| `ok=False` | 查询失败，停止依赖该结果的计算 |
| `ok=True, count=0` | 查询成功但无匹配记录；平均值不是零，也不能除以零 |
| `ok=True, count>0` | 使用实际数据计算，保留币种和来源 |

工具成功不等于模型回答正确；还要核对模型是否使用了实际返回值。教学数据不是实际财务数据，读取报销规则也不代表执行了审批或付款。

## 3. 规划执行：先拆步骤，再传递前序结果

### 3.1 Planner、Executor 与 Python 调度

本项目由命令入口把 `day3` 映射为 `plan` 模式，模型不会自行切换课程模式。

- **Planner**：没有工具，输出 1–3 个步骤。
- **Executor**：带工具，完成当前步骤。
- **Python**：解析和校验计划、顺序运行、传递结果并处理失败。

两种角色使用同一个 `selected_model`；区别是提示词、工具与阶段名称，不是换了一个叫 Planner 的模型。

```python
planning = invoke('planner', question, [],
    '只制定1到3个有依赖顺序的具体步骤，不执行也不猜结果。'
    '只输出JSON对象：{"steps":["步骤一","步骤二"]}。')
plan = parse_object(planning).get('steps')
```

程序还会检查 `steps` 是否为包含 1–3 个非空字符串的列表；格式合法不代表业务步骤一定合理。

### 3.2 history 如何传给下一步

下面保留前序结果传递与失败检查，省略统计和报告字段设置：

```python
history = []
for step in plan:
    mark = len(trace.events)
    output = invoke('executor', json.dumps({
        'original_task': question,
        'current_step': step,
        'previous_results': history,
    }, ensure_ascii=False), selected_tools,
        '只完成当前步骤，引用此前步骤的实际结果。工具失败时停止，不猜测。')
    history.append({'step': step, 'result': output})
    if tool_failed(trace.events[mark:]):
        break
```

第一步收到空列表，第二步收到第一步结果，第三步收到前两步结果。`output` 是模型的最后一条回答文本，不是原始数据库字典；下一步靠阅读文字理解前序结果。

每步是新的 Agent 执行，能利用前面结果依赖 Python 显式传入。`previous_results` 不会自动禁止重复查询。外层失败检查发生在当前 `invoke` 返回后，不能瞬间中断其内部循环。

## 4. Trace、调用次数与费用

### 4.1 读 Trace 判断实际发生了什么

课程通过 LangChain 回调写 JSONL 日志。下面是课程自定义事件名，不是所有 LangChain 项目固定使用的名称。

| 事件 | 记录内容 |
|---|---|
| `start` / `stage` / `plan` | 任务、当前角色阶段、已解析计划 |
| `model_request` / `model_response` | 模型请求与响应、工具调用意图、用量 |
| `tool_request` / `tool_result` / `tool_error` | 工具参数、实际结果或异常 |
| `failure` / `finish` | 运行异常或最终汇总 |

判断工具是否执行，要看工具请求及其结果；最终答案说“我用了工具”本身不构成证据。课程工具事件用 `run_id` 配对，模型工具调用用调用 ID 关联消息。

`model_request` 只记录消息数量，不保存完整输入。Trace 能确认调用与返回，不能证明模型内部动机；`completed` 表示流程正常结束，不代表答案已验收。

### 4.2 请求数、token 与直接执行对照

创建 Agent 不请求模型；执行过程中真正发出的模型请求产生用量。**一次 `invoke` 不等于一次 API 请求，也不是每次请求收同一笔固定费用。** 按 token 计价时，需要输入、输出、缓存计价和服务商单价；最终以账单为准。

本地计算、文件和 SQLite 查询本身不消耗模型 token；结果交回模型后，会计入后续请求输入。外部工具服务可能单独计费。

原笔记记录的同题对照如下：

| 指标 | 直接执行 `tools` | 规划执行 `plan` |
|---|---:|---:|
| Agent 执行数 | 1 | 4：1 个 Planner、3 个 Executor |
| 模型请求数 | 4 | 7 |
| 工具调用数 | 5 | 5 |
| 总 token | 5308 | 10898 |
| 最终金额 | 759.00 CNY | 759.00 CNY |

两者都正确完成 `(550+140)*1.10`。该样例中规划增加了规划和分步总结开销，不能推导成所有任务都更慢或更准确。以上为历史笔记记录，原始两份对照 trace 当前未找到，本次未重跑模型验证。

另一次首次规划运行是 **8 次模型请求、6 次工具调用**，与表中运行不同；多出的查询发生在 Executor 内部。已拥有答案不保证模型选择最省调用的路径，程序需要限制可用工具时必须明确实现。
