# Stage 4：LangGraph 多 Agent 编排

> 目标：把前 3 个 Stage 的能力（FastAPI + LLM + RAG）整合成一条完整的多 Agent 工作流。
> 时长：约 4-5 天（最复杂的 Stage）

---

## 1. 本阶段学什么

| 概念 | 为什么重要 |
|---|---|
| LangGraph StateGraph | 真正的 Agent 编排框架 |
| State（共享状态） | 多 Agent 之间数据传递的核心 |
| Node（节点） | 把 Agent 包装成可执行单元 |
| Edge（边） | 节点之间的连接 |
| 顺序流水线 | 最简单的多 Agent 模式 |
| Supervisor 模式 | 多 Agent 协作的经典模式 |
| 错误传递与降级 | 工作流的鲁棒性 |
| 流式执行 | 前端逐步展示结果 |
| RAG + LLM 真正闭环 | Stage 3 检索 + Stage 2 LLM 第一次协作 |

---

## 2. 为什么要用 LangGraph

### Stage 2 的"伪 Agent"

Stage 2 的 `HealthAssessmentAgent` 是一个**手写的 Python 类**：

```python
class HealthAssessmentAgent:
    def assess(self, profile):
        bmi = calc_bmi(...)
        result = self.llm.invoke(...)
        return HealthAssessment(...)
```

它的问题：
- 单 Agent 单文件，**互相调用要硬编码**
- 没有统一状态，每个 Agent 自己管自己的数据
- 加一个 Agent 要改一堆地方
- 无法可视化、回放、断点

### LangGraph 的解法

LangGraph 提供 **State + Node + Edge** 三件套：

```python
graph = StateGraph(StateType)
graph.add_node("health", health_fn)    # 注册节点
graph.add_edge("health", "nutrition")  # 定义边
compiled = graph.compile()
result = compiled.invoke(initial_state)
```

把 Agent 协作抽象成**图数据结构**，自动处理：
- 状态传递（State）
- 执行顺序（Edge）
- 错误隔离
- 并行调度（虽然我们这次没用到）
- Checkpoint / 重放（进阶能力）

---

## 3. 核心概念详解

### 3.1 State：共享数据总线

```python
class DietPlanState(TypedDict, total=False):
    profile: HealthProfile       # 输入
    health: HealthAssessment     # 健康 Agent 输出
    nutrition: NutritionPlan     # 营养 Agent 输出
    recipe: MealPlan             # 菜谱 Agent 输出
    exercise: ExercisePlan       # 运动 Agent 输出
    final_plan: DietPlan         # Supervisor 输出
    errors: list[str]            # 错误收集
```

**`total=False`** 让所有字段可选——不同 Node 只填一部分。

**为什么用 TypedDict 而不是 Pydantic？**
- LangGraph 内部用 dict 操作，避免 BaseModel 序列化开销
- 字段值本身可以是 Pydantic 对象（享受类型校验）
- 兼容 LangGraph 的 channel / reducer 机制

### 3.2 Node：状态变换函数

Node 的契约很简单：

```python
def node(state: State) -> dict:
    # 读：state["xxx"]
    # 算：调用 Agent
    # 写：返回部分更新
    return {"yyy": result}
```

**返回 dict，不是完整 State**。LangGraph 会自动合并。这样：
- Node 之间解耦（不需要知道其他字段）
- 多个 Node 可以并行更新不同字段

### 3.3 Edge：流向定义

```python
graph.add_edge(START, "health_node")          # 起点 → health
graph.add_edge("health_node", "nutrition_node")  # health → nutrition
...
graph.add_edge("supervisor_node", END)        # supervisor → 终点
```

**`add_conditional_edges`**（本阶段没用到，进阶能力）：

```python
def route_by_bmi(state):
    if state["health"].bmi > 28:
        return "specialist_node"
    return "standard_node"

graph.add_conditional_edges(
    "health_node",       # 源
    route_by_bmi,        # 路由函数
    {"specialist_node": "X", "standard_node": "Y"},
)
```

适合 BMI > 28 走"肥胖专科 Agent"，其他走"标准 Agent"的场景。

### 3.4 节点名不能和 State 字段重名 ⚠️

我踩的第一个坑：

```python
graph.add_node("health", health_node)  # ❌ ValueError
```

报错：`'health' is already being used as a state key`

因为 State 里已经有 `health` 字段。**节点名加 `_node` 后缀**：

```python
graph.add_node("health_node", health_node)  # ✅
```

---

## 4. 5 个 Agent 的职责

```
START
  ↓
health_node (健康评估)
  - 代码：BMI/BMR/TDEE 计算
  - LLM：生成健康摘要
  ↓
nutrition_node (营养规划)
  - 代码：三餐热量分配（30/40/25/5）
  - LLM：进食时机建议
  ↓
recipe_node (菜谱生成) ⭐ 调用 RAG
  - 小模型：检索食材/菜谱（Stage 3 复用）
  - 大模型：组合成三餐
  ↓
exercise_node (运动建议)
  - 代码：MET 表 + 每周模板
  - LLM：注意事项
  ↓
supervisor_node (整合)
  - 大模型：生成完整方案摘要
  ↓
END
```

**关键协作点**：`recipe_node` 同时使用了：
- Stage 3 的小模型（`get_retriever()`）
- Stage 2 的大模型（`get_llm()`）

这就是 **RAG 完整闭环**：
- **R**etrieval：小模型检索食材
- **A**ugmented：检索结果塞进 LLM prompt
- **G**eneration：大模型生成菜谱

---

## 5. 文件结构

```
app/
├── models/
│   └── diet.py              # ★ 新：4 个 Pydantic 数据结构
├── agents/
│   ├── nutrition.py         # ★ 新：NutritionPlannerAgent
│   ├── recipe.py            # ★ 新：RecipeAgent（调 RAG）
│   ├── exercise.py          # ★ 新：ExerciseAdvisorAgent
│   └── supervisor.py        # ★ 新：SupervisorAgent
├── graph/                   # ★ 全新包
│   ├── state.py             # DietPlanState 定义
│   ├── nodes.py             # 5 个 Node 函数
│   └── workflow.py          # build/compile/invoke/stream
└── api/v1/
    └── diet_plan.py         # ★ 新：POST /api/v1/diet-plan
                              #        POST /api/v1/diet-plan/stream
```

---

## 6. 一个请求的完整生命周期

以 `POST /api/v1/diet-plan` 为例：

```
用户提交 profile
    ↓
api/v1/diet_plan.py → run_diet_plan(profile)
    ↓
workflow.py → workflow.invoke({"profile": profile, "errors": []})
    ↓
┌──── LangGraph 内部调度 ────┐
│                              │
│  health_node 执行             │
│    state["health"] = ...     │
│         ↓                    │
│  nutrition_node 执行          │
│    state["nutrition"] = ...  │
│         ↓                    │
│  recipe_node 执行             │
│    ├── recipe_node 内部:     │
│    │   retriever.search(...)  │ ← Stage 3 RAG
│    │   llm.invoke(...)        │ ← Stage 2 LLM
│    state["recipe"] = ...     │
│         ↓                    │
│  exercise_node 执行           │
│    state["exercise"] = ...   │
│         ↓                    │
│  supervisor_node 执行         │
│    state["final_plan"] = ... │
│                              │
└──────────────────────────────┘
    ↓
返回 final_plan → FastAPI 序列化 → 用户
```

---

## 7. 关键代码走读

### 7.1 工作流构建

`app/graph/workflow.py:42`
```python
def build_diet_plan_graph():
    graph = StateGraph(DietPlanState)
    graph.add_node("health_node", health_node)
    ...
    graph.add_edge(START, "health_node")
    graph.add_edge("health_node", "nutrition_node")
    ...
    graph.add_edge("supervisor_node", END)
    return graph.compile()
```

**注意编译**：`graph.compile()` 把声明式图变成可执行对象。编译后可以：
- `.invoke(state)` 同步执行
- `.stream(state)` 流式执行
- `.stream(state, stream_mode="values")` 看完整 state 演变

### 7.2 错误隔离

`app/graph/nodes.py:25`
```python
def _on_error(state, node_name, err):
    msg = f"{node_name}: {type(err).__name__}: {err}"
    errors = list(state.get("errors", []))
    errors.append(msg)
    return {"errors": errors}

def health_node(state):
    try:
        ...
        return {"health": assessment}
    except Exception as e:
        return _on_error(state, "health_node", e)
```

**关键设计**：任何 Node 失败时**不抛异常**，只把错误记到 `state["errors"]`。这样后续 Node 还能继续跑（虽然可能产出降级结果）。

### 7.3 RAG + LLM 协作（菜谱 Agent 核心）

`app/agents/recipe.py:113`
```python
def generate(self, nutrition, profile):
    # 步骤 1：小模型检索（Stage 3 能力）
    context, rag_sources = self._retrieve_context(profile.goal)

    # 步骤 2：拼接 prompt（检索结果作为"参考资料"）
    prompt = HUMAN_TEMPLATE.format(..., context=context)

    # 步骤 3：大模型生成
    response = self.llm.invoke(messages)
    return MealPlan(...)
```

**RAG 的本质**：把外部知识"塞进" LLM 的上下文，让它**基于事实答题**而非凭记忆。

### 7.4 流式执行

`app/graph/workflow.py:108`
```python
def stream_diet_plan(profile):
    workflow = get_workflow()
    yield from workflow.stream(initial_state, stream_mode="updates")
```

**`stream_mode="updates"`**：每个 Node 完成时 yield 它的更新（增量）。
**`stream_mode="values"`**：每次 yield 完整 State 快照。

前端用 SSE 接收，逐个展示每个 Agent 的结果，用户体验更好。

---

## 8. 关键设计决策

### 8.1 为什么用顺序流水线而不是并行

LangGraph 支持"fan-out + join"并行模式，但我没用，因为：
- 数据依赖：营养需要 BMR/TDEE，菜谱需要营养目标
- **学习项目优先简单**：并行模式调试复杂
- 真实场景并行收益不大（LLM 调用是瓶颈，并行也省不了多少）

后续优化方向：
- recipe + exercise 可以并行（两者都只依赖 health/nutrition）
- 多个不同目标的菜谱可以并行生成（提高多样性）

### 8.2 为什么 Supervisor 在最后

LangGraph 多 Agent 有两种主流模式：
- **Supervisor 模式**（本项目）：Supervisor 在最后整合所有结果
- **Router 模式**：Supervisor 在最前面，根据请求路由到不同子 Agent

我们用前者，因为：
- 业务上需要"完整方案"，不只是单个 Agent 答案
- 整合层可以加摘要、加行动建议、加风险提示
- 简历关键词："多 Agent 协作" 标配

### 8.3 每个 Agent 都有 Mock 兜底

无 API key、网络故障、模型下载失败——所有情况都让流程继续跑。

**核心思想**：LLM 是**增强**项，不是**必需**项。每个 Agent 的结构化输出永远存在，只是质量有差。

---

## 9. 跑通验证

### 9.1 跑测试

```bash
cd health-diet-rag
source .venv/bin/activate
pytest tests/test_workflow.py -v    # 22 passed
pytest tests/                        # 全量 84 passed
```

### 9.2 同步接口

```bash
# 启动服务
uvicorn app.main:app --reload

# 先建索引（如果还没建）
curl -X POST http://localhost:8000/api/v1/index

# 调用完整方案
curl -X POST http://localhost:8000/api/v1/diet-plan \
  -H "Content-Type: application/json" \
  -d '{
    "height_cm": 175, "weight_kg": 70, "age": 28,
    "gender": "male", "activity_level": "moderate", "goal": "lose_weight"
  }'
```

### 9.3 流式接口

```bash
curl -N -X POST http://localhost:8000/api/v1/diet-plan/stream \
  -H "Content-Type: application/json" \
  -d '{"height_cm":175,"weight_kg":70,"age":28,"gender":"male","goal":"lose_weight"}'
```

会看到 5 个 SSE 事件依次到达：

```
data: {"node": "health_node", "result": {...}}
data: {"node": "nutrition_node", "result": {...}}
data: {"node": "recipe_node", "result": {...}}
data: {"node": "exercise_node", "result": {...}}
data: {"node": "supervisor_node", "result": {...}}
data: [DONE]
```

### 9.4 Swagger UI

http://localhost:8000/docs 找 "完整方案" 分组。

---

## 10. 常见坑点

### 10.1 节点名 vs State 字段名冲突

```python
# ❌ 报错：'health' is already being used as a state key
graph.add_node("health", ...)
# ✅ 加 _node 后缀
graph.add_node("health_node", ...)
```

### 10.2 Node 返回完整 State 而非部分

```python
# ❌ 性能差且容易覆盖未读字段
def node(state):
    new_state = state.copy()
    new_state["x"] = ...
    return new_state

# ✅ 只返回更新部分
def node(state):
    return {"x": ...}
```

### 10.3 TypedDict 没加 total=False

```python
# ❌ 所有字段必须填，编译时报错
class State(TypedDict):
    x: int

# ✅ 字段可选，不同 Node 只填一部分
class State(TypedDict, total=False):
    x: int
```

### 10.4 State 里的 list/dict 没用 reducer

如果两个 Node 都向 `state["errors"]` append，后写的会覆盖前面的。需要用 LangGraph 的 `Annotated[list, operator.add]` reducer：

```python
from typing import Annotated
import operator

class State(TypedDict, total=False):
    errors: Annotated[list[str], operator.add]  # 自动 append 合并
```

本项目用了简单的 `list(state.get("errors", [])) + [msg]` 模拟，因为我们是顺序流水线，不存在并发写。

### 10.5 LLM JSON 输出格式漂移

LLM 经常在 JSON 外加文字、加 markdown 围栏、字段名乱写。我们的 `_parse_json_response` 做了三层容错（去围栏 / 找首尾大括号 / json.loads）。

---

## 11. 检查清单

完成本阶段后，你应该能：

- [ ] 解释 LangGraph 的 State / Node / Edge 三件套
- [ ] 写一个最小可用的 StateGraph
- [ ] 解释 Node 为什么只返回部分更新而非全量 State
- [ ] 解释 Supervisor 模式 vs Router 模式的区别
- [ ] 调通 `POST /api/v1/diet-plan` 接口
- [ ] 用 SSE 流式接收每个 Node 的输出
- [ ] 解释 RAG 在菜谱 Agent 里的作用（R+A+G 三步）
- [ ] 处理 LLM JSON 输出的格式容错

---

## 12. 下一阶段预告

Stage 5 会做：
- 前端简易 UI（HTML + JS，调 SSE 接口可视化）
- 异步执行优化（async/await 改造）
- CORS 配置
- 错误响应规范化
- 性能优化（caching / 并行 Node）

---

## 13. 关键代码索引

| 文件 | 行 | 内容 |
|---|---|---|
| `app/graph/state.py` | 30 | `DietPlanState` 定义 |
| `app/graph/nodes.py` | 35 | `health_node` 等 5 个节点 |
| `app/graph/nodes.py` | 25 | `_on_error` 错误隔离 |
| `app/graph/workflow.py` | 42 | `build_diet_plan_graph()` 工作流构建 |
| `app/graph/workflow.py` | 108 | `stream_diet_plan()` 流式执行 |
| `app/agents/recipe.py` | 113 | 菜谱 Agent 的 RAG+LLM 协作 |
| `app/agents/recipe.py` | 158 | `_retrieve_context` RAG 检索 |
| `app/agents/supervisor.py` | 95 | Supervisor 整合所有结果 |
| `app/api/v1/diet_plan.py` | 19 | `POST /diet-plan` 同步接口 |
| `app/api/v1/diet_plan.py` | 53 | `POST /diet-plan/stream` SSE 接口 |
