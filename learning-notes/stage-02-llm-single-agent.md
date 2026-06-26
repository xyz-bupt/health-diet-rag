# Stage 2：LLM 接入 + 单 Agent 调用

> 目标：接入 DeepSeek LLM，实现第一个 Agent（健康评估），跑通"用户输入 → 结构化输出"的完整链路。
> 时长：约 2 天

---

## 0. 本阶段切换了 LLM 提供商

原设计用智谱 GLM，本阶段起改用 **DeepSeek**，理由：
- API 价格更便宜（输入 1 元/百万 token 量级）
- `deepseek-chat`（V3）质量稳定，国内可直连
- **API 完全兼容 OpenAI 格式** → 直接用 `langchain-openai` 接入，零额外学习成本

⚠️ DeepSeek 暂不提供 embedding API，所以 **Stage 3 起的向量检索会用本地 BGE-M3**（HuggingFace 本地推理，免费且效果在中文场景不输商业 API）。

---

## 1. 本阶段学什么

| 概念 | 为什么重要 |
|---|---|
| LLM 客户端抽象（工厂模式） | 后续所有 Agent 共用一个客户端，便于切换/测试 |
| DeepSeek（OpenAI 兼容）接入 | 国产 LLM 接入的标准方式 |
| MockLLM 降级策略 | 没配 Key 也能跑通流程，方便学习/测试/CI |
| Pydantic 结构化输出 | 让 LLM 的"自由文本"变成"可编程对象" |
| 确定性计算 vs LLM 解读 | RAG/Agent 的核心工程哲学 |
| Prompt 工程（System/Human） | 与 LLM 沟通的标准模板 |
| JSON 输出容错解析 | LLM 经常不老实输出，要兜底 |

---

## 2. 核心概念讲解

### 2.1 为什么需要"LLM 工厂"

**错误做法**：每个 Agent 自己 new 一个客户端

```python
# health.py
llm = ChatOpenAI(model="deepseek-chat", api_key="...")

# nutrition.py
llm = ChatOpenAI(model="deepseek-chat", api_key="...")  # 重复！
```

问题：
1. 配置散落：换 key 要改一堆地方
2. 测试困难：单测里没法替换成 Mock
3. 资源浪费：每个 Agent 都创建独立 HTTP 连接池

**正确做法**：工厂模式（本项目采用）

```python
# app/core/llm.py
@lru_cache(maxsize=1)
def get_llm() -> LLMLike:
    if not has_key():
        return MockLLM()      # 兜底
    return ChatOpenAI(...)    # 真实

# 任何 Agent
from app.core.llm import get_llm
llm = get_llm()  # 全局单例，可注入、可 Mock
```

工厂的三个核心价值：**配置集中、切换方便、可测试**。

### 2.2 DeepSeek 接入：OpenAI 兼容是什么意思

OpenAI 制定了聊天 API 的事实标准，请求/响应格式被广泛复制。DeepSeek 完全照搬这套格式：

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model="deepseek-chat",
    api_key=settings.DEEPSEEK_API_KEY,
    base_url="https://api.deepseek.com",   # ← 关键：把 base_url 换掉就行
    temperature=0.3,
)
```

**好处**：换成 OpenAI / Moonshot / 智谱（OpenAI 模式） / 通义千问（OpenAI 模式）只改 `base_url` 和 `api_key`。

### 2.3 MockLLM 降级：让项目随时可跑

```python
class MockLLM:
    def invoke(self, messages, **kwargs):
        return AIMessage(content="[MockLLM] 占位响应")
```

为什么这么做？三个场景：
1. **刚 clone 项目**：还没申请 key，先跑通流程
2. **单元测试**：完全不依赖网络，确定性输出
3. **CI/CD**：流水线里不消耗真实 token

**核心思想**：LLM 是增强项，不是阻塞项。业务逻辑应该在 LLM 不可用时降级运行。

### 2.4 确定性计算 vs LLM 解读（最重要的工程哲学）

| 任务 | 谁来做 | 为什么 |
|---|---|---|
| 算 BMR | **代码** | 有公式，100% 可复现 |
| 算 TDEE | **代码** | BMR × 系数，机械运算 |
| 算 BMI 分类 | **代码** | 简单阈值判断 |
| 算宏量营养素 | **代码** | 乘除法 + 比例表 |
| 解读数字含义 | **LLM** | 语言任务，LLM 擅长 |
| 给个性化建议 | **LLM** | 需要综合判断 |
| 生成自然语言摘要 | **LLM** | LLM 的强项 |

**为什么这样分？**
- LLM 算数会出错（特别是小数、单位）
- 代码不会出错，但不会写"友好文案"
- **让代码做它擅长的（确定性），让 LLM 做它擅长的（语言）**

这就是"Tool-augmented LLM"思想的雏形。Stage 4 用 LangGraph 编排时会更明显。

### 2.5 Pydantic v2 结构化输出

输入：
```python
class HealthProfile(BaseModel):
    height_cm: float = Field(..., gt=80, lt=250, description="身高（厘米）")
    gender: Literal["male", "female"]
```

好处：
- **FastAPI 自动校验**：身高 50 直接 422，不用写 if
- **Swagger 自动文档**：description 直接出现在 /docs
- **类型提示**：IDE 自动补全

输出：
```python
class HealthAssessment(BaseModel):
    bmi: float
    summary: str
    recommendations: list[str]
    llm_used: bool
```

让 LLM 的输出从"一段字符串"变成"可编程对象"，下游处理方便得多。

### 2.6 Prompt 工程：System vs Human

```python
messages = [
    SystemMessage(content="你是一名营养师...输出要求..."),  # 设定角色和规则
    HumanMessage(content="用户身高 175cm...请评估"),        # 提供具体任务
]
```

- **SystemMessage**：给 LLM 设定"人设"和约束（每次固定）
- **HumanMessage**：每次的具体输入（变量化）

类比：System 是职位说明书，Human 是工作任务书。

### 2.7 JSON 输出容错解析

LLM 经常不老实输出 JSON：

```
好的，以下是结果：
```json
{
  "summary": "..."
}
```
希望对你有帮助。
```

我们的 `_parse_json_response` 做了三层容错：
1. 去掉 markdown 代码块 ` ```json ... ``` `
2. 找第一个 `{` 到最后一个 `}` 之间的内容
3. 用 `json.loads` 解析

**工程教训**：永远假设 LLM 输出会"脏"，做好解析兜底。

---

## 3. 文件结构变化

```
health-diet-rag/
├── app/
│   ├── core/
│   │   ├── config.py        # 改：DeepSeek 配置
│   │   └── llm.py           # 新：LLM 工厂 + MockLLM
│   ├── models/              # 新
│   │   └── health.py        # 新：Pydantic 输入/输出
│   ├── agents/              # 新
│   │   ├── health_calc.py   # 新：纯计算（BMR/TDEE/宏量）
│   │   └── health.py        # 新：HealthAssessmentAgent
│   ├── api/v1/
│   │   ├── health.py        # 老
│   │   └── assess.py        # 新：POST /api/v1/assess
│   └── main.py              # 改：注册 assess 路由
└── tests/
    ├── test_health_calc.py  # 新：纯计算测试
    └── test_assess.py       # 新：Agent + API 集成测试
```

**为什么把计算和 Agent 分文件？**
- `health_calc.py` 完全无依赖，单元测试超快
- `health.py` 依赖 LLM，做集成测试
- 改公式时不会动到 LLM 代码，反之亦然

---

## 4. 跑通验证

### 4.1 跑测试

```bash
cd health-diet-rag
source .venv/bin/activate
pytest tests/ -v
# 应看到 32 passed
```

### 4.2 不填 API Key 也能跑

此时接口返回的 `llm_used: false`，summary 是 MockLLM 生成的规则化文本。

### 4.3 启用真实 LLM

```bash
# 编辑 .env，把 your_api_key_here 换成真实 DeepSeek key
DEEPSEEK_API_KEY=sk-xxxxxxxxxxxx
```

然后：

```bash
uvicorn app.main:app --reload
```

打开 http://localhost:8000/docs，找到 `POST /api/v1/assess`，点 "Try it out"，填：

```json
{
  "height_cm": 175,
  "weight_kg": 70,
  "age": 28,
  "gender": "male",
  "activity_level": "moderate",
  "goal": "lose_weight"
}
```

返回里 `llm_used: true`，summary 和 recommendations 变成 DeepSeek 生成的真实个性化文案。

### 4.4 命令行直测

```bash
curl -X POST http://localhost:8000/api/v1/assess \
  -H "Content-Type: application/json" \
  -d '{"height_cm":175,"weight_kg":70,"age":28,"gender":"male","activity_level":"moderate","goal":"lose_weight"}'
```

---

## 5. 关键代码走读

### 5.1 LLM 工厂的判断逻辑

`app/core/llm.py:81`
```python
key = settings.DEEPSEEK_API_KEY.strip()
if not key or key == "your_api_key_here":
    return MockLLM()
return _build_real_llm()
```

**坑**：占位符 `your_api_key_here` 也要判空，否则会被当成"有效 key"去调真 API，然后失败。

### 5.2 Agent 的 try/except 降级

`app/agents/health.py:124`
```python
try:
    response = self.llm.invoke(messages)
    return _parse_json_response(response.content)
except Exception as e:
    print(f"LLM 调用失败：{e}")
    return _mock_llm_output(profile, data)  # 降级
```

**工程教训**：网络 API 调用一定要 try/except，让接口返回 200 + 降级内容，而不是 500 + stack trace。

### 5.3 宏量分配查表

`app/agents/health_calc.py:103`
```python
ratios = {
    "lose_weight": (0.40, 0.35, 0.25),
    "maintain":    (0.30, 0.45, 0.25),
    "gain_muscle": (0.30, 0.50, 0.20),
}
```

**为什么不写 if/elif？** 查表更易扩展（加新目标只改一行），更易测试，更易读。

---

## 6. 常见坑点

### 6.1 依赖冲突：langchain 系列

`langchain-openai` 最新版会强制升级 `langchain-core` 到 1.x，与 `langchain 0.3.x` 冲突。
解决：固定 `langchain-openai==0.2.14`（与 0.3.x 系列兼容的最后一批）。

### 6.2 ChatOpenAI 与 base_url

不传 `base_url` 会默认连 `https://api.openai.com`，国内访问会超时。
必须传：`base_url="https://api.deepseek.com"`。

### 6.3 LLM 输出的 JSON 不是纯 JSON

LLM 经常在 JSON 外面包 ```json ``` 或加额外文字，必须容错解析（见 2.7）。

### 6.4 temperature 设置

- 0：完全确定性（每次答案一样）
- 0.3：偏稳定（适合事实性问答）
- 0.7：平衡（默认）
- 1.0：创造性（适合写诗）

健康评估用 0.3，避免给的建议天马行空。

### 6.5 Field 约束的方向

```python
height_cm: float = Field(..., gt=80, lt=250)  # 80 < x < 250
age: int = Field(..., ge=10, le=120)          # 10 <= x <= 120
```

`gt` (greater than) vs `ge` (greater or equal)：注意区分。

---

## 7. 检查清单

完成本阶段后，你应该能：

- [ ] 解释为什么 LLM 客户端要用工厂模式
- [ ] 写一个 OpenAI 兼容的 DeepSeek 客户端
- [ ] 解释 MockLLM 降级的意义
- [ ] 解释"确定性计算 vs LLM 解读"的分工原则
- [ ] 写一个带 Field 约束的 Pydantic 模型
- [ ] 解释 SystemMessage 和 HumanMessage 的区别
- [ ] 实现 JSON 输出的容错解析
- [ ] 在 Swagger UI 里调通 `/api/v1/assess` 接口
- [ ] 让 LLM 不可用时接口仍能返回合理结果

---

## 8. 下一阶段预告

Stage 3 会做：
- 引入本地 BGE-M3 embedding 模型（HuggingFace 本地推理）
- 用 ChromaDB 建立食材/菜谱向量库
- 实现语义检索：输入"低脂高蛋白早餐"，召回相关食材
- 学习 chunking、检索器、相似度阈值等 RAG 核心概念

---

## 9. 关键代码索引

| 文件 | 行 | 内容 |
|---|---|---|
| `app/core/llm.py` | 81 | `get_llm()` 工厂主逻辑 |
| `app/core/llm.py` | 35 | `MockLLM` 实现 |
| `app/agents/health_calc.py` | 76 | `calc_bmr` Mifflin-St Jeor 公式 |
| `app/agents/health_calc.py` | 103 | 宏量分配查表 |
| `app/agents/health.py` | 109 | `HealthAssessmentAgent.assess()` 主入口 |
| `app/agents/health.py` | 124 | LLM try/except 降级 |
| `app/api/v1/assess.py` | 15 | POST 接口 |
