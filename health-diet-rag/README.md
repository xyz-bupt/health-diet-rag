# 🥗 健康饮食 RAG 助手

> 基于 **RAG 检索增强 + 多 Agent 协作** 的个性化健康饮食方案生成服务。
> 输入身体数据，输出含健康评估、营养规划、三餐食谱、运动建议的完整方案。

![Python](https://img.shields.io/badge/Python-3.13-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-green)
![LangGraph](https://img.shields.io/badge/LangGraph-0.2-purple)
![License](https://img.shields.io/badge/license-MIT-brightgreen)

---

## ✨ 核心特性

- 🤖 **5 Agent 协作**：健康评估 → 营养规划 → 菜谱生成 → 运动建议 → Supervisor 整合
- 🔍 **本地 RAG 知识库**：ChromaDB + BGE 中文嵌入，零外部依赖
- 🌊 **流式输出**：SSE 推送，每个 Agent 完成即时显示
- 🧠 **DeepSeek LLM**：国产模型，便宜、国内直连、OpenAI 兼容
- 🎯 **确定性 + 智能化**：BMR/TDEE 等数学计算交给代码，自然语言解读交给 LLM
- 🛡️ **优雅降级**：无 API Key / 无网络也能跑完整流程（Mock 兜底）
- 🐳 **一键部署**：`docker compose up`

---

## 🏗 架构图

```
┌──────────────────────────────────────────────────────────────┐
│ 浏览器  http://localhost:8000/                                │
└────────────────────────┬─────────────────────────────────────┘
                         │ SSE / JSON
┌────────────────────────▼─────────────────────────────────────┐
│ FastAPI（async + CORS + 缓存 + 统一异常）                     │
└────────────────────────┬─────────────────────────────────────┘
                         │
┌────────────────────────▼─────────────────────────────────────┐
│ LangGraph StateGraph（5 节点顺序流水线）                      │
│                                                               │
│  START → health_node ──────────────────────┐                  │
│           ↓                                  │                 │
│         nutrition_node                       │                 │
│           ↓                                  │                 │
│         recipe_node ──→ [小模型 RAG 检索]    │                 │
│           ↓                                  │                 │
│         exercise_node                        │                 │
│           ↓                                  │                 │
│         supervisor_node ◄───────────────────┘                 │
│           ↓                                                   │
│          END                                                  │
└──────────────────────────────────────────────────────────────┘
        │                                  │
        ▼                                  ▼
┌───────────────────┐         ┌───────────────────────────────┐
│  DeepSeek LLM     │         │  ChromaDB + BGE-small-zh      │
│  (大模型，解读+生成) │         │  (小模型，本地 embedding+检索) │
└───────────────────┘         └───────────────────────────────┘
```

---

## 🚀 一键启动

### 方式 1：Docker（推荐）

```bash
git clone <repo>
cd health-diet-rag
cp .env.example .env
# 编辑 .env，填入你的 DeepSeek API Key

docker compose up
```

首次启动会：
1. 构建镜像（约 2-3 分钟）
2. 下载 BGE-small-zh embedding 模型（90MB，仅一次）
3. 建立 RAG 索引（37 条文档）
4. 启动 4 个 gunicorn worker

打开 http://localhost:8000/ 即可使用。

### 方式 2：本地运行（开发）

```bash
git clone <repo>
cd health-diet-rag
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# 编辑 .env，填入 DEEPSEEK_API_KEY

# 建索引（首次）
python -c "from app.rag.indexer import get_indexer; get_indexer().index_all()"

# 启动开发服务
uvicorn app.main:app --reload
```

---

## 📋 API 概览

| Method | Path | 说明 |
|---|---|---|
| GET | `/` | 前端页面 |
| GET | `/health` | 健康检查（Docker healthcheck 用） |
| POST | `/api/v1/assess` | 单 Agent 健康评估（Stage 2） |
| POST | `/api/v1/index` | 建立 RAG 索引 |
| GET | `/api/v1/index/status` | 查看索引状态 |
| GET | `/api/v1/foods/search?q=...` | 语义检索食材/菜谱 |
| POST | `/api/v1/diet-plan` | **完整方案（同步）** |
| POST | `/api/v1/diet-plan/stream` | **完整方案（SSE 流式，推荐）** |
| GET | `/api/v1/cache/stats` | 缓存命中率统计 |
| POST | `/api/v1/cache/clear` | 清空缓存 |
| GET | `/docs` | Swagger UI |

### 请求示例

```bash
# 生成完整方案
curl -X POST http://localhost:8000/api/v1/diet-plan \
  -H "Content-Type: application/json" \
  -d '{
    "height_cm": 175,
    "weight_kg": 70,
    "age": 28,
    "gender": "male",
    "activity_level": "moderate",
    "goal": "lose_weight"
  }'
```

---

## 🛠 技术栈

| 层 | 技术 | 选型理由 |
|---|---|---|
| Web 框架 | **FastAPI 0.115** | 异步、自动 OpenAPI 文档 |
| Agent 编排 | **LangGraph 0.2** | 多 Agent 协作事实标准 |
| LLM | **DeepSeek-Chat** | 国产、便宜（输入 0.5 元/百万 token）、国内直连 |
| LLM 客户端 | **LangChain-OpenAI 0.2** | OpenAI 兼容接口 |
| 向量数据库 | **ChromaDB 1.x** | 本地持久化、零依赖 |
| Embedding | **fastembed + BGE-small-zh-v1.5** | 90MB / ONNX / 无 torch / CPU 友好 |
| 数据校验 | **Pydantic v2** | FastAPI 原生集成 |
| 进程管理 | **gunicorn + uvicorn workers** | 生产标配 |
| 容器 | **Docker + docker-compose** | 一键部署 |

---

## 📁 项目结构

```
health-diet-rag/
├── app/
│   ├── main.py              # FastAPI 入口（CORS + 异常 + 静态）
│   ├── core/                # 基础设施
│   │   ├── config.py        # 配置
│   │   ├── llm.py           # LLM 工厂（MockLLM 降级）
│   │   ├── cache.py         # TTL+LRU 缓存
│   │   └── exceptions.py    # 业务异常 + 处理器
│   ├── models/              # Pydantic 数据模型
│   ├── agents/              # 5 个 Agent
│   │   ├── health.py        # 健康评估（含 BMR/TDEE 公式）
│   │   ├── nutrition.py     # 营养规划
│   │   ├── recipe.py        # 菜谱生成（调 RAG）
│   │   ├── exercise.py      # 运动建议
│   │   └── supervisor.py    # 整合 Agent
│   ├── rag/                 # RAG 流水线
│   │   ├── embedder.py      # embedding 工厂
│   │   ├── loaders.py       # JSON/MD → Document
│   │   ├── indexer.py       # ChromaDB 写入
│   │   └── retriever.py     # 语义检索
│   ├── graph/               # LangGraph 工作流
│   │   ├── state.py         # 共享状态
│   │   ├── nodes.py         # 5 个 Node
│   │   └── workflow.py      # 同步/异步入口
│   └── api/v1/              # 4 套路由
├── static/                  # 前端（HTML+CSS+JS）
├── data/                    # 食材/菜谱/指南 + ChromaDB 持久化
├── tests/                   # 108 个测试
├── Dockerfile               # 多阶段构建
├── docker-compose.yml       # 一键部署
├── gunicorn.conf.py         # 生产配置
├── docker-entrypoint.sh     # 启动脚本（自动建索引）
└── requirements.txt
```

---

## 📚 学习路径（6 阶段）

本项目按"渐进式学习"设计，每个 Stage 都有完整讲解：

| Stage | 主题 | 关键技术 | 笔记 |
|---|---|---|---|
| 1 | FastAPI 基础 | 项目骨架、应用工厂、TestClient | [stage-01](../learning-notes/stage-01-fastapi-basics.md) |
| 2 | LLM 接入 | DeepSeek、Pydantic 结构化输出、Mock 降级 | [stage-02](../learning-notes/stage-02-llm-single-agent.md) |
| 3 | RAG 基础 | fastembed、ChromaDB、BGE 中文嵌入、检索 | [stage-03](../learning-notes/stage-03-rag-fundamentals.md) |
| 4 | 多 Agent | LangGraph StateGraph、RAG+LLM 闭环 | [stage-04](../learning-notes/stage-04-langgraph-multi-agent.md) |
| 5 | 整合 + 前端 | async、CORS、SSE、缓存、简易 UI | [stage-05](../learning-notes/stage-05-fastapi-integration.md) |
| 6 | Docker 化 | 多阶段构建、gunicorn、健康检查 | [stage-06](../learning-notes/stage-06-docker-polish.md) |

---

## 🔧 配置项（`.env`）

```bash
# 必填：DeepSeek API Key
DEEPSEEK_API_KEY=sk-...
DEEPSEEK_BASE_URL=https://api.deepseek.com

# LLM 模型（可选，有默认值）
LLM_MODEL=deepseek-chat

# Embedding 模型（可选）
EMBEDDING_MODEL=BAAI/bge-small-zh-v1.5
EMBEDDING_PROVIDER=fastembed

# RAG 检索参数（可选）
RAG_TOP_K=4
RAG_CHUNK_SIZE=500
RAG_CHUNK_OVERLAP=50

# 应用配置（可选）
HOST=0.0.0.0
PORT=8000
ENV=prod               # prod 用 gunicorn，dev 用 uvicorn --reload
HOST_PORT=8000         # 宿主机映射端口（docker-compose 用）

# 向量库（可选）
CHROMA_PERSIST_DIR=./data/chroma
CHROMA_COLLECTION=health_diet
```

---

## 🔒 安全说明

- ✅ **API Key 不进镜像**：`.dockerignore` 排除 `.env`，`Dockerfile` 不 COPY 它
- ✅ **密钥运行时注入**：通过 `docker-compose.yml` 的 `env_file: .env` 传入容器
- ✅ **非 root 运行**：容器内使用 `appuser` 用户
- ✅ **`.env` 已在 `.gitignore`**：不会被 Git 追踪
- ✅ **镜像可公开**：构建产物无敏感信息

---

## 🧪 测试

```bash
# 跑全部测试（108 个）
pytest tests/ -v

# 跑特定 Stage
pytest tests/test_rag.py -v       # RAG 模块
pytest tests/test_workflow.py -v  # 工作流
```

---

## 📊 性能

| 场景 | 耗时 | 说明 |
|---|---|---|
| 单次 `/api/v1/diet-plan`（首次） | ~100s | 5 个 Agent 顺序调 LLM |
| 单次 `/api/v1/diet-plan`（缓存命中） | ~5ms | TTL 5 分钟 |
| `/api/v1/foods/search` | ~50ms | 小模型 embed + ChromaDB 检索 |
| 单 worker QPS（缓存命中） | ~200 | 纯内存 |
| 4 worker QPS（缓存命中） | ~800 | CPU 多核利用 |

---

## 🎯 简历关键词覆盖

✅ RAG / 检索增强生成　✅ 多 Agent 协作　✅ LangGraph　✅ 向量数据库
✅ FastAPI 异步编程　✅ Embedding / 语义检索　✅ Pydantic 数据校验
✅ SSE 流式输出　✅ 缓存优化（TTL+LRU）　✅ Docker 容器化　✅ 多阶段构建

---

## 📝 License

MIT License - 仅供学习参考。实际营养建议请咨询注册营养师。
