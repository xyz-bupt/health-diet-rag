# 项目状态快照（用于跨会话恢复）

> 最近更新：2026-06-26（项目完成 🎉）
> 用途：当用户重新进入 Claude Code 时，让 AI 读取此文件即可恢复完整上下文。

## 给恢复后的 Claude 的指令

```
请阅读 /Users/abc/健康饮食rag/PROJECT_STATE.md
然后阅读 /Users/abc/健康饮食rag/health-diet-rag/docs/plans/2026-06-18-health-diet-rag-design.md
最后阅读 /Users/abc/健康饮食rag/learning-notes/README.md
接着等用户指示（项目已全部完成）。
```

---

## 1. 项目概览

- **项目名**：健康饮食 RAG + 多 Agent（**已完成所有 6 个 Stage**）
- **路径**：`/Users/abc/健康饮食rag/`
- **目标**：简历项目，集成 RAG 检索增强与多 Agent 协作的个性化健康饮食方案生成服务
- **方案**：方案 B（标准简历版）
- **技术栈**：FastAPI（async）+ LangGraph + ChromaDB + DeepSeek + fastembed (BGE-small-zh) + Pydantic v2 + Docker

## 2. 用户画像与偏好

- Python 基础，RAG/Agent 概念了解但没做过完整项目
- **所有非破坏性操作均允许**，无需每次确认
- **按步骤报告 + 讲解**（教学方式）
- LLM：**DeepSeek**（OpenAI 兼容）
- Embedding：**本地 fastembed + BGE-small-zh-v1.5**（90MB / ONNX / 无 torch）
- venv Python 版本：**3.13**
- **DeepSeek API key 已配置**（在 `.env`，已在 `.gitignore` 和 `.dockerignore`）

## 3. 目录结构

```
/Users/abc/健康饮食rag/
├── health-diet-rag/                         # 项目代码
│   ├── app/
│   │   ├── main.py                          # FastAPI 入口（CORS+异常+静态）
│   │   ├── core/
│   │   │   ├── config.py                    # 配置
│   │   │   ├── llm.py                       # LLM 工厂（MockLLM 降级）
│   │   │   ├── cache.py                     # TTL+LRU 缓存
│   │   │   └── exceptions.py                # 业务异常
│   │   ├── models/                          # health.py / diet.py
│   │   ├── agents/                          # health/nutrition/recipe/exercise/supervisor
│   │   ├── rag/                             # embedder/loaders/indexer/retriever
│   │   ├── graph/                           # LangGraph state/nodes/workflow
│   │   └── api/v1/                          # health/assess/rag/diet_plan
│   ├── static/                              # 前端 (HTML/CSS/JS)
│   ├── data/                                # 食材/菜谱/指南 + ChromaDB
│   ├── tests/                               # 147 个测试
│   ├── Dockerfile                           # ★ 多阶段构建
│   ├── docker-compose.yml                   # ★ 一键部署
│   ├── .dockerignore                        # ★ 安全过滤
│   ├── docker-entrypoint.sh                 # ★ 启动脚本
│   ├── gunicorn.conf.py                     # ★ 生产配置
│   ├── .env / .env.example / requirements.txt
│   └── README.md                            # ★ 产品级
└── learning-notes/
    ├── stage-01-fastapi-basics.md
    ├── stage-02-llm-single-agent.md
    ├── stage-03-rag-fundamentals.md
    ├── stage-04-langgraph-multi-agent.md
    ├── stage-05-fastapi-integration.md
    └── stage-06-docker-polish.md            # ★ Stage 6 完整讲解
```

## 4. 进度

### 全部完成 ✅

| Stage | 主题 | 关键产出 |
|---|---|---|
| 1 | FastAPI 骨架 | 项目结构 + 配置 + lifespan |
| 2 | LLM 接入 | DeepSeek + MockLLM + BMR/TDEE 公式 |
| 3 | RAG 基础 | fastembed + ChromaDB + 语义检索 |
| 4 | 多 Agent | LangGraph 5 节点 + RAG+LLM 闭环 |
| 5 | 整合 + 前端 | async + SSE + 缓存 + 异常 + HTML 前端 |
| 6 | Docker 化 | 多阶段构建 + gunicorn + 安全验证 |

### Stage 6 完成项 ✅

- Dockerfile 多阶段构建（builder + runtime，~450MB vs 1.5GB）
- `.dockerignore` 排除 `.env` 和其他敏感文件
- `docker-compose.yml` 一键启动 + env_file 注入 + volume 持久化
- `docker-entrypoint.sh` 启动前自动建索引
- `gunicorn.conf.py` 生产配置（4 worker + 120s timeout）
- 重写 README：架构图 + API 概览 + 学习路径 + 安全说明
- **密钥安全验证**：7 项静态扫描全部通过
- 39 个 Stage 6 测试通过

## 5. 关键设计决策

| 决策 | 选择 | 理由 |
|---|---|---|
| Web 框架 | FastAPI | 异步、自动文档 |
| Agent 编排 | LangGraph StateGraph | 多 Agent 事实标准 |
| 向量库 | chromadb 1.x 直连 | 绕过 langchain-chroma 依赖地狱 |
| LLM | DeepSeek | OpenAI 兼容、便宜 |
| Embedding | fastembed + bge-small-zh-v1.5 | 90MB / 无 torch |
| 异步模式 | `asyncio.to_thread` 包装同步代码 | 最小改造 |
| 流式方案 | SSE | 单向推送足够 |
| 缓存 | TTL+LRU 内存 | 学习项目够用 |
| 错误处理 | 业务异常 + 统一响应 | 前端易处理 |
| **Docker 构建** | **多阶段** | 体积小、安全 |
| **进程管理** | **gunicorn + uvicorn workers** | 生产标配 |
| **PID 1** | **tini** | 正确处理信号 |
| **运行用户** | **非 root (appuser)** | 最小权限 |
| **密钥注入** | **env_file (run 时)** | 不进镜像 |

## 6. 验证状态

```bash
cd /Users/abc/健康饮食rag/health-diet-rag
source .venv/bin/activate

# 全量测试
pytest tests/ -v                # 147 个测试

# 静态安全检查（快速）
pytest tests/test_stage6.py -v  # 39 个

# 本地运行
uvicorn app.main:app --reload
# 浏览器：http://localhost:8000/

# Docker 部署
docker compose up
```

## 7. 简历关键词覆盖度

| 关键词 | 状态 | 实现位置 |
|---|---|---|
| RAG / 检索增强生成 | ✅ 完整闭环 | recipe_agent + rag/ |
| 多 Agent 协作 | ✅ 5 Agent | graph/ |
| LangGraph | ✅ StateGraph | graph/workflow.py |
| 向量数据库 | ✅ ChromaDB | rag/indexer.py |
| FastAPI 异步 | ✅ async/await | api/ + workflow.py |
| Embedding / 语义检索 | ✅ BGE | rag/embedder.py |
| Pydantic 数据校验 | ✅ v2 | models/ |
| Mock 兜底 | ✅ 全链路 | 每个 Agent |
| SSE 流式输出 | ✅ Server-Sent Events | diet_plan/stream + app.js |
| 缓存优化 | ✅ TTL+LRU | core/cache.py |
| 统一异常处理 | ✅ 业务异常类 | core/exceptions.py |
| CORS 跨域 | ✅ 中间件 | main.py |
| 前端整合 | ✅ HTML+CSS+JS | static/ |
| **Docker 容器化** | ✅ 多阶段构建 | Dockerfile |
| **生产部署** | ✅ gunicorn + compose | gunicorn.conf.py + docker-compose.yml |
| **密钥安全** | ✅ 不进镜像 | .dockerignore + env_file |

## 8. 后续可探索方向

| 方向 | 学什么 |
|---|---|
| 生产级 K8s | Helm chart / Ingress / ConfigMap / Secret |
| 可观测性 | Prometheus / Grafana / OpenTelemetry |
| CI/CD | GitHub Actions / 自动构建镜像 / 安全扫描 |
| 模型优化 | reranker / hybrid search / fine-tuning |
| 更复杂 Agent | LangGraph 条件路由 / 并行 Node / 人机协作 |
| 真实数据 | 接 USDA API / 中国食物成分表 |

## 9. 聊天记录要点（避免重复对话）

- 6 个 Stage 全部完成
- Stage 5 期间踩过 3 个 bug（全部已修）：
  1. `str.format()` 把 prompt 模板里 JSON 示例的 `{}` 当占位符
  2. SSE 序列化没处理嵌套 Pydantic 模型
  3. LLM 30s 超时不够，改 90s
- Stage 5 之后又修了一个前端 bug：renderNodeResult 没拆 LangGraph 的 wrapper key
- Stage 6 重点：API key 安全（已通过 7 项静态测试验证不泄漏）
- 项目可以直接交付（GitHub 推送 + 简历引用 + 面试演示）
