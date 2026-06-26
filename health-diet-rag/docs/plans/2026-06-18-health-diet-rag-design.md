# 健康饮食 RAG + 多 Agent 项目设计文档

- **创建日期**：2026-06-18
- **方案**：B（标准简历版）
- **状态**：已通过 brainstorming 确认

## 1. 项目目标

构建一个集成 RAG 检索增强与多 Agent 协作的健康饮食程序：

- 用户输入身高/体重/年龄/性别/活动量/目标
- 系统输出：个性化饮食方案（含宏量营养素分配、三餐食谱、运动建议）
- 基于营养知识库的 RAG 检索，避免 LLM 幻觉
- 多 Agent 分工协作，由 Supervisor 统一编排

## 2. 技术栈

| 类别 | 选型 | 备注 |
|---|---|---|
| Web 框架 | FastAPI + Uvicorn | 异步、自带 OpenAPI 文档 |
| LLM 编排 | LangGraph | 多 Agent 编排事实标准 |
| LLM SDK | LangChain + 智谱 GLM SDK | 国产模型，国内可直连 |
| 向量库 | ChromaDB（本地） | 零依赖、易学习 |
| Embedding | 智谱 embedding-3 或 BGE-M3 | 国产模型配套 |
| 数据校验 | Pydantic v2 | FastAPI 原生集成 |
| 配置管理 | pydantic-settings + .env | 标准实践 |
| 测试 | pytest + httpx | FastAPI 标配 |
| 容器 | Docker + docker-compose | 一键启动 |

## 3. 系统架构

```
用户请求 (FastAPI)
     ↓
[Supervisor 总览 Agent]  ← LangGraph 编排
     ↓分发任务
┌────┴────┬────────┬─────────┐
↓         ↓        ↓         ↓
健康评估  营养规划  菜谱生成   运动建议
 Agent    Agent    Agent     Agent
                    ↓
              [RAG 检索器]
                    ↓
            ChromaDB 知识库
                    ↓
        (食材营养/菜谱/饮食指南)
     ↓汇总
[Supervisor 整合最终方案]
     ↓
返回用户 (JSON + 流式可选)
```

## 4. Agent 职责

| Agent | 输入 | 工具/RAG | 输出 |
|---|---|---|---|
| 🩺 健康评估 | 身高/体重/年龄/性别/活动量/目标 | BMR/TDEE 计算工具 | 健康画像（BMI、热量目标） |
| 🥗 营养规划 | 健康画像 | 营养素分配规则 | 蛋白质/碳水/脂肪克数 |
| 🔪 菜谱生成 | 营养目标 + 饮食偏好 | **RAG 检索** 食材+菜谱库 | 三餐食谱（含克数） |
| 🏃 运动建议 | 体重目标 + 活动量 | 运动消耗参考表 | 每周运动计划 |
| 📋 Supervisor | 用户原始请求 | 分发 + 整合 | 最终方案 |

## 5. RAG Pipeline

```
data/
├── ingredients.json     # 食材营养数据
├── recipes.json         # 菜谱数据
└── nutrition_guides.md  # 营养指南文档

→ loaders (JSON/Markdown)
→ splitters (recursive + semantic)
→ embeddings (智谱 embedding-3)
→ ChromaDB (persist directory)
→ retriever (MMR + top-k=4)
→ rerank (可选，Cohere 或交叉编码器)
→ context 拼接到菜谱 Agent 的 prompt
```

## 6. API 设计

| 方法 | 路径 | 说明 |
|---|---|---|
| POST | `/api/v1/diet-plan` | 同步生成方案 |
| POST | `/api/v1/diet-plan/stream` | SSE 流式生成 |
| GET | `/api/v1/health/{user_id}` | 查询历史 |
| GET | `/api/v1/foods/search` | 食材检索（调试用） |
| GET | `/docs` | Swagger 自动生成 |

## 7. 文件结构

```
health-diet-rag/
├── app/
│   ├── main.py                # FastAPI 入口
│   ├── core/
│   │   ├── config.py
│   │   └── llm.py
│   ├── agents/
│   │   ├── supervisor.py
│   │   ├── health.py
│   │   ├── nutrition.py
│   │   ├── recipe.py
│   │   └── exercise.py
│   ├── rag/
│   │   ├── indexer.py
│   │   ├── retriever.py
│   │   └── splitters.py
│   ├── graph/
│   │   └── workflow.py
│   ├── api/
│   │   └── v1/
│   │       └── diet.py
│   ├── models/
│   ├── data/
│   └── utils/
├── tests/
├── docs/plans/
├── .env.example
├── requirements.txt
├── Dockerfile
└── README.md
```

## 8. 学习路径（6 阶段）

| 阶段 | 主题 | 时长 | 产出 |
|---|---|---|---|
| Stage 1 | FastAPI 项目骨架 + Hello World | 1 天 | 可启动的 FastAPI 应用 |
| Stage 2 | LLM 接入 + 单 Agent 调用 | 2 天 | 健康评估 Agent 可工作 |
| Stage 3 | RAG 基础（加载→向量库→检索） | 3 天 | 食材知识库可检索 |
| Stage 4 | LangGraph 多 Agent 编排 | 4-5 天 | 5 个 Agent 协同工作 |
| Stage 5 | FastAPI 整合 + 简易前端 | 2 天 | 完整 Web 接口 |
| Stage 6 | Docker + 优化 + README | 2 天 | 可部署的完整项目 |

## 9. 数据来源策略

**混合策略**：
- 先用模拟 JSON/CSV 数据跑通架构
- 预留 `data_loader` 抽象接口，后续可接入：
  - USDA 食物数据库（公开 API）
  - 中国食物成分表
  - 营养指南 PDF 文档

## 10. 简历关键词覆盖

- ✅ RAG / 检索增强生成
- ✅ 多 Agent 协作 / LangGraph
- ✅ 向量数据库（ChromaDB）
- ✅ FastAPI 异步编程
- ✅ Embedding / 语义检索
- ✅ Pydantic 数据校验
- ✅ Docker 容器化
- ✅ 模块化架构设计
