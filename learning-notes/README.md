# 健康饮食 RAG + 多 Agent 学习笔记

> 与 `../health-diet-rag/` 项目配套使用。每个 Stage 对应项目实现的一个阶段，包含：
> - **概念讲解**：核心知识点
> - **代码示例**：最小可运行版本
> - **常见坑点**：易错位置
> - **检查清单**：本阶段学完后应该掌握的内容

## 学习路径总览

| 阶段 | 主题 | 笔记文件 | 时长 |
|---|---|---|---|
| Stage 1 | FastAPI 基础 + 项目骨架 | [stage-01-fastapi-basics.md](./stage-01-fastapi-basics.md) | 1 天 |
| Stage 2 | LLM 接入 + 单 Agent | [stage-02-llm-single-agent.md](./stage-02-llm-single-agent.md) | 2 天 |
| Stage 3 | RAG 检索增强基础 | [stage-03-rag-fundamentals.md](./stage-03-rag-fundamentals.md) | 3 天 |
| Stage 4 | LangGraph 多 Agent 编排 | [stage-04-langgraph-multi-agent.md](./stage-04-langgraph-multi-agent.md) | 4-5 天 |
| Stage 5 | FastAPI + 前端整合 | [stage-05-fastapi-integration.md](./stage-05-fastapi-integration.md) | 2 天 |
| Stage 6 | Docker 容器化与优化 | [stage-06-docker-polish.md](./stage-06-docker-polish.md) | 2 天 |

## 学习原则

1. **先概念后代码**：每节先理解为什么这么做，再写代码
2. **最小可运行**：先跑通最小版本，再加优化
3. **主动提问**：每个阶段结束问自己三个问题
   - 我刚才学到的核心概念是什么？
   - 我能用一句话解释它吗？
   - 我能不查文档默写出大致结构吗？
4. **错误是好朋友**：报错就贴出来一起分析

## 配套技术栈

- Python 3.11+
- FastAPI
- LangChain + LangGraph
- 智谱 GLM SDK
- ChromaDB
- Pydantic v2
- Docker

## 全局资源

- LangChain 文档：https://python.langchain.com/
- LangGraph 文档：https://langchain-ai.github.io/langgraph/
- FastAPI 文档：https://fastapi.tiangolo.com/
- 智谱 GLM 文档：https://open.bigmodel.cn/dev/api
- ChromaDB 文档：https://docs.trychroma.com/

---

设计文档：`../health-diet-rag/docs/plans/2026-06-18-health-diet-rag-design.md`
