#!/bin/bash
# =============================================================================
# 容器启动入口脚本
# =============================================================================
# 职责：
#   1. 如果 ChromaDB 索引未建 → 自动建（首次启动）
#   2. 启动 gunicorn（生产）或 uvicorn（开发，看 ENV 配置）
# =============================================================================

set -e  # 任何命令失败立即退出

echo "=========================================="
echo "  健康饮食 RAG 助手 - 容器启动"
echo "=========================================="
echo "  时间:    $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "  环境:    ${ENV:-prod}"
echo "  工作目录: $(pwd)"
echo "  Python:  $(python --version 2>&1)"
echo "=========================================="

# -----------------------------------------------------------------------------
# 1. 检查 RAG 索引，未建则自动建
# -----------------------------------------------------------------------------
echo "[entrypoint] 检查 RAG 索引状态..."
python -c "
from app.rag.indexer import get_indexer
from app.rag.embedder import is_real_embeddings

ix = get_indexer()
emb_ok = is_real_embeddings()
count = ix.count()

print(f'[entrypoint] embedding provider: {type(ix._embeddings).__name__}')
print(f'[entrypoint] 当前索引文档数: {count}')

if count == 0:
    print('[entrypoint] 索引为空，开始建立...')
    result = ix.index_all()
    print(f'[entrypoint] ✅ 索引建立完成: {result}')
else:
    print('[entrypoint] ✓ 索引已存在，跳过')

if not emb_ok:
    print('[entrypoint] ⚠️  warning: embedding 降级到 Mock，检索质量会下降')
"

# -----------------------------------------------------------------------------
# 2. 启动 Web 服务
# -----------------------------------------------------------------------------
if [ "${ENV}" = "dev" ]; then
    echo "[entrypoint] 开发模式：uvicorn --reload"
    exec uvicorn app.main:app \
        --host "${HOST:-0.0.0.0}" \
        --port "${PORT:-8000}" \
        --reload
else
    echo "[entrypoint] 生产模式：gunicorn ($(nproc) CPUs 可用)"
    exec gunicorn app.main:app \
        --config gunicorn.conf.py \
        --bind "${HOST:-0.0.0.0}:${PORT:-8000}" \
        --chdir /app
fi
