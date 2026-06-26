# =============================================================================
# 健康饮食 RAG 助手 - 多阶段 Dockerfile
# =============================================================================
# 安全说明：
#   - 绝不 COPY .env 进镜像（密钥通过 docker-compose env_file 注入）
#   - .dockerignore 已排除 .env / .venv / __pycache__
#   - 构建产物可公开（公开仓库 / 镜像 registry 都没问题）
# =============================================================================

# -----------------------------------------------------------------------------
# Stage 1: builder - 装编译依赖 + 安装 wheel
# -----------------------------------------------------------------------------
FROM python:3.13-slim AS builder

# 编译需要的基础工具（chromadb 的 hnswlib 要编译）
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        gcc \
        curl \
    && rm -rf /var/lib/apt/lists/*

# 用 uv 加速 pip（可选；不用 uv 也能装，只是慢点）
RUN pip install --no-cache-dir --upgrade pip

WORKDIR /build

# 先只拷 requirements.txt，最大化利用 Docker 层缓存
COPY requirements.txt .

# 装到 /install（后续只拷这一份，builder 的脏东西全留在 builder 阶段）
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt


# -----------------------------------------------------------------------------
# Stage 2: runtime - 干净的生产镜像
# -----------------------------------------------------------------------------
FROM python:3.13-slim AS runtime

# 运行时只需要 curl（healthcheck 用）
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
        tini \
    && rm -rf /var/lib/apt/lists/*

# 拷贝 builder 阶段装好的依赖
COPY --from=builder /install /usr/local

# 创建非 root 用户运行应用（最小权限原则）
RUN useradd --create-home --shell /bin/bash appuser

WORKDIR /app

# 拷贝应用代码
# 注意：.dockerignore 已排除 .env，所以这里 COPY 不会带密钥
COPY --chown=appuser:appuser app/ ./app/
COPY --chown=appuser:appuser static/ ./static/
COPY --chown=appuser:appuser data/ ./data/
COPY --chown=appuser:appuser requirements.txt docker-entrypoint.sh gunicorn.conf.py ./

# entrypoint 要可执行
RUN chmod +x docker-entrypoint.sh

# 切换到非 root 用户
USER appuser

# 容器内默认环境变量（可被 docker-compose / docker run -e 覆盖）
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HOST=0.0.0.0 \
    PORT=8000 \
    ENV=prod \
    CHROMA_PERSIST_DIR=/app/data/chroma

# 健康检查（FastAPI /health 端点）
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD curl -fsS http://localhost:8000/health || exit 1

EXPOSE 8000

# tini 作为 init 进程，正确处理信号（gunicorn 优雅退出）
ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["./docker-entrypoint.sh"]
