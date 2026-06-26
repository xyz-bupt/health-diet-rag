# Stage 6：Docker 容器化与产品打磨

> 目标：把"能跑"的项目变成"能交付"的产品——一键 `docker compose up` 启动，密钥安全，README 漂亮。
> 时长：约 2 天

---

## 1. 本阶段学什么

| 概念 | 为什么重要 |
|---|---|
| 多阶段 Docker 构建 | 镜像体积从 1.5GB → 400MB |
| `.dockerignore` 与密钥安全 | 防止 API key 泄漏到镜像 |
| Build 时 vs Run 时 | 密钥该在哪一步进入容器 |
| Docker HEALTHCHECK | 容器自愈的基础 |
| gunicorn + uvicorn worker | 生产环境多进程管理 |
| tini 作为 PID 1 | 正确处理信号（优雅退出） |
| 非 root 用户 | 最小权限原则 |
| Volume 持久化 | 容器删除后数据不丢 |
| docker-compose 编排 | 一条命令启动整条服务链 |

---

## 2. 核心概念详解

### 2.1 多阶段构建：为什么 / 怎么做

**问题**：单阶段 Dockerfile 出来的镜像会包含：
- 编译工具（gcc、build-essential）→ 几百 MB
- pip 缓存 → 几十 MB
- 中间产物 → 几十 MB

这些**运行时根本用不到**，但占了 80% 体积。

**多阶段思路**：

```dockerfile
# Stage 1: builder（装满编译工具）
FROM python:3.13-slim AS builder
RUN apt-get install gcc build-essential ...
RUN pip install -r requirements.txt  # 装到 /install

# Stage 2: runtime（干净的运行时）
FROM python:3.13-slim AS runtime
# 只拷 site-packages，不拷 gcc
COPY --from=builder /install /usr/local
COPY app/ ./app/
```

**体积对比**：
- 单阶段：~1.5 GB
- 多阶段：~450 MB（Python slim 自身 ~150MB + 依赖 ~250MB + 应用 ~50MB）

### 2.2 密钥安全：build vs run（最重要）

**最大的反模式** ❌：

```dockerfile
COPY .env /app/.env   # 把 key 烤进镜像层！
ENV DEEPSEEK_API_KEY=sk-xxx   # 同样烤进镜像！
```

**这样做的后果**：
- 镜像推到 Docker Hub / 私有仓库 → **任何能 pull 的人都能 `docker history` 看到 key**
- 即使后来删掉，前面的层仍然含 key
- 一旦泄漏，必须立刻 revoke

**正确做法** ✅：

```dockerfile
# 1. .dockerignore 排除 .env（build context 就不含）
.env
.env.*
!.env.example

# 2. Dockerfile 不 COPY .env，只 COPY .env.example
COPY .env.example ./.env.example

# 3. docker-compose.yml 用 env_file 运行时注入
services:
  app:
    env_file: .env
```

**build 时**：
```
开发机 .env (sk-xxx)
   ↓ .dockerignore 排除
[Build Context] ← 不含 .env
   ↓ docker build
[Image] ← 不含 key（可公开）
```

**run 时**：
```
开发机 .env (sk-xxx)
   ↓ docker-compose.yml env_file
[Container 进程环境变量] ← key 在这里
   ↓ app.settings.DEEPSEEK_API_KEY 读到
[应用能调 LLM] ✓
```

**关键洞察**：镜像层是不可变的（ immutable），放进去就拿不出来。**永远不要把密钥放到 build 阶段**。

### 2.3 验证密钥安全

构建后跑这几个检查：

```bash
# 1. 看 Dockerfile 所有 COPY 命令，确认没有 COPY .env
grep "^COPY" Dockerfile

# 2. 扫描镜像层历史，确认没有 sk- 开头的字符串
docker history --no-trunc health-diet-rag:latest | grep -i "sk-"

# 3. 启动容器，看文件系统里有没有 .env
docker run --rm health-diet-rag:latest ls /app/

# 4. 看镜像里所有环境变量（不应有 API key）
docker run --rm health-diet-rag:latest env | grep -i key
```

本项目用静态分析做同样的事（见 `tests/test_stage6.py::TestSecretSafety`），CI 跑测试就能拦截。

### 2.4 Docker HEALTHCHECK：容器自愈

```dockerfile
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD curl -fsS http://localhost:8000/health || exit 1
```

**作用**：Docker 每 30 秒调一次 `/health`，连续 3 次失败就标记为 `unhealthy`，配合 `restart: unless-stopped` 可以自愈。

**为什么单独配 start_period**：

```yaml
healthcheck:
  start_period: 60s   # 容器启动后 60 秒内不算失败
```

应用启动时要建索引 + 下载 embedding 模型（~30-60 秒），不设这个就会被误判挂掉。

### 2.5 gunicorn + uvicorn worker：生产标配

**为什么不用 uvicorn 单进程**？
- uvicorn 单进程，4 核 CPU 浪费 3 个
- gunicorn 是成熟的进程管理器（自动重启挂掉的 worker）

**关键配置**（`gunicorn.conf.py`）：

```python
workers = multiprocessing.cpu_count() * 2 + 1   # 经典公式
worker_class = "uvicorn.workers.UvicornWorker"  # FastAPI 必须
timeout = 120                                    # LLM 调用慢
max_requests = 1000                              # 防 memory leak
```

**worker_class 为什么是 uvicorn worker**？因为 FastAPI 是异步框架，gunicorn 默认的 sync worker 不支持 asyncio。`UvicornWorker` 让 gunicorn 用 uvicorn 处理异步请求。

### 2.6 tini 作为 PID 1：信号处理

**问题**：容器里的 PID 1 默认是入口进程（这里是 gunicorn）。但 gunicorn 不知道怎么处理 SIGTERM 之类的内核信号，会导致：
- `docker stop` 等待 10 秒才强杀（数据可能丢）
- 子进程变僵尸（zombie）

**tini 是个微型 init 系统**：

```dockerfile
ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["./docker-entrypoint.sh"]
```

tini 接管 PID 1，正确处理信号，转发给子进程。`docker stop` 立刻优雅退出。

### 2.7 非 root 用户：最小权限

```dockerfile
RUN useradd --create-home --shell /bin/bash appuser
USER appuser
```

**为什么重要**？
- 即使容器被攻破，攻击者只有 appuser 权限
- 不能改容器内系统文件
- 不能装新软件
- 不能访问其他用户的文件

这是容器安全的"洋葱"模型——每多一层防御，攻击面缩小一截。

### 2.8 Volume 持久化：容器删了数据还在

```yaml
volumes:
  - ./data:/app/data   # bind mount
  - app_cache:/tmp/...  # named volume
```

**两种 volume**：

| 类型 | 写法 | 用途 |
|---|---|---|
| Bind mount | `./data:/app/data` | 把宿主机目录映射进容器，开发调试友好 |
| Named volume | `app_cache:/tmp/...` | Docker 管理的卷，生产推荐 |

**本项目的 `./data` 是 bind mount**——因为：
- 开发者可以直接看 `data/chroma/` 调试
- 容器删除后，宿主机还有
- 不需要 `docker volume create`

### 2.9 docker compose：编排多服务

```yaml
services:
  app:
    build: .
    ports: ["8000:8000"]
    env_file: .env
    volumes: [./data:/app/data]
    restart: unless-stopped
    healthcheck: {...}
```

学习项目只用 1 个服务（app）。但 compose 的威力在多服务：
- 加 Redis 缓存
- 加 PostgreSQL 元数据存储
- 加 Nginx 反向代理

只要在 `services:` 下加一段，**一条 `docker compose up` 就能起整条链**。

---

## 3. 文件结构变化

```
health-diet-rag/
├── Dockerfile                # ★ 新：多阶段构建
├── docker-compose.yml        # ★ 新：编排
├── .dockerignore             # ★ 新：排除敏感/无用文件
├── docker-entrypoint.sh      # ★ 新：启动脚本（建索引+启动 gunicorn）
├── gunicorn.conf.py          # ★ 新：生产 worker 配置
├── requirements.txt          # 改：加 gunicorn
├── .env.example              # 改：补全所有配置项
├── README.md                 # ★ 重写：产品级文档
└── tests/test_stage6.py      # ★ 新：39 个部署/安全测试
```

---

## 4. 容器启动完整流程

```
docker compose up
    ↓
docker-compose.yml 解析
    ↓
读取 .env（注入到运行时环境变量）
    ↓
docker build（首次）：
    ├─ Stage 1: builder 装依赖
    └─ Stage 2: runtime 拷 site-packages + 代码
    ↓
启动容器：
    1. tini 接管 PID 1
    2. docker-entrypoint.sh 执行
    3. 检查 ChromaDB 索引：
       ├─ 已有 → 跳过
       └─ 空 → 调 indexer.index_all()（下载 embedding 模型 + 建索引）
    4. 启动 gunicorn（4 个 worker）
    5. gunicorn worker 各自 import FastAPI app
    ↓
HEALTHCHECK 启动（60s start_period）
    ↓
curl /health 200 → 标记 healthy
    ↓
对外服务 :8000
```

---

## 5. 跑通验证

### 5.1 验证密钥安全（不依赖 Docker daemon）

```bash
# 静态检查
pytest tests/test_stage6.py::TestSecretSafety -v
```

### 5.2 本地构建 + 运行（如果 Docker Hub 能访问）

```bash
docker compose build
docker compose up

# 看日志
docker compose logs -f app

# 测试
curl http://localhost:8000/health
```

### 5.3 国内网络加速 Docker Hub

如果遇到 `context deadline exceeded`，配镜像加速器：

**Docker Desktop**：Settings → Docker Engine，加：

```json
{
  "registry-mirrors": [
    "https://docker.mirrors.ustc.edu.cn",
    "https://hub-mirror.c.163.com"
  ]
}
```

或者用阿里云镜像服务（每个账号有自己的 URL）。

---

## 6. 常见坑点

### 6.1 把 .env COPY 进 Dockerfile ❌

```dockerfile
COPY .env /app/.env  # ❌ 烤进镜像，泄漏！
```

正确：用 `.dockerignore` 排除 + `env_file` 运行时注入。

### 6.2 没有 HEALTHCHECK → Docker 不知道容器挂了

容器进程在跑，但应用死了（比如死锁），没有 HEALTHCHECK 时 Docker 还以为它健康。

### 6.3 用 root 跑 → 容器被攻破后能改系统文件

加一行 `USER appuser` 是最便宜的防御。

### 6.4 没设 start_period → 启动慢被误判挂掉

应用启动要建索引 30 秒，HEALTHCHECK 默认 0 秒就开始检查，立刻失败。

### 6.5 ENV 在 ARG 后面引用

```dockerfile
ARG API_KEY      # build 时传
ENV KEY=$API_KEY # ❌ 烤进镜像层
```

`ARG` 是 build 时变量，但如果赋给 `ENV`，就和直接写 `ENV` 一样危险。

### 6.6 bind mount 权限问题

`./data` 在宿主机是 root 创建的话，容器内的 `appuser` 可能读不了。

**解法**：在宿主机 `chown -R 1000:1000 ./data`（1000 通常是容器内 appuser 的 UID）。

---

## 7. 检查清单

完成本阶段后，你应该能：

- [ ] 解释多阶段构建为什么能减小镜像体积
- [ ] 解释 build 时 vs run 时的密钥安全区别
- [ ] 写一个安全的 Dockerfile（不泄漏 key）
- [ ] 配置 HEALTHCHECK + start_period
- [ ] 解释 gunicorn + uvicorn worker 的协作
- [ ] 解释 tini 为什么是 PID 1
- [ ] 用 docker compose up 一键启动
- [ ] 在 README 里写清楚一键启动步骤

---

## 8. 项目最终成就

完成 Stage 6 后，项目达到**"可交付"**状态：

✅ **简历可写**：所有关键词覆盖（RAG / 多 Agent / LangGraph / 向量库 / Docker）
✅ **GitHub 可推**：README 漂亮 + 一键启动
✅ **面试可演示**：5 秒内启动完整应用
✅ **部署可上**：任何装 Docker 的机器都能跑
✅ **密钥安全**：通过 7 项安全测试验证
✅ **测试完备**：147 个测试覆盖所有 Stage

---

## 9. 后续可探索方向（学习项目到此为止）

如果想继续精进，建议方向：

| 方向 | 学什么 |
|---|---|
| **生产级 K8s** | Helm chart / Ingress / ConfigMap / Secret |
| **可观测性** | Prometheus / Grafana / OpenTelemetry |
| **CI/CD** | GitHub Actions / 自动构建镜像 / 安全扫描 |
| **模型优化** | reranker / hybrid search / fine-tuning |
| **更复杂 Agent** | LangGraph 条件路由 / 并行 Node / 人机协作 |
| **真实数据** | 接 USDA API / 中国食物成分表 |

---

## 10. 关键代码索引

| 文件 | 内容 |
|---|---|
| `Dockerfile` | 多阶段构建（builder + runtime） |
| `.dockerignore` | 第一段就是 `.env`（安全第一） |
| `docker-compose.yml` | env_file 注入密钥 + volume 持久化 |
| `docker-entrypoint.sh` | 启动前自动建索引 |
| `gunicorn.conf.py` | 4 worker + 120s timeout |
| `tests/test_stage6.py` | 39 个部署/安全测试 |
