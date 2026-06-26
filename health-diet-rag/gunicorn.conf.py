# =============================================================================
# Gunicorn 配置（生产环境用）
# =============================================================================
# 启动: gunicorn app.main:app --config gunicorn.conf.py
#
# 关键参数说明：
#   workers    = 进程数，公式 (2 * CPU) + 1 是经典推荐
#   worker_class = uvicorn.UvicornWorker，让 gunicorn 管理 uvicorn（FastAPI 需要）
#   timeout    = worker 处理单个请求的最长时间，LLM 调用慢，设 120s
#   grace      = 优雅退出宽限期，让正在处理的请求完成
# =============================================================================

import multiprocessing
import os

# 绑定地址（被 entrypoint 的 --bind 覆盖；这里只是默认）
bind = "0.0.0.0:8000"

# worker 数量：默认按 CPU 数 * 2 + 1
# 可以通过环境变量 GUNICORN_WORKERS 覆盖
workers = int(os.environ.get("GUNICORN_WORKERS", min(4, multiprocessing.cpu_count() * 2 + 1)))

# uvicorn worker（FastAPI 必须用这个）
worker_class = "uvicorn.workers.UvicornWorker"

# 单请求超时（LLM 调用偶尔较慢，给足时间）
timeout = 120

# 优雅退出宽限期
graceful_timeout = 30

# 长连接保持时间（FastAPI 流式响应需要）
keepalive = 5

# 日志
accesslog = "-"          # stdout
errorlog = "-"           # stderr
loglevel = os.environ.get("LOG_LEVEL", "info")
access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s" %(L)s秒'

# 进程名（在 docker exec ps 里能看到）
proc_name = "health-diet-rag"

# 预加载应用（节省内存 + 启动时间，但要确保代码无副作用）
preload_app = False  # 因为 entrypoint 已经建索引了，不重复

# 最大请求数（防内存泄漏，每 worker 处理 1000 个请求后重启）
max_requests = 1000
max_requests_jitter = 50

# 优雅停止信号
timeout_graceful_shutdown = True
