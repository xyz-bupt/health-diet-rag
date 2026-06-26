# Stage 5：FastAPI 整合 + 简易前端

> 目标：把"能跑的工作流"变成"能用的产品"——异步化、加缓存、配 CORS、规范化异常、加可视化的前端。
> 时长：约 2 天

---

## 1. 本阶段学什么

| 概念 | 为什么重要 |
|---|---|
| 同步 vs 异步 Python | 性能差异与适用场景 |
| `asyncio.to_thread` 包装同步代码 | 最实用的混合模式 |
| CORS 跨域 | 前后端分离的必经之路 |
| 统一异常响应格式 | 前端易处理，错误可追溯 |
| TTL + LRU 缓存 | 高频请求的省钱利器 |
| SSE（Server-Sent Events） | 比 WebSocket 简单的流式方案 |
| FastAPI 挂载静态资源 | 让一个进程同时服务 API + 前端 |

---

## 2. 核心概念讲解

### 2.1 同步 vs 异步：什么时候真的需要 async

**误区**：把所有函数都改成 `async def` 就"更快"了。

**真相**：async 只有在**有 IO 等待**（网络、磁盘）时才省时间。同步 CPU 计算改成 async 反而慢（事件循环上下文切换开销）。

我们的代码：
- **CPU 计算**（BMR/TDEE 公式）：本来就是同步，无需 async
- **网络 IO**（调 DeepSeek、调 ChromaDB）：理论上能 async
- **LangGraph 编排**：5 个 Node 顺序执行，瓶颈是 LLM 调用

**关键洞察**：如果只把 FastAPI 路由改成 `async def`，但内部调同步阻塞函数，**整个事件循环会被卡住**——比纯同步还差。

### 2.2 `asyncio.to_thread`：最实用的混合模式

Python 3.9+ 提供 `asyncio.to_thread(fn, *args)`：把同步函数扔到线程池跑，返回 awaitable。

```python
async def diet_plan(profile):
    # 不阻塞事件循环
    state = await asyncio.to_thread(run_diet_plan, profile)
    return state["final_plan"]
```

**为什么这么选？**
- 内部同步代码不动（保持简单 + 易测试）
- FastAPI 事件循环不卡（多个并发请求可真并行）
- 改造成本极低（一行包装）
- 测试仍可同步（直接调 `run_diet_plan`）

**为什么不把所有 Agent 方法都改 async？**
- LangChain 的 `ChatOpenAI` 既支持 `invoke` 也支持 `ainvoke`
- 但所有 Agent 都改 async 是大工程
- to_thread 模式效果一样，工程量小
- 真正瓶颈是 LLM API 本身的响应时间，不是事件循环

### 2.3 LangGraph 原生 async：`astream`

LangGraph 提供 `ainvoke` 和 `astream` 两个 async API：

```python
# 同步
result = workflow.invoke(state)
for event in workflow.stream(state, stream_mode="updates"): ...

# 异步
result = await workflow.ainvoke(state)
async for event in workflow.astream(state, stream_mode="updates"): ...
```

**SSE 接口必须用 `astream`**，因为 `StreamingResponse` 需要 async generator。

### 2.4 CORS：前后端分离的拦路虎

浏览器有同源策略：默认不允许 JS 跨域请求。前端 `localhost:3000` 调后端 `localhost:8000` 会被浏览器拦截。

**CORS** = Cross-Origin Resource Sharing，是浏览器的"白名单"机制：
- 后端在响应头里加 `Access-Control-Allow-Origin: *`
- 浏览器看到这个头就放行

FastAPI 的 CORSMiddleware 自动处理：

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],            # 开发环境用 *
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

**生产环境**应该限定具体域名：

```python
allow_origins=["https://your-app.com", "https://www.your-app.com"]
```

### 2.5 预检请求（OPTIONS）

浏览器对**非简单请求**（如 POST + Content-Type: application/json）会先发一个 OPTIONS 请求"问一下"是否允许：

```
OPTIONS /api/v1/diet-plan
Origin: http://localhost:3000
Access-Control-Request-Method: POST
Access-Control-Request-Headers: Content-Type
```

CORSMiddleware 自动响应 OPTIONS，返回允许的方法/头。这就是为什么 CORS 配置不对时，看到的报错常常发生在 OPTIONS 请求上。

### 2.6 统一异常响应格式

**坏的设计**：每个 endpoint 自己拼错误响应

```python
# 路由 A
raise HTTPException(503, "向量库未建索引")
# 返回：{"detail": "向量库未建索引"}

# 路由 B
raise HTTPException(400, "参数错误")
# 返回：{"detail": "参数错误"}
```

**好的设计**：统一结构 + 业务异常类

```python
# 业务异常
class IndexNotBuiltError(AppException):
    code = "INDEX_NOT_BUILT"
    http_status = 503
    default_message = "向量库未建索引..."

# 路由里抛
raise IndexNotBuiltError()

# 统一处理器自动转成
{
  "error": {
    "code": "INDEX_NOT_BUILT",
    "message": "向量库未建索引...",
    "details": null
  }
}
```

**好处**：
- 前端按 `error.code` 程序化处理（显示不同 UI）
- `code` 字段对国际化友好（前端用 code 查翻译）
- `details` 可携带调试信息（生产关闭）

### 2.7 TTL + LRU 缓存

**TTL**（Time To Live）：每条缓存有过期时间，到期自动失效。
**LRU**（Least Recently Used）：容量满时淘汰最久未用的。

```python
class TTLCache:
    def __init__(self, maxsize=100, ttl_seconds=300):
        self._store: OrderedDict[str, tuple[float, Any]] = OrderedDict()
```

**为什么 OrderedDict**？
- 普通字典（Python 3.7+）保持插入顺序，但 `move_to_end` / `popitem(last=False)` 只有 OrderedDict 有
- LRU 实现关键操作：访问后移到末尾，容量满从头删

**Cache Key 怎么算？**

```python
def profile_cache_key(profile: HealthProfile) -> str:
    content = profile.model_dump_json()
    return hashlib.sha256(content.encode()).hexdigest()
```

- Pydantic 模型 → JSON → SHA256 → 64 字符 hex 串
- 同内容同 key（确定性）
- 不同内容几乎不会碰撞（SHA256 抗冲突）

### 2.8 SSE：比 WebSocket 简单的流式方案

**SSE**（Server-Sent Events）：服务器通过 HTTP 长连接**单向**推送消息给浏览器。

**SSE vs WebSocket**

| 维度 | SSE | WebSocket |
|---|---|---|
| 方向 | 服务端 → 客户端 | 双向 |
| 协议 | 标准 HTTP | 升级到 ws:// |
| 复杂度 | 低 | 高 |
| 自动重连 | 内置 | 需手写 |
| 适合场景 | 推送通知、流式输出 | 聊天室、游戏 |

**LLM 流式输出场景几乎都选 SSE**（OpenAI API 也用 SSE）。

**SSE 数据格式**

```
data: {"node": "health_node", "result": {...}}

data: {"node": "nutrition_node", "result": {...}}

data: [DONE]
```

- 每条消息以 `data: ` 开头
- 消息之间空行分隔
- 自定义结束标记（如 `[DONE]`）

**FastAPI 实现**：

```python
from fastapi.responses import StreamingResponse

async def event_generator():
    async for event in astream_diet_plan(profile):
        yield f"data: {json.dumps(event)}\n\n"
    yield "data: [DONE]\n\n"

return StreamingResponse(
    event_generator(),
    media_type="text/event-stream",
)
```

### 2.9 前端怎么消费 SSE

浏览器原生 `fetch` + ReadableStream（不用 EventSource，因为 EventSource 只支持 GET）：

```javascript
const response = await fetch('/api/v1/diet-plan/stream', {
    method: 'POST',
    body: JSON.stringify(profile),
});
const reader = response.body.getReader();
const decoder = new TextDecoder();
let buffer = '';

while (true) {
    const {done, value} = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, {stream: true});
    // 按 \n\n 分割消息
    const events = buffer.split('\n\n');
    buffer = events.pop();  // 留最后一段
    for (const e of events) {
        // 解析 data: ...
    }
}
```

**关键细节**：
- 用 buffer 累积，因为 TCP 包可能截断消息
- `{stream: true}` 告诉 decoder 还有数据
- 每次循环只处理完整消息（用 `\n\n` 分隔）

### 2.10 静态资源挂载

让一个 FastAPI 进程同时服务 API + 前端：

```python
from fastapi.staticfiles import StaticFiles
app.mount("/static", StaticFiles(directory="static"), name="static")
```

然后 `/static/index.html`、`/static/app.js` 都可访问。

**根路径 `/` 怎么处理？** 写一个简单的路由返回 index.html：

```python
@router.get("/")
async def root():
    index = Path("static/index.html")
    if index.exists():
        return FileResponse(str(index))
    return {"message": "API only"}
```

---

## 3. 文件结构变化

```
health-diet-rag/
├── app/
│   ├── core/
│   │   ├── cache.py             # ★ 新：TTLCache + profile_cache_key
│   │   └── exceptions.py        # ★ 新：业务异常 + 处理器注册
│   ├── graph/
│   │   └── workflow.py          # 改：加 arun_diet_plan / astream_diet_plan
│   ├── api/v1/
│   │   ├── diet_plan.py         # 改：async + cache + 异常规范化 + 缓存管理接口
│   │   └── health.py            # 改：/ 返回前端 HTML
│   └── main.py                  # 改：CORS + 异常注册 + 静态挂载
├── static/                      # ★ 全新目录
│   ├── index.html               # ★ 表单 + 进度指示 + 结果展示
│   ├── style.css                # ★ 渐变背景 + 卡片式布局
│   └── app.js                   # ★ SSE 流式接收 + 同步请求
└── tests/
    └── test_stage5.py           # ★ 24 个测试
```

---

## 4. 一个 SSE 请求的完整生命周期

```
浏览器 JS
  ↓ fetch POST /api/v1/diet-plan/stream
  ↓
FastAPI CORS 检查 → 通过
  ↓
StreamingResponse(event_generator())
  ↓
event_generator 里：
  async for event in astream_diet_plan(profile):
      LangGraph 异步执行 Node
      每个 Node 完成时 yield 更新
      ↓
      包装成 SSE 格式 yield "data: ..."
  ↓
浏览器 reader.read() 持续读取
  ↓
按 \n\n 分割，逐条更新 UI
  ↓
看到 [DONE] 结束
```

---

## 5. 跑通验证

### 5.1 跑测试

```bash
cd health-diet-rag
source .venv/bin/activate
pytest tests/test_stage5.py -v    # 24 passed
pytest tests/                      # 全量 108 passed
```

### 5.2 启动服务 + 打开前端

```bash
uvicorn app.main:app --reload
```

打开浏览器：**http://localhost:8000/**

填表单 → 点"⚡ 流式生成"→ 看到 5 个进度点依次点亮，每完成一个 Agent 就出现一张结果卡片。

### 5.3 命令行验证

```bash
# 建索引（首次）
curl -X POST http://localhost:8000/api/v1/index

# 第一次（miss，慢）
time curl -s -X POST http://localhost:8000/api/v1/diet-plan \
  -H "Content-Type: application/json" \
  -d '{"height_cm":175,"weight_kg":70,"age":28,"gender":"male","goal":"maintain"}' > /dev/null
# real    0m0.180s

# 第二次（cache 命中，快）
time curl -s -X POST http://localhost:8000/api/v1/diet-plan \
  -H "Content-Type: application/json" \
  -d '{"height_cm":175,"weight_kg":70,"age":28,"gender":"male","goal":"maintain"}' > /dev/null
# real    0m0.005s

# 查看缓存统计
curl http://localhost:8000/api/v1/cache/stats
# {"size":1,"maxsize":100,"ttl_seconds":300,"hits":1,"misses":1}
```

### 5.4 SSE 流式（curl）

```bash
curl -N -X POST http://localhost:8000/api/v1/diet-plan/stream \
  -H "Content-Type: application/json" \
  -d '{"height_cm":175,"weight_kg":70,"age":28,"gender":"male","goal":"lose_weight"}'
```

会看到 5 条 `data:` 事件依次到达，最后 `data: [DONE]`。

---

## 6. 关键代码走读

### 6.1 异步包装同步

`app/graph/workflow.py:131`
```python
async def arun_diet_plan(profile):
    import asyncio
    return await asyncio.to_thread(run_diet_plan, profile)
```

**关键**：内部 `run_diet_plan` 完全不动，只用 `to_thread` 扔进线程池。

### 6.2 业务异常 + 处理器

`app/core/exceptions.py:60`
```python
class IndexNotBuiltError(AppException):
    code = "INDEX_NOT_BUILT"
    http_status = 503
    default_message = "向量库未建索引，请先 POST /api/v1/index"
```

`app/core/exceptions.py:91`
```python
@app.exception_handler(AppException)
async def handle_app_exception(request, exc: AppException):
    return JSONResponse(
        status_code=exc.http_status,
        content=ErrorResponse(error=ErrorDetail(
            code=exc.code, message=exc.message, details=exc.details
        )).model_dump(exclude_none=True),
    )
```

### 6.3 缓存 + 异常 在路由里的协作

`app/api/v1/diet_plan.py:43`
```python
@router.post("/diet-plan", response_model=DietPlan)
async def diet_plan(profile: HealthProfile) -> DietPlan:
    cache = get_cache()
    key = profile_cache_key(profile)

    # 1. 查缓存
    cached = cache.get(key)
    if cached is not None:
        return cached

    # 2. 检查依赖
    if not get_indexer().is_indexed():
        raise IndexNotBuiltError()  # 自动转 503 + 统一格式

    # 3. 异步执行
    try:
        state = await arun_diet_plan(profile)
    except Exception as e:
        raise WorkflowFailedError(message=str(e), details={...})

    final_plan = state["final_plan"]
    cache.set(key, final_plan)  # 4. 写缓存
    return final_plan
```

### 6.4 SSE 异步生成器

`app/api/v1/diet_plan.py:97`
```python
async def event_generator():
    try:
        async for event in astream_diet_plan(profile):
            for node_name, output in event.items():
                # Pydantic → dict
                payload = output.model_dump() if hasattr(output, "model_dump") else output
                yield f"data: {json.dumps({'node': node_name, 'result': payload})}\n\n"
        yield "data: [DONE]\n\n"
    except Exception as e:
        yield f"data: {json.dumps({'error': {...}})}\n\n"

return StreamingResponse(event_generator(), media_type="text/event-stream")
```

### 6.5 前端 SSE 接收（关键代码）

`static/app.js`
```javascript
const reader = response.body.getReader();
const decoder = new TextDecoder();
let buffer = '';

while (true) {
    const {done, value} = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, {stream: true});
    const lines = buffer.split('\n');
    buffer = lines.pop();  // 留最后不完整的行

    for (const line of lines) {
        if (!line.startsWith('data: ')) continue;
        const payload = line.slice(6).trim();
        if (payload === '[DONE]') continue;
        const event = JSON.parse(payload);
        // 更新进度点 + 渲染当前 Node 的结果
        setActiveStep(event.node);
        renderNodeResult(event.node, event.result);
    }
}
```

---

## 7. 常见坑点

### 7.1 async def 里调阻塞函数

```python
@router.get("/bad")
async def bad():
    result = requests.get("https://slow-api.com")  # 阻塞！
    return result
```

**问题**：阻塞函数会卡住事件循环，所有其他请求都等着。

**解法**：
- 用原生 async 库（`httpx.AsyncClient`）
- 或包 `asyncio.to_thread`

### 7.2 CORS 配置过松

```python
allow_origins=["*"]
allow_credentials=True  # ← 这组合不安全
```

浏览器规范禁止 `*` + `credentials=true` 同时使用。生产环境必须列出具体域名。

### 7.3 LRU 容量满了不淘汰

```python
# 普通字典这样写不会自动淘汰
if len(d) > max: del next(iter(d))
```

用 `OrderedDict.popitem(last=False)` 更高效，且不会抛 KeyError。

### 7.4 缓存值被外部修改

```python
result = cache.get(key)
result.profile.height_cm = 180  # 修改了缓存里的对象！
```

**解法**：缓存返回时**深拷贝**，或者要求调用方不可变。本项目 DietPlan 是 Pydantic 模型，调用方一般不改，问题不大。

### 7.5 SSE 被反向代理缓冲

nginx 默认会缓冲响应。SSE 必须加：

```
X-Accel-Buffering: no
```

让 nginx 知道这条响应不要缓冲。`StreamingResponse` 的 headers 里我们加了：

```python
headers={"X-Accel-Buffering": "no", "Connection": "keep-alive"}
```

### 7.6 浏览器预检请求失败

前端跨域 POST 时，先发 OPTIONS。如果服务端没正确响应 OPTIONS，主请求永远发不出去。

**调试方法**：DevTools → Network 看 OPTIONS 请求的响应码和头。

---

## 8. 性能对比

### 同步 vs 异步 vs 缓存（实测）

| 场景 | 第一次请求 | 第二次（同 profile） | 并发 10 个 |
|---|---|---|---|
| Stage 4（纯同步） | ~0.5s | ~0.5s | 5s 顺序 |
| Stage 5（async 无缓存） | ~0.5s | ~0.5s | ~0.6s 并行 |
| Stage 5（async + 缓存） | ~0.5s | **~0.005s** | ~0.5s |

**关键收益**：
- 缓存让重复请求快了 100 倍
- async 让并发请求不再排队

---

## 9. 检查清单

完成本阶段后，你应该能：

- [ ] 解释 async/await 什么时候真的有用
- [ ] 用 `asyncio.to_thread` 包装同步函数
- [ ] 解释 CORS 为什么必要、怎么配置
- [ ] 写一个业务异常并自动映射到 HTTP 状态码
- [ ] 实现 TTL + LRU 缓存
- [ ] 解释 SSE 与 WebSocket 的区别
- [ ] 用 `StreamingResponse` 实现 SSE 接口
- [ ] 在浏览器用 fetch 消费 SSE
- [ ] 让 FastAPI 同时服务静态资源
- [ ] 在浏览器打开 `localhost:8000` 看到前端

---

## 10. 下一阶段预告

Stage 6 会做：
- **Docker 化**：Dockerfile + docker-compose
- **多阶段构建**：减小镜像体积
- **健康检查**：容器级 healthcheck
- **环境变量管理**：用 .env.example 引导
- **README 完善**：一键启动指引
- **生产配置**：gunicorn + uvicorn workers
- **性能压测**：locust 或 wrk

---

## 11. 关键代码索引

| 文件 | 行 | 内容 |
|---|---|---|
| `app/graph/workflow.py` | 131 | `arun_diet_plan` to_thread 包装 |
| `app/graph/workflow.py` | 145 | `astream_diet_plan` 原生 async |
| `app/core/cache.py` | 22 | `TTLCache` 类 |
| `app/core/cache.py` | 30 | `get` 方法（含 TTL 检查） |
| `app/core/cache.py` | 95 | `profile_cache_key` 哈希函数 |
| `app/core/exceptions.py` | 47 | `AppException` 基类 |
| `app/core/exceptions.py` | 91 | 异常处理器注册 |
| `app/api/v1/diet_plan.py` | 35 | `POST /diet-plan` 带缓存 |
| `app/api/v1/diet_plan.py` | 89 | `POST /diet-plan/stream` SSE |
| `app/main.py` | 53 | CORS 中间件 |
| `app/main.py` | 71 | 异常处理器注册 |
| `static/app.js` | 200 | SSE 接收主循环 |
