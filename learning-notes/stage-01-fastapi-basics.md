# Stage 1：FastAPI 项目骨架

> 目标：搭建可启动的 FastAPI 项目，理解 Web 框架的核心概念和工程化约定。
> 时长：约 1 天

---

## 1. 本阶段学什么

| 概念 | 为什么重要 |
|---|---|
| FastAPI 基本结构 | 后续所有接口都基于此 |
| Pydantic Settings | 配置管理标准做法 |
| 应用工厂模式 | 测试和扩展的基础 |
| Lifespan 事件 | 启动/关闭钩子，后续要初始化向量库 |
| TestClient | 不发真实 HTTP 也能测接口 |
| 项目目录约定 | 模块化，便于扩展到多 Agent |

---

## 2. FastAPI 是什么

FastAPI 是基于 Python 类型提示的现代 Web 框架。三大卖点：

1. **快**：基于 Starlette + Pydantic，性能接近 Node.js / Go
2. **类型安全**：用 Python 类型提示自动校验请求/响应
3. **自动文档**：根据类型提示自动生成 Swagger / ReDoc 文档

**对比 Flask**：
- Flask 要手写校验，FastAPI 用类型提示自动校验
- Flask 没有 Async，FastAPI 原生支持 async/await
- Flask 自动文档要装扩展，FastAPI 内置

---

## 3. 核心概念讲解

### 3.1 路由（Router）

路由就是把 URL 路径绑定到一个函数。FastAPI 用装饰器语法：

```python
from fastapi import APIRouter

router = APIRouter()

@router.get("/health")
async def health_check():
    return {"status": "healthy"}
```

- `@router.get(...)` 表示处理 GET 请求（还有 post/put/delete）
- `async def` 是异步函数，FastAPI 推荐用它
- 返回字典会被自动转成 JSON

**为什么要用 APIRouter 而不是直接 `@app.get`**？
- 模块化：每个业务一个文件，便于维护
- 可移植：同一路由可挂到多个 app 上
- 分组：可以统一加前缀、标签、依赖

### 3.2 配置管理（Pydantic Settings）

老式做法：用 `os.getenv("KEY")` 一行行读，容易出错。

新式做法（本项目采用）：定义一个 `Settings` 类，自动从 `.env` 加载：

```python
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env")
    APP_NAME: str = "默认名"
    PORT: int = 8000

settings = Settings()  # 全局单例
```

**好处**：
- 类型安全：`PORT: int` 写明类型，传字符串会报错
- 默认值：未设置时用默认值
- 自动加载 `.env` 文件
- 全局单例：导入即用，不用到处传参

### 3.3 应用工厂模式

```python
def create_app() -> FastAPI:
    app = FastAPI(...)
    app.include_router(health.router)
    return app

app = create_app()
```

为什么不直接 `app = FastAPI()`？
- **测试隔离**：测试时可创建新实例，避免污染
- **配置灵活**：根据环境（dev/prod）创建不同 app
- **顺序控制**：先创建实例，再依次添加中间件、路由、事件

### 3.4 Lifespan 事件

应用启动和关闭时需要做的工作（比如连数据库、加载模型）：

```python
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    # 启动时执行
    print("启动中...")
    yield  # 应用运行期间停在这里
    # 关闭时执行
    print("已关闭")

app = FastAPI(lifespan=lifespan)
```

> 注意：旧的 `@app.on_event("startup")` 已废弃，统一用 lifespan。
> 本项目后续会在这里：初始化 ChromaDB、预热 LLM 客户端。

### 3.5 TestClient

测试时不发真实 HTTP 请求，但能完整测试路由逻辑：

```python
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
```

底层是 httpx + ASGI 协议直连，速度快、无需端口。

---

## 4. 项目结构约定

```
app/
├── main.py              # 入口（create_app + 全局 app 实例）
├── core/                # 核心基础设施（配置、LLM 客户端、日志）
├── api/v1/              # API 路由（v1 表示第一版，便于版本管理）
├── agents/              # Agent 实现（Stage 2+）
├── rag/                 # RAG 组件（Stage 3+）
└── graph/               # LangGraph 编排（Stage 4+）
```

**为什么按业务领域分目录而不是按类型？**
- 改一个 Agent 时不用跨多个目录找文件
- 删功能时整目录删
- 便于扩展：加新 Agent 直接新建文件

---

## 5. 启动流程

### 5.1 安装依赖

```bash
cd health-diet-rag
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 5.2 配置环境

```bash
cp .env.example .env
# Stage 1 不用填 API Key，默认值就能跑
```

### 5.3 启动服务

```bash
uvicorn app.main:app --reload
```

参数说明：
- `app.main:app`：`app/main.py` 文件里的 `app` 变量
- `--reload`：代码改动自动重启（开发用，生产不要加）

### 5.4 访问接口

打开浏览器：
- http://localhost:8000/ → 根路径
- http://localhost:8000/health → 健康检查
- http://localhost:8000/docs → **Swagger UI**（最有用，可以在线试接口）
- http://localhost:8000/redoc → ReDoc（另一种风格的文档）

### 5.5 运行测试

```bash
pytest tests/ -v
```

---

## 6. 常见坑点

### 6.1 `ModuleNotFoundError: No module named 'app'`

原因：不在项目根目录运行。
解决：必须在 `health-diet-rag/` 目录下执行 `uvicorn`。

### 6.2 端口被占用

报错：`[Errno 48] address already in use`
解决：换端口 `uvicorn app.main:app --port 8001`

### 6.3 `.env` 没加载

检查 `.env` 文件是否在项目根目录，编码是否 UTF-8。

---

## 7. 检查清单

完成本阶段后，你应该能：

- [ ] 解释 FastAPI 的三个卖点
- [ ] 解释为什么用 APIRouter 而不是 `@app.get`
- [ ] 写一个最小可运行的 FastAPI 应用
- [ ] 解释应用工厂模式的好处
- [ ] 解释 lifespan 中 yield 前后代码的执行时机
- [ ] 用 TestClient 写一个测试用例
- [ ] 访问 `/docs` 看到自动生成的 Swagger

---

## 8. 下一阶段预告

Stage 2 会做：
- 接入智谱 GLM SDK
- 实现「健康评估 Agent」：输入身高体重，让 LLM 计算并解释 BMR/TDEE
- 学会用 Pydantic 校验 LLM 的输出（结构化输出）
