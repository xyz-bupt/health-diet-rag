"""
应用配置模块。

使用 pydantic-settings 从环境变量 / .env 文件加载配置。
这是 FastAPI 项目的标准做法，好处：
1. 类型安全：每个配置项都有类型提示
2. 自动验证：缺失必填项会启动时报错
3. 环境隔离：开发/生产用不同 .env 文件即可
"""

from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """应用配置。所有字段都会自动从 .env 文件或环境变量读取。"""

    # 模型配置：让 pydantic 知道去哪里读 .env
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",  # .env 里的额外字段不报错
    )

    # === 应用基础配置 ===
    APP_NAME: str = "健康饮食 RAG 助手"
    APP_VERSION: str = "0.1.0"
    DEBUG: bool = True
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    ENV: Literal["dev", "prod", "test"] = "dev"

    # === DeepSeek LLM 配置（Stage 2 起使用）===
    # DeepSeek API 完全兼容 OpenAI 格式，所以用 langchain-openai 接入
    # 申请地址：https://platform.deepseek.com/api_keys
    DEEPSEEK_API_KEY: str = ""
    DEEPSEEK_BASE_URL: str = "https://api.deepseek.com"
    LLM_MODEL: str = "deepseek-chat"  # 备选：deepseek-reasoner（R1 推理模型，更慢但更强）

    # === Embedding 配置（Stage 3 起使用）===
    # 用 fastembed（ONNX 运行时）跑本地 BGE 模型，无需 torch
    # bge-small-zh-v1.5: 90MB / 512 维 / 中文优化 / CPU 上极快
    EMBEDDING_MODEL: str = "BAAI/bge-small-zh-v1.5"
    EMBEDDING_PROVIDER: str = "fastembed"  # fastembed | zhipuai | openai

    # === RAG 检索配置 ===
    RAG_TOP_K: int = 4                 # 默认召回数量
    RAG_CHUNK_SIZE: int = 500          # 文档切分大小（字符）
    RAG_CHUNK_OVERLAP: int = 50        # 切分重叠（防止语义截断）

    # === 向量库配置（Stage 3 才会用）===
    CHROMA_PERSIST_DIR: str = "./data/chroma"
    CHROMA_COLLECTION: str = "health_diet"


# 全局单例：整个应用共用一个 Settings 实例
settings = Settings()
