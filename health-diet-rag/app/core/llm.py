"""
LLM 客户端工厂。

为什么需要"工厂"？
----------------------
项目里很多地方都会用到 LLM（健康评估 Agent、营养规划 Agent、菜谱 Agent……）。
如果每个 Agent 都自己 new 一个客户端，会有 3 个问题：
1. 配置散落：API Key、模型名到处写
2. 切换困难：想换模型/提供商时要改一堆地方
3. 无法测试：单测里没法替换成 Mock

工厂模式解决这 3 个问题：所有客户端由 `get_llm()` 统一创建，配置集中、切换方便、可注入。

降级策略
--------
没填 API Key 时（如刚 clone 项目跑测试），工厂返回一个 `MockLLM`，
让所有逻辑能在没有真实 LLM 的情况下跑起来。这是 RAG 项目里很重要的工程实践：
**LLM 是增强项，不是阻塞项**——核心业务逻辑应该能在没有 LLM 时降级运行。
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any, Protocol

from app.core.config import settings


# ---------------------------------------------------------------------------
# 1. LLM 协议（Protocol）：定义"什么是一个 LLM"
# ---------------------------------------------------------------------------

class LLMLike(Protocol):
    """所有 LLM 客户端（无论真实还是 Mock）都该满足的最小接口。

    LangChain 的 BaseChatModel 自带 invoke()，所以 ChatOpenAI 天然满足。
    我们自己写的 MockLLM 也实现 invoke()，就能无缝替换。
    """

    def invoke(self, messages: Any, **kwargs: Any) -> Any: ...


# ---------------------------------------------------------------------------
# 2. Mock LLM：没填 API Key 时用它兜底
# ---------------------------------------------------------------------------

class MockLLM:
    """占位 LLM：不调用任何外部 API，返回固定结构。

    用途：
    - 单元测试：完全确定性，不依赖网络
    - 首次跑项目：还没申请 DeepSeek key 时也能看到完整流程
    - CI/CD：避免在流水线里消耗 token

    返回的 content 是一个 JSON 字符串，故意做成与真实 LLM 输出结构一致，
    这样下游解析逻辑对真实/Mock 是同一份代码。
    """

    def __init__(self, model: str = "mock-llm") -> None:
        self.model = model

    def invoke(self, messages: Any, **kwargs: Any) -> Any:
        """返回一个伪装成 ChatResult 的简单对象。"""
        from langchain_core.messages import AIMessage

        # 这里只回一个简短文本；真实业务里 Agent 会自己组装更精细的 prompt，
        # Mock 时只关心"能跑通管道"，不关心内容质量。
        return AIMessage(content="[MockLLM] 未配置 DEEPSEEK_API_KEY，返回占位响应。")

    # 让 MockLLM 也可以像 ChatOpenAI 那样链式调用
    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.invoke(*args, **kwargs)


# ---------------------------------------------------------------------------
# 3. 工厂函数：根据配置返回合适的 LLM
# ---------------------------------------------------------------------------

def _build_real_llm() -> ChatOpenAI:  # type: ignore[name-defined]
    """创建真实的 DeepSeek ChatOpenAI 客户端。

    DeepSeek API 与 OpenAI 完全兼容，只需把 base_url 指向 DeepSeek 即可。
    其他参数（temperature / max_tokens）由调用方按需覆盖。
    """
    # 延迟 import：装依赖但不一定启用真实 LLM 时也能 import 本模块
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=settings.LLM_MODEL,
        api_key=settings.DEEPSEEK_API_KEY,
        base_url=settings.DEEPSEEK_BASE_URL,
        temperature=0.3,        # 健康建议要有一定稳定性，temperature 偏低
        max_tokens=1024,
        timeout=90,             # 长 prompt（含 RAG 上下文）需要更长时间
        max_retries=2,
    )


@lru_cache(maxsize=1)
def get_llm() -> LLMLike:
    """获取全局 LLM 客户端单例。

    - 配了真实 API Key → 返回 ChatOpenAI（DeepSeek）
    - 没配 Key（key 为空或仍是占位符）→ 返回 MockLLM

    用 lru_cache 保证整个进程只创建一次（连接复用、节省资源）。
    单元测试里可以调 `get_llm.cache_clear()` 重置。
    """
    key = settings.DEEPSEEK_API_KEY.strip()
    if not key or key == "your_api_key_here":
        return MockLLM()
    return _build_real_llm()


def is_llm_available() -> bool:
    """快速判断当前是否启用了真实 LLM。

    路由层和测试都用得上：知道是不是 Mock，可以决定要不要打 skip 标记。
    """
    key = settings.DEEPSEEK_API_KEY.strip()
    return bool(key) and key != "your_api_key_here"
