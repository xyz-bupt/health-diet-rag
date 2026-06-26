"""
简易 TTL 缓存。

为什么需要缓存？
----------------
完整工作流要跑 5 个 Agent，单次约 1-3 秒（LLM 调用是瓶颈）。
用户经常因为刷新或重试触发相同请求——重复算很浪费。

缓存策略
--------
- Key：用户 profile 的哈希（model_dump_json → sha256）
- Value：完整 DietPlan
- TTL：5 分钟（300 秒）
- 上限：100 条（防内存无限增长）

进阶可换 Redis / Memcached（学习项目先用内存版）。

为什么不用 functools.lru_cache？
-------------------------------
1. lru_cache 不支持 TTL
2. lru_cache 的 key 必须可哈希，Pydantic 模型默认不行
3. 我们需要明确控制缓存粒度
"""

from __future__ import annotations

import hashlib
import time
from collections import OrderedDict
from typing import Any

from app.models.health import HealthProfile


class TTLCache:
    """带 TTL 的 LRU 缓存。

    用 OrderedDict 实现 LRU：访问/插入时移到末尾，满了从头删。
    """

    def __init__(self, maxsize: int = 100, ttl_seconds: int = 300) -> None:
        self.maxsize = maxsize
        self.ttl = ttl_seconds
        self._store: OrderedDict[str, tuple[float, Any]] = OrderedDict()
        # 统计信息（调试用）
        self.hits = 0
        self.misses = 0

    def get(self, key: str) -> Any | None:
        """取值。过期或不存在返回 None。"""
        if key not in self._store:
            self.misses += 1
            return None

        ts, value = self._store[key]
        # 检查 TTL
        if time.time() - ts > self.ttl:
            # 过期，删掉
            del self._store[key]
            self.misses += 1
            return None

        # LRU：访问后移到末尾
        self._store.move_to_end(key)
        self.hits += 1
        return value

    def set(self, key: str, value: Any) -> None:
        """写入。"""
        self._store[key] = (time.time(), value)
        self._store.move_to_end(key)
        # 超容量，从头删（最久未用）
        while len(self._store) > self.maxsize:
            self._store.popitem(last=False)

    def clear(self) -> None:
        self._store.clear()
        self.hits = 0
        self.misses = 0

    def stats(self) -> dict[str, int]:
        """返回缓存统计。"""
        return {
            "size": len(self._store),
            "maxsize": self.maxsize,
            "ttl_seconds": self.ttl,
            "hits": self.hits,
            "misses": self.misses,
        }


# ---------------------------------------------------------------------------
# 全局单例
# ---------------------------------------------------------------------------

_cache = TTLCache(maxsize=100, ttl_seconds=300)


def get_cache() -> TTLCache:
    return _cache


# ---------------------------------------------------------------------------
# 辅助：HealthProfile → cache key
# ---------------------------------------------------------------------------

def profile_cache_key(profile: HealthProfile) -> str:
    """把 HealthProfile 哈希成稳定的 cache key。

    用 model_dump_json 序列化后 sha256，保证：
    - 同内容同 key（确定性）
    - 不同内容几乎不会碰撞（sha256 抗冲突）
    """
    content = profile.model_dump_json()
    return hashlib.sha256(content.encode("utf-8")).hexdigest()
