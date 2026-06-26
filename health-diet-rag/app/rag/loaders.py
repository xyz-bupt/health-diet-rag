"""
文档加载器。

把不同格式的数据源（JSON/Markdown）统一转换成 LangChain Document。
Document 是 RAG 流水线的"通用货币"：loader 产出 Document，splitter 切 Document，
embedder 向量化 Document 的 page_content，retriever 召回 Document。

为什么需要"扁平化"成文本？
--------------------------
向量检索靠的是语义相似度，需要把结构化数据拼成自然语言。
比如食材 {"name":"鸡胸肉", "protein":31, "tags":["高蛋白","低脂"]}
拼成："鸡胸肉。分类：肉禽蛋。特点：高蛋白、低脂。每 100g 含热量 133 kcal，
蛋白质 31g，碳水 0g，脂肪 1.2g。描述：经典低脂高蛋白肉类..."
这样 BGE 才能算出有意义的语义向量。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.core.config import settings


# ---------------------------------------------------------------------------
# 食材加载：JSON → Document（一条食材 = 一条 Document）
# ---------------------------------------------------------------------------

def load_ingredients(path: str | Path) -> list[Document]:
    """加载食材 JSON，每条食材生成一个 Document。"""
    data = _read_json(path)
    docs = []
    for item in data:
        n = item["nutrition_per_100g"]
        text = (
            f"【食材】{item['name']}\n"
            f"分类：{item['category']}\n"
            f"标签：{', '.join(item.get('tags', []))}\n"
            f"每 100g 营养：热量 {n['calories_kcal']} kcal / "
            f"蛋白质 {n['protein_g']}g / 碳水 {n['carbs_g']}g / 脂肪 {n['fat_g']}g\n"
            f"适用目标：{', '.join(item.get('suitable_for', []))}\n"
            f"常见做法：{', '.join(item.get('common_uses', []))}\n"
            f"说明：{item.get('description', '')}"
        )
        metadata = {
            "source": "ingredients",
            "name": item["name"],
            "category": item["category"],
            "tags": ",".join(item.get("tags", [])),
            "calories": n["calories_kcal"],
            "protein": n["protein_g"],
        }
        docs.append(Document(page_content=text, metadata=metadata))
    return docs


# ---------------------------------------------------------------------------
# 菜谱加载：JSON → Document（一条菜谱 = 一条 Document）
# ---------------------------------------------------------------------------

def load_recipes(path: str | Path) -> list[Document]:
    """加载菜谱 JSON，每条菜谱生成一个 Document。"""
    data = _read_json(path)
    docs = []
    for item in data:
        m = item["macros"]
        text = (
            f"【菜谱】{item['name']}\n"
            f"标签：{', '.join(item.get('tags', []))}\n"
            f"类型：{item.get('meal_type', '未分类')} / "
            f"准备时间 {item.get('prep_time_min', '?')} 分钟\n"
            f"每份热量：{item['calories_per_serving']} kcal "
            f"(蛋白 {m['protein_g']}g / 碳水 {m['carbs_g']}g / 脂肪 {m['fat_g']}g)\n"
            f"食材：{'; '.join(item.get('ingredients', []))}\n"
            f"步骤：{' '.join(item.get('steps', []))}\n"
            f"适用目标：{', '.join(item.get('suitable_for', []))}\n"
            f"说明：{item.get('description', '')}"
        )
        metadata = {
            "source": "recipes",
            "name": item["name"],
            "meal_type": item.get("meal_type", ""),
            "tags": ",".join(item.get("tags", [])),
            "calories": item["calories_per_serving"],
            "prep_time": item.get("prep_time_min", 0),
        }
        docs.append(Document(page_content=text, metadata=metadata))
    return docs


# ---------------------------------------------------------------------------
# Markdown 加载：先读全文 → 切片 → 每片一个 Document
# ---------------------------------------------------------------------------

def load_guides(path: str | Path) -> list[Document]:
    """加载 Markdown 指南，按章节切分成多个 Document。

    与食材/菜谱不同，指南是长文本，需要 splitter 切分：
    - RecursiveCharacterTextSplitter：递归按 [段落, 句子, 字符] 切
    - chunk_size + overlap 避免语义截断
    """
    text = Path(path).read_text(encoding="utf-8")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.RAG_CHUNK_SIZE,
        chunk_overlap=settings.RAG_CHUNK_OVERLAP,
        separators=["\n\n\n", "\n\n", "\n", "。", "；", " "],
    )
    chunks = splitter.split_text(text)
    docs = []
    for i, chunk in enumerate(chunks):
        # 尝试从 chunk 头部提取标题（## 开头）
        title = _extract_title(chunk)
        docs.append(Document(
            page_content=chunk.strip(),
            metadata={
                "source": "guides",
                "chunk_index": i,
                "title": title,
            },
        ))
    return docs


# ---------------------------------------------------------------------------
# 加载所有数据：聚合入口
# ---------------------------------------------------------------------------

def load_all(data_dir: str | Path | None = None) -> list[Document]:
    """加载 data/ 目录下的全部数据，返回 Document 列表。

    供 indexer 一次性建索引使用。
    """
    data_dir = Path(data_dir or "data")
    docs: list[Document] = []
    docs += load_ingredients(data_dir / "ingredients.json")
    docs += load_recipes(data_dir / "recipes.json")
    docs += load_guides(data_dir / "nutrition_guides.md")
    return docs


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

def _read_json(path: str | Path) -> list[dict[str, Any]]:
    """读取 JSON 文件，返回 list[dict]。"""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"数据文件不存在：{p}")
    return json.loads(p.read_text(encoding="utf-8"))


def _extract_title(chunk: str) -> str:
    """从 markdown chunk 里提取标题（第一个 # 行）。"""
    for line in chunk.split("\n"):
        line = line.strip()
        if line.startswith("#"):
            return line.lstrip("# ").strip()
    return ""
