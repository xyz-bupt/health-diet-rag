# Stage 3：RAG 检索增强基础

> 目标：建一个能语义检索的健康知识库，输入"低脂高蛋白早餐"就能召回相关食材/菜谱/指南。
> 时长：约 3 天

---

## 1. 本阶段学什么

| 概念 | 为什么重要 |
|---|---|
| Embedding 是什么 | RAG 的数学基础 |
| 为什么不用 torch + sentence-transformers | Python 3.13 的依赖现实 |
| fastembed（ONNX）替代方案 | CPU 上更轻更快的路径 |
| 文档加载与"扁平化" | 把结构化数据转成向量库能消化的形式 |
| ChromaDB 向量库 | RAG 的存储与检索核心 |
| 距离 vs 相似度 | cosine / L2 / 点积的取舍 |
| 元数据过滤 | 在向量召回前先用条件筛 |
| 检索质量评估 | 知道"召回得好不好" |

---

## 2. 核心概念讲解

### 2.1 Embedding：把文字变成向量

**直觉**：让计算机理解"鸡胸肉"和"瘦牛肉"语义接近，但和"牛仔裤"语义远。

**做法**：训练一个模型，把任意文本映射到一个固定维度的浮点向量（比如 512 维），让**语义相近的文本在向量空间中也相近**。

```
"鸡胸肉"     → [0.12, -0.05, 0.34, ..., 0.21]  (512 维)
"瘦牛肉"     → [0.15, -0.04, 0.31, ..., 0.19]  (与上面接近)
"牛仔裤"     → [-0.40, 0.88, 0.02, ..., -0.55] (与上面差异大)
```

**怎么衡量"向量相近"？** 余弦相似度（Cosine Similarity）：

```
similarity = (A·B) / (|A| × |B|)
```

值域 [-1, 1]，越接近 1 越相似。**BGE 系列就是用这种方式训练的中文嵌入模型**。

### 2.2 为什么不用 sentence-transformers + torch

PyTorch 是 sentence-transformers 的运行时依赖。截至 2026-06：
- **PyTorch 没有 Python 3.13 的 macOS wheel**
- 即使有，torch 包大小 ~800MB（对学习项目太重）

**解法**：用 **fastembed**
- 基于 ONNX Runtime（CPU 优化，无 torch 依赖）
- 预打包主流 BGE / MiniLM 模型
- 包大小 ~50MB，模型文件 90MB
- 在 CPU 上推理速度反而比 PyTorch 快（ONNX 计算图优化）

**核心权衡**：
- sentence-transformers：生态最全，模型多，但重
- fastembed：模型精选，轻量，CPU 友好

学习项目选 fastembed 完全正确。

### 2.3 BGE-small-zh-v1.5 vs BGE-M3

| 维度 | bge-small-zh-v1.5 | bge-m3 |
|---|---|---|
| 模型大小 | 90 MB | 2.2 GB |
| 向量维度 | 512 | 1024 |
| 多语言 | 仅中文 | 100+ 语言 |
| CPU 单条推理 | ~50ms | ~500ms |
| 长文档 | ≤512 token | ≤8192 token |
| 中文检索质量 | 优秀 | 略胜 |

对学习项目（20 食材+10 菜谱+1 篇指南），**bge-small-zh 又快又够用**。换 bge-m3 只需改一行配置。

### 2.4 文档"扁平化"：让结构化数据能被向量化

向量库只能索引文本，不能直接索引 JSON。所以要把结构化数据拼成自然语言：

```python
# 原始 JSON
{"name": "鸡胸肉", "protein_g": 31, "tags": ["高蛋白", "低脂"]}

# 扁平化文本
"【食材】鸡胸肉
分类：肉禽蛋
标签：高蛋白, 低脂
每 100g 营养：热量 133 kcal / 蛋白质 31g / 碳水 0g / 脂肪 1.2g
..."

# 元数据（保留原结构，用于过滤）
{"source": "ingredients", "name": "鸡胸肉", "calories": 133}
```

**关键设计**：
- `page_content`：自然语言，会被 embedding
- `metadata`：结构化字段，**不会被 embedding**，用于过滤和展示

### 2.5 ChromaDB：本地向量库

**核心抽象**：

```
PersistentClient (path=./data/chroma)
     ↓
Collection (name="health_diet", metric=cosine)
     ↓
Document { id, text, embedding, metadata }
```

**为什么选 ChromaDB**：
- pip 装完即可，不用起服务
- 数据持久化到本地 SQLite + Parquet
- LangChain 原生支持（但我们绕过了，见下文）

**为什么本项目绕过 langchain-chroma？**

`langchain-chroma 0.1.x` 锁了 `chromadb<0.6`，但 `chromadb<0.6` 在 Python 3.13 上又和 numpy 2.x 冲突。直接用 `chromadb` 官方客户端反而：
- 解决依赖地狱
- 少一层抽象，更易理解
- 后续迁移到 FAISS / Qdrant 改一个文件即可

### 2.6 距离 vs 相似度

ChromaDB 的 `query()` 返回 `distances`（距离），越小越相似。
但用户习惯看"相似度"（0-1，越大越相似）。

转换公式（cosine 模式下）：

```python
similarity = 1.0 - distance
```

代码里这样转：

```python
score = max(0.0, 1.0 - dist)
```

加 `max(0, ...)` 是为了防止浮点误差导致负数。

### 2.7 元数据过滤：先 SQL 后向量

`retriever.search(query, source_filter="ingredients")` 会变成：

```python
col.query(
    query_embeddings=[q_vec],
    n_results=4,
    where={"source": "ingredients"}  # 先按元数据过滤
)
```

ChromaDB 先用 `where` 筛出符合条件的子集，再在子集内做向量搜索。这比"先全量召回再过滤"快很多。

**用途**：
- 只搜菜谱不搜食材：`source="recipes"`
- 只搜某一类：`category="肉禽蛋"`
- 只搜低热量：`calories={"$lt": 200}`（ChromaDB 支持 MongoDB 风格操作符）

---

## 3. 文件结构变化

```
health-diet-rag/
├── app/
│   ├── rag/                    # ★ Stage 3 新增
│   │   ├── embedder.py         # embedding 工厂（FastEmbed + Mock）
│   │   ├── loaders.py          # JSON/MD → Document
│   │   ├── indexer.py          # ChromaDB 写入
│   │   └── retriever.py        # ChromaDB 检索
│   ├── api/v1/
│   │   └── rag.py              # ★ 新增：/index, /foods/search
│   ├── core/config.py          # 改：加 RAG 配置
│   └── main.py                 # 改：注册 rag 路由
├── data/                       # ★ Stage 3 新增
│   ├── ingredients.json        # 20 条食材营养数据
│   ├── recipes.json            # 10 条菜谱数据
│   └── nutrition_guides.md     # 多类人群营养指南
└── tests/
    └── test_rag.py             # ★ 30 个 RAG 测试
```

---

## 4. 数据流：从用户查询到检索结果

```
用户 GET /api/v1/foods/search?q=低脂高蛋白的肉&k=3
                │
                ▼
┌────────────────────────────────────┐
│ retriever.search(query, k=3)       │
└────────────────────────────────────┘
                │
   ┌────────────┼─────────────┐
   ▼            ▼             ▼
[q 向量化]   [取索引]      [构造 where]
embed_query  _get_collection   (None)
   │            │
   └──→ col.query(query_embeddings=[v], n_results=3, where=None)
                │
                ▼
        ChromaDB 内部流程：
        1. 用 HNSW 索引快速找候选
        2. 计算 cosine 距离
        3. 按 distance 升序取前 3
        4. 返回 documents/metadata/distances
                │
                ▼
        retriever._parse_raw_results
                │
        dist → similarity = 1 - dist
                │
                ▼
        返回 list[SearchResult]
                │
                ▼
        FastAPI 序列化为 JSON → 用户
```

---

## 5. 跑通验证

### 5.1 跑测试

```bash
cd health-diet-rag
source .venv/bin/activate
pytest tests/test_rag.py -v
# 应看到 30 passed
```

### 5.2 第一次建索引

启动服务：

```bash
uvicorn app.main:app --reload
```

调用建索引接口（首次会下载 90MB 模型，约 10-30 秒）：

```bash
curl -X POST http://localhost:8000/api/v1/index
```

返回：

```json
{
  "status": "ok",
  "counts": {"ingredients": 20, "recipes": 10, "guides": 7, "total": 37},
  "embeddings_used": "FastEmbedWrapper"
}
```

### 5.3 语义检索

```bash
curl 'http://localhost:8000/api/v1/foods/search?q=低脂高蛋白的肉&k=3'
```

返回 top-3 召回：鸡胸肉 / 瘦牛肉 / 鸡蛋。

### 5.4 Swagger 调试

打开 http://localhost:8000/docs，找到 "RAG 检索" 分组，可视化试各种 query。

---

## 6. 关键代码走读

### 6.1 Embedding 单例 + Mock 降级

`app/rag/embedder.py:135`
```python
@lru_cache(maxsize=1)
def get_embeddings() -> EmbeddingsLike:
    if provider == "fastembed":
        try:
            return FastEmbedWrapper(model)
        except Exception:
            return MockEmbeddings(dim=512)  # 网络问题降级
```

**坑**：第一次调用会下载模型，循环 import 时不要触发。

### 6.2 numpy float32 → Python float

`app/rag/embedder.py:73`
```python
return [[float(x) for x in vec] for vec in self._model.embed(texts)]
```

chromadb 1.x 严格类型检查，拒绝 numpy.float32。**这是踩坑后的修复**。

### 6.3 索引幂等性

`app/rag/indexer.py:55`
```python
def index_all(self):
    docs = load_all()
    self.clear()            # 先清空
    self._add_documents(docs)
```

不清空就 add 会导致**数据重复**（同一文档出现多次）。`clear()` 保证幂等。

### 6.4 metadata 值的类型强制

`app/rag/indexer.py:188`
```python
def _coerce_meta_value(v):
    if isinstance(v, list):
        return ",".join(...)  # list 不能直接存，要拼成字符串
```

ChromaDB metadata 只支持 str/int/float/bool。**list/dict 必须先序列化**。

### 6.5 距离 → 相似度转换

`app/rag/retriever.py:114`
```python
score = max(0.0, 1.0 - dist)
```

cosine 模式下，距离 = 1 - 相似度。`max(0, ...)` 防浮点误差。

---

## 7. 常见坑点

### 7.1 依赖地狱：torch / chromadb / numpy

Python 3.13 + macOS 上：
- torch 无 wheel
- chromadb <0.6 需要 numpy<2
- fastembed 在 Python 3.13 上要 numpy>=2.1

**解决**：用 chromadb ≥ 1.0（支持 numpy 2），并绕过 langchain-chroma。

### 7.2 模型首次下载慢

fastembed 第一次会从 HuggingFace 下载模型。中国网络可能慢。
**解法**：设置 HF 镜像：
```bash
export HF_ENDPOINT=https://hf-mirror.com
```

### 7.3 索引重建后查不到数据

可能原因：
1. collection 名变了
2. persist_dir 路径变了
3. 数据格式变了没 clear 就 add

**排查**：`GET /api/v1/index/status` 看 count。

### 7.4 检索结果全是 chunk 边界

Markdown 切分时如果 `chunk_size` 太小，会切在句子中间。
**解法**：
- 用 `RecursiveCharacterTextSplitter`（按段落→句子→字符递归切）
- 设合理的 `chunk_overlap`（默认 50-100）

### 7.5 cosine 距离 vs L2 距离

ChromaDB 默认用 L2（欧氏距离）。**文本场景应该用 cosine**：

```python
client.create_collection(
    name="...",
    metadata={"hnsw:space": "cosine"}  # ← 关键
)
```

L2 受向量模长影响，cosine 只看方向。

### 7.6 metadata 不能存 list

```python
# 错：ChromaDB 会报错
metadata={"tags": ["高蛋白", "低脂"]}

# 对：先拼字符串
metadata={"tags": "高蛋白,低脂"}
```

---

## 8. 检索质量评估（重要！）

光"能召回"不够，还要"召回得准"。简单评估方法：

### 8.1 人工抽样

准备 10 个查询 + 期望答案，看 top-3 召回是否包含：

| 查询 | 期望命中 |
|---|---|
| 低脂高蛋白的肉 | 鸡胸肉 / 虾仁 / 瘦牛肉 |
| 富含 Omega-3 的鱼 | 三文鱼 |
| 低 GI 主食 | 糙米 / 燕麦 / 红薯 / 藜麦 |
| 减脂期适合的早餐 | 希腊酸奶 / 燕麦 / 鸡蛋 |
| 增肌吃什么 | 瘦牛肉 / 鸡蛋 / 三文鱼 |

本项目实现了自动化版本（见 `tests/test_rag.py::TestRetriever`）。

### 8.2 量化指标（进阶）

- **Recall@k**：top-k 内命中期望答案的比例
- **MRR**：Mean Reciprocal Rank，第一名得 1，第二名 0.5，第三名 0.33...
- **NDCG**：考虑相关度排序的指标

学习项目用人工抽样即可。生产需要标注集。

---

## 9. 检查清单

完成本阶段后，你应该能：

- [ ] 解释 embedding 是什么、为什么能做语义检索
- [ ] 解释 cosine 相似度和 L2 距离的区别
- [ ] 写一个最小可用的 ChromaDB 索引器
- [ ] 解释 metadata 过滤为什么比先全量召回再过滤快
- [ ] 实现距离 → 相似度的转换
- [ ] 调通 `/api/v1/index` 建索引
- [ ] 调通 `/api/v1/foods/search` 语义检索
- [ ] 知道如何评估检索质量
- [ ] 解释为什么本项目不用 sentence-transformers

---

## 10. 下一阶段预告

Stage 4 会做：
- 引入 **LangGraph** 真正的 Agent 框架
- 把"健康评估 Agent"和"菜谱 Agent"用 LangGraph 编排
- 菜谱 Agent 调用 RAG 检索器拿食材知识
- Supervisor Agent 协调多个子 Agent
- 学习 StateGraph / Node / Edge / Conditional Routing

---

## 11. 关键代码索引

| 文件 | 行 | 内容 |
|---|---|---|
| `app/rag/embedder.py` | 75 | `FastEmbedWrapper` 适配层 |
| `app/rag/embedder.py` | 135 | `get_embeddings()` 工厂 |
| `app/rag/loaders.py` | 32 | `load_ingredients()` JSON→Document |
| `app/rag/loaders.py` | 95 | `load_guides()` Markdown 切分 |
| `app/rag/indexer.py` | 52 | `index_all()` 主流程 |
| `app/rag/indexer.py` | 122 | `_add_documents()` ChromaDB 写入 |
| `app/rag/retriever.py` | 60 | `search()` 语义检索 |
| `app/rag/retriever.py` | 114 | 距离 → 相似度 |
| `app/api/v1/rag.py` | 65 | `GET /foods/search` 接口 |
