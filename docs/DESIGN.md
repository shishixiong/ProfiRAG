# ProfiRAG 系统设计文档

## 1. 项目概述

ProfiRAG 是一个高级 RAG (Retrieval-Augmented Generation) 系统，支持完整的 RAG 工作流程：
- **文档摄入**: 多格式文档加载、智能分块
- **向量存储**: 多后端支持 (Qdrant/PostgreSQL/本地文件)
- **混合检索**: 向量检索 + BM25 关键词检索
- **查询增强**: HyDE、查询重写、多查询生成
- **重排序**: Cross-Encoder 精排
- **响应生成**: 多种合成模式 (compact/refine/tree_summarize)

## 2. 系统架构

```
┌─────────────────────────────────────────────────────────────────┐
│                        ProfiRAG Pipeline                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐       │
│  │  Ingestion   │    │   Retrieval  │    │  Generation  │       │
│  ├──────────────┤    ├──────────────┤    ├──────────────┤       │
│  │ Loaders      │    │ Pre-Retrieval│    │ Synthesizer  │       │
│  │ Splitters    │───▶│ HybridRetriever───▶│ Formatter    │       │
│  │ Pipelines    │    │ Reranker     │    │ Prompts      │       │
│  │ ImageProcessor│   │ SparseVectorizer│  └──────────────┘       │
│  └──────────────┘    └──────────────┘           │               │
│         │                   │                    ▼               │
│         ▼                   ▼            ┌──────────────┐       │
│  ┌──────────────┐    ┌──────────────┐    │   Agent      │       │
│  │   Storage    │    │   Embedding  │    │  (ReAct)    │       │
│  ├──────────────┤    ├──────────────┤    └──────────────┘       │
│  │ QdrantStore  │    │ CustomOpenAI │                           │
│  │ LocalStore   │    │ Embedding    │                           │
│  │ PostgresStore│    └──────────────┘                           │
│  └──────────────┘                                                │
│         │                                                         │
│         ▼                                                         │
│  ┌──────────────┐                                                │
│  │    Config    │                                                │
│  ├──────────────┤                                                │
│  │ RAGConfig    │                                                │
│  │ EnvSettings  │                                                │
│  └──────────────┘                                                │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
```

## 3. 核心模块设计

### 3.1 配置管理 (`config/settings.py`)

采用 Pydantic 进行配置管理，支持 `.env` 文件和环境变量。

```
RAGConfig (完整配置)
├── StorageConfig (向量存储配置)
│   ├── type: "qdrant" | "local" | "postgres"
│   └── config: Dict[str, Any]
├── EmbeddingConfig (嵌入模型配置)
│   ├── model, dimension, api_key, base_url
├── LLMConfig (LLM配置)
│   ├── model, temperature, max_tokens, api_key, base_url
├── PreRetrievalConfig (预检索配置)
│   ├── use_hyde, use_rewrite, multi_query
├── RetrievalConfig (检索配置)
│   ├── top_k, alpha, use_hybrid, use_bm25
├── RerankingConfig (重排序配置)
│   ├── enabled, model, top_n
└── GenerationConfig (生成配置)
    ├── response_mode, streaming
```

**关键特性**:
- 支持 OpenAI API 兼容的第三方服务 (MiniMax/DashScope)
- Embedding 和 LLM 可使用不同的 API endpoint
- 自动从 `.env` 加载配置

### 3.2 文档摄入 (`ingestion/`)

#### 3.2.1 文档加载器 (`loaders.py`)

```python
DocumentLoader
├── 支持格式: PDF, TXT, MD, DOCX, HTML, JSON, CSV, XLSX, PPTX
├── 方法:
│   ├── load_directory()  # 目录批量加载
│   ├── load_file()       # 单文件加载
│   ├── load_files()      # 多文件加载
│   ├── load_text()       # 纯文本加载
│   └── load_texts()      # 多文本加载
```

#### 3.2.2 文本分块器 (`splitters.py`)

支持三种分块策略：

| 策略 | 类 | 说明 | 适用场景 |
|------|-----|------|---------|
| **Sentence** | `SentenceSplitter` | 按句子分割，保持语义完整性 | 通用场景 |
| **Token** | `TokenTextSplitter` | 按 token 数硬分割 | 需精确控制长度 |
| **Semantic** | `SemanticSplitterNodeParser` | 按语义相似度分割 | 需语义边界清晰 |
| **Chinese** | `ChineseTextSplitter` | 中文标点分割 + 字符长度控制 | 中文文档 |

**ChineseTextSplitter 分块逻辑**:
```
输入文本
    ↓
按中文标点分割 ([。！？；\n])
    ↓
句子逐个加入当前块
    ↓
超過 chunk_size → 保存块 → 新块从 overlap 开始
    ↓
输出 TextNode 列表
```

**关键参数**:
- `chunk_size`: 默认 512 (tokens/characters)
- `chunk_overlap`: 默认 50 (保留上下文连贯性)

#### 3.2.3 摄入管道 (`pipelines.py`)

```python
IngestionPipeline
├── 组合: Loader + Splitter
├── 方法:
│   ├── ingest_directory()
│   ├── ingest_file()
│   ├── ingest_files()
│   ├── ingest_text()
│   └── ingest_texts()
```

### 3.3 向量存储 (`storage/`)

#### 3.3.1 抽象基类 (`base.py`)

```python
BaseVectorStore (ABC)
├── 必须实现方法:
│   ├── add(nodes) → List[node_id]
│   ├── delete(ref_doc_id/node_ids) → bool
│   ├── query(query_bundle) → List[NodeWithScore]
│   ├── get_node(node_id) → TextNode
│   ├── get_ref_doc_info(ref_doc_id) → RefDocInfo
│   ├── persist(path)
│   ├── count() → int
│   ├── clear()
│   └── from_config(config) → BaseVectorStore
├── 辅助方法:
│   └── to_llamaindex_vector_store() → LlamaIndex VectorStore
```

#### 3.3.2 存储后端

| 后端 | 文件 | 特点 |
|------|-----|------|
| **Qdrant** | `qdrant_store.py` | 高性能、支持云端/本地 |
| **PostgreSQL** | `postgres_store.py` | pgvector 扩展、适合现有数据库 |
| **Local** | `local_store.py` | 本地文件存储、无需外部依赖 |

##### Qdrant

**BM25-Only Mode**

当设置 `dense_vector_name=None` 时，Qdrant collection 仅使用稀疏向量创建：

```python
# Example: BM25-only ingestion
store = QdrantStore(
    collection_name="bm25_only_collection",
    client=client,
    use_bm25=True,
    dense_vector_name=None,  # No dense vectors stored
)
```

此模式：
- 减少存储占用（无密集向量）
- 启用纯关键字/BM25检索
- 适用于不需要向量搜索的文本密集型工作负载

#### 3.3.3 存储注册器 (`registry.py`)

```python
StorageRegistry
├── get_store(type, config) → BaseVectorStore
├── register_store(type, cls)
├── get_available_stores() → List[str]
```

### 3.4 嵌入模型 (`embedding/`)

#### 3.4.1 自定义嵌入模型 (`custom_embedding.py`)

**问题**: LlamaIndex 的 `OpenAIEmbedding` 验证模型名称，不支持第三方服务的自定义模型名 (如 `text-embedding-v4`)。

**解决方案**: `CustomOpenAIEmbedding` 继承 `BaseEmbedding`，直接调用 OpenAI SDK:

```python
CustomOpenAIEmbedding(BaseEmbedding)
├── 字段:
│   ├── model: str        # 任意模型名
│   ├── api_key: str
│   ├── api_base: str     # 自定义 endpoint
│   ├── dimensions: int   # 可选
│   └── embed_batch_size: int = 10  # DashScope 限制
├── 核心方法:
│   ├── _get_embedding(text) → List[float]
│   ├── _get_embeddings(texts) → List[List[float]]
│   ├── _get_query_embedding(query)
│   ├── _get_text_embedding(text)
│   ├── _get_text_embeddings(texts)  # 批量处理
│   └── 异步版本: _aget_*
```

### 3.5 检索模块 (`retrieval/`)

#### 3.5.1 预检索 (`query_transform.py`)

三种查询增强技术：

| 技术 | 类 | 原理 | 适用场景 |
|------|-----|------|---------|
| **HyDE** | `HyDEQueryTransform` | 生成假设文档，用其 embedding 检索 | 查询过于简短 |
| **Rewrite** | `QueryRewriter` | LLM 重写查询使其更明确 | 查询模糊不清 |
| **MultiQuery** | `MultiQueryGenerator` | 生成多个查询变体扩大覆盖 | 需广泛检索 |

```python
PreRetrievalPipeline
├── 组合: HyDE + Rewrite + MultiQuery (可选启用)
├── transform(query) → List[QueryBundle]
│   ├── 原始查询
│   ├── HyDE 假设文档 (可选)
│   ├── 重写查询 (可选)
│   └── 多查询变体 (可选)
```

#### 3.5.2 混合检索 (`hybrid.py`)

```python
SparseVectorizer (别名: BM25Index)
├── 中文分词: jieba
├── TF-IDF 加权稀疏向量计算
├── 用于 Qdrant native BM25 hybrid retrieval
├── 方法:
│   ├── fit(texts) / fit_nodes(nodes)  # 构建词汇表和 IDF
│   ├── compute_sparse_vector(text)    # 计算查询稀疏向量
│   └── get_idf_payload() / load_idf_from_payload()

HybridRetriever
├── 组合: VectorStoreIndex + Qdrant native BM25 (或 SparseVectorizer)
├── 融合: Reciprocal Rank Fusion (RRF)
│   score = α/(k + rank_vector) + (1-α)/(k + rank_bm25)
├── 方法:
│   ├── retrieve(query, top_k) → List[NodeWithScore]
│   └── _rrf_fusion(vector_nodes, bm25_nodes)
```

#### 3.5.3 重排序 (`reranker.py`)

```python
CrossEncoderReranker(BaseNodePostprocessor)
├── 模型: cross-encoder/ms-marco-MiniLM-L-6-v2 (默认)
├── 方法:
│   └── _postprocess_nodes(nodes, query_bundle) → List[NodeWithScore]

Reranker
├── 包装器，支持启用/禁用
├── 方法:
│   ├── rerank(query, nodes) → List[NodeWithScore]
│   ├── set_enabled(bool), set_top_n(int)
```

### 3.6 生成模块 (`generation/`)

#### 3.6.1 响应合成器 (`synthesizer.py`)

```python
ResponseSynthesizer
├── 模式:
│   ├── compact: 单次生成 (所有 context 合入一个 prompt)
│   ├── refine: 迭代精炼 (逐块优化答案)
│   ├── tree_summarize: 层级摘要
├── 方法:
│   ├── synthesize(query, nodes) → str
│   ├── synthesize_streaming(query, nodes) → generator
│   ├── synthesize_custom(query, nodes, prompt) → str

ResponseFormatter
├── format_with_sources(response, nodes) → Dict
├── format_markdown(response, nodes) → str
```

#### 3.6.2 Prompt 模板 (`prompts.py`)

```python
PromptTemplates
├── DEFAULT_PROMPT_TEMPLATE    # 英文默认
├── CHINESE_PROMPT_TEMPLATE    # 中文
├── COMPACT_PROMPT_TEMPLATE    # 精简版
├── REFINE_PROMPT_TEMPLATE     # 精炼版
├── 方法:
│   ├── get_template(language, style)
│   ├── format_context(nodes, max_length)
│   └── format_prompt(query, context, template)
```

### 3.7 主管道 (`pipeline/rag_pipeline.py`)

```python
RAGPipeline
├── 初始化:
│   ├── _create_embed_model() → CustomOpenAIEmbedding
│   ├── _create_llm() → OpenAI
│   ├── _create_vector_store() → BaseVectorStore
│   ├── _create_index() → VectorStoreIndex
│   ├── PreRetrievalPipeline
│   ├── HybridRetriever
│   ├── Reranker
│   ├── ResponseSynthesizer
├── 摄入:
│   ├── ingest_documents(docs) → List[doc_id]
│   └── ingest_nodes(nodes) → List[node_id]
├── 查询:
│   ├── query(query_str, top_k) → Dict
│   │   ├── 1. Pre-retrieval: 查询变换
│   │   ├── 2. Retrieval: 混合检索
│   │   ├── 3. Post-retrieval: 重排序
│   │   ├── 4. Generation: 响应合成
│   ├── query_stream(query_str) → generator
├── 管理:
│   ├── delete_document(doc_id)
│   ├── clear()
│   ├── get_stats()
```

### 3.8 评估模块 (`evaluation/`)

#### 3.8.1 评估数据集 (`dataset.py`)

```python
EvalItem
├── query: str                      # 查询字符串
├── expected_ids: List[str]         # 期望的相关文档 ID
├── expected_texts: List[str]       # 期望文本（可选）
├── reference_answer: str           # 参考答案（可选，用于 correctness）

EvalDataset
├── items: List[EvalItem]
├── 方法:
│   ├── from_json(path)             # JSON 加载
│   ├── from_csv(path)              # CSV 加载
│   ├── save(path)                  # 保存
│   ├── get_queries()               # 获取所有查询
│   ├── get_expected_ids()          # 获取所有期望 ID
│   └── get_reference_answers()     # 获取所有参考答案
```

#### 3.8.2 检索评估 (`retrieval.py`)

支持指标：

| 指标 | 说明 | 计算方式 |
|------|------|---------|
| `hit_rate` | 命中率 | 是否检索到相关文档 (0/1) |
| `mrr` | 平均倒数排名 | 第一个相关文档位置的倒数 |
| `precision` | 精确度 | 检索结果中相关文档比例 |
| `recall` | 召回率 | 相关文档被检索到的比例 |
| `ndcg` | 归一化折损累积增益 | 排序质量评估 |
| `ap` | 平均精确度 | Precision 曲线平均值 |

```python
RetrievalEvaluator
├── 初始化:
│   ├── retriever: BaseRetriever    # 检索器
│   ├── metrics: List[str]          # 指标列表
├── 方法:
│   ├── evaluate(query, expected_ids) → RetrievalEvalResult
│   ├── evaluate_batch(queries, expected_ids_list) → List[Result]
│   ├── evaluate_dataset(dataset) → List[Result]
│   ├── get_metrics_summary(results) → Dict[str, float]
```

#### 3.8.3 响应评估 (`response.py`)

支持评估器：

| 评估器 | 说明 | 需要参数 |
|--------|------|---------|
| `faithfulness` | 忠实度 | query, response, contexts |
| `relevancy` | 相关性 | query, response, contexts |
| `correctness` | 正确性 | query, response, reference |
| `answer_relevancy` | 答案相关性 | query, response |
| `context_relevancy` | 上下文相关性 | query, contexts |

```python
ResponseEvaluator
├── 初始化:
│   ├── llm: LLM                    # LLM 用于评估
│   ├── evaluators: List[str]       # 评估器列表
├── 方法:
│   ├── evaluate(query, response, contexts, reference) → Dict[str, EvaluationResult]
│   ├── evaluate_batch(...) → Dict[str, List[EvaluationResult]]
│   ├── get_metrics_summary(results) → Dict[str, Dict]
│   │   # 返回 {"faithfulness": {"mean": 0.85, "passing_rate": 0.90}, ...}
```

#### 3.8.4 批量评估执行器 (`runner.py`)

```python
RAGEvalRunner
├── 初始化:
│   ├── pipeline: RAGPipeline       # RAG 管道
│   ├── llm: LLM                    # LLM
│   ├── retrieval_metrics: List     # 检索指标
│   ├── response_metrics: List      # 响应评估器
│   ├── top_k: int                  # 检索数量
├── 方法:
│   ├── run_single(item) → EvalResultItem
│   ├── run_evaluation(dataset) → RAGEvalResults
│   ├── quick_eval(queries, expected_ids_list) → RAGEvalResults
│   └── get_available_metrics() → Dict

RAGEvalResults
├── items: List[EvalResultItem]
├── retrieval_summary: Dict[str, float]
├── response_summary: Dict[str, Dict]
├── total_time: float
├── 方法:
│   ├── save(path)
│   └── get_summary_text() → str
```

## 4. 数据流

### 4.1 文档摄入流程

```
原始文档 (PDF/TXT/MD/...)
    ↓
DocumentLoader.load_directory()
    ↓
Document 对象列表
    ↓
TextSplitter.split_documents()
    ↓
TextNode 列表 (分块)
    ↓
CustomOpenAIEmbedding._get_text_embeddings()
    ↓
向量列表
    ↓
VectorStore.add(nodes)
    ├── Qdrant native BM25: SparseVectorizer 计算稀疏向量
    └── 其他存储: 向量存储完成摄入
    ↓
完成摄入
```

### 4.2 查询流程

```
用户查询
    ↓
PreRetrievalPipeline.transform()
    ├── HyDE (假设文档)
    ├── Rewrite (重写查询)
    └── MultiQuery (查询变体)
    ↓
QueryBundle 列表
    ↓
HybridRetriever.retrieve()
    ├── VectorStoreIndex.as_retriever()
    ├── BM25Index.retrieve()
    └── RRF 融合
    ↓
NodeWithScore 列表
    ↓
Reranker.rerank()
    ↓
精排后的 NodeWithScore 列表
    ↓
ResponseSynthesizer.synthesize()
    ├── Context 截断 (可选)
    └── LLM 生成
    ↓
响应文本 + Source 信息
```

## 5. 脚本工具

### 5.1 文档摄入脚本 (`scripts/ingest_documents.py`)

```bash
python scripts/ingest_documents.py --documents ./documents
python scripts/ingest_documents.py --file ./documents/example.pdf
```

功能:
- 加载配置 (.env)
- 初始化 RAGPipeline
- 加载文档并摄入向量存储
- 输出统计信息

### 5.2 分块脚本 (`scripts/chunk_documents.py`)

```bash
python scripts/chunk_documents.py --input ./documents --output ./chunks
python scripts/chunk_documents.py --input ./documents --output ./chunks --splitter chinese
```

功能:
- 仅执行分块，不做 embedding 或向量存储
- 支持输出格式: txt/json/jsonl
- 自动检测语言

### 5.3 评估脚本 (`scripts/evaluate_rag.py`)

```bash
# 运行评估
python scripts/evaluate_rag.py --dataset ./eval_data.json
python scripts/evaluate_rag.py --dataset ./eval_data.json --output ./results.json

# 查看可用指标
python scripts/evaluate_rag.py --list-metrics

# 创建样本数据集
python scripts/evaluate_rag.py --create-sample --dataset ./eval_data.json

# 自定义指标
python scripts/evaluate_rag.py \
    --dataset ./eval_data.json \
    --retrieval-metrics hit_rate,mrr,precision,recall,ndcg \
    --response-metrics faithfulness,relevancy,correctness
```

功能:
- 检索评估: hit_rate, mrr, precision, recall, ndcg, ap
- 响应评估: faithfulness, relevancy, correctness, answer_relevancy, context_relevancy
- 输出 JSON 结果和摘要文本

评估数据集格式 (`eval_data.json`):
```json
{
    "items": [
        {
            "query": "查询问题",
            "expected_ids": ["doc_id_1", "doc_id_2"],
            "reference_answer": "参考答案（可选）"
        }
    ]
}
```

## 6. 扩展点

### 6.1 新增向量存储后端

1. 继承 `BaseVectorStore`
2. 实现所有抽象方法
3. 在 `StorageRegistry` 注册

### 6.2 新增分块策略

1. 在 `TextSplitter._create_splitter()` 添加新类型
2. 或创建独立的分块器类

### 6.3 新增查询变换

1. 继承 `BaseQueryTransform` (HyDE 风格)
2. 或创建独立组件 (Rewrite 风格)
3. 在 `PreRetrievalPipeline` 集成

### 6.4 新增响应模式

1. 使用 LlamaIndex 内置模式
2. 或在 `ResponseSynthesizer.synthesize_custom()` 自定义 prompt

## 7. 配置示例

### 7.1 .env 文件

```env
# LLM 配置 (MiniMax)
OPENAI_API_KEY=sk-xxx
OPENAI_LLM_MODEL=MiniMax-M2.7
OPENAI_BASE_URL=https://api.minimax.chat/v1

# Embedding 配置 (DashScope)
OPENAI_EMBEDDING_API_KEY=sk-xxx
OPENAI_EMBEDDING_MODEL=text-embedding-v4
OPENAI_EMBEDDING_DIMENSION=1536
OPENAI_EMBEDDING_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1

# 存储配置
PROFIRAG_STORAGE_TYPE=qdrant
QDRANT_HOST=localhost
QDRANT_PORT=6333
QDRANT_COLLECTION_NAME=profirag

# 检索配置
PROFIRAG_TOP_K=10
PROFIRAG_ALPHA=0.5
PROFIRAG_USE_HYBRID=true
PROFIRAG_USE_BM25=true

# 重排序配置
PROFIRAG_RERANK_ENABLED=true
PROFIRAG_RERANK_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2
```

## 8. 依赖关系

```
ProfiRAG
├── llama-index (核心框架)
│   ├── llama-index-core
│   ├── llama-index-embeddings-openai
│   └── llama-index-llms-openai
├── qdrant-client (向量数据库)
├── sentence-transformers (Cross-Encoder)
├── jieba (中文分词)
├── pydantic / pydantic-settings (配置管理)
├── openai (API SDK)
└── python-dotenv (环境变量)
```

## 9. 目录结构

```
ProfiRAG/
├── src/profirag/
│   ├── __init__.py
│   ├── config/
│   │   ├── __init__.py
│   │   └── settings.py          # 配置管理
│   ├── embedding/
│   │   ├── __init__.py
│   │   └── custom_embedding.py  # 自定义嵌入模型
│   ├── generation/
│   │   ├── __init__.py
│   │   ├── prompts.py           # Prompt 模板
│   │   └── synthesizer.py       # 响应合成
│   ├── ingestion/
│   │   ├── __init__.py
│   │   ├── loaders.py           # 文档加载
│   │   ├── pipelines.py         # 摄入管道
│   │   ├── splitters.py         # 文本分块
│   │   └── image_processor.py   # PDF 图片处理 + VLM 描述生成
│   ├── pipeline/
│   │   ├── __init__.py
│   │   └── rag_pipeline.py      # 主 RAG 管道
│   ├── retrieval/
│   │   ├── __init__.py
│   │   ├── hybrid.py            # 混合检索
│   │   ├── sparse_vectorizer.py # TF-IDF 稀疏向量 (BM25Index 别名)
│   │   ├── query_transform.py   # 查询变换 (HyDE/Rewrite/MultiQuery)
│   │   └── reranker.py          # 重排序
│   ├── agent/
│   │   ├── __init__.py
│   │   ├── react_agent.py       # ReAct 代理
│   │   └── tools.py             # RAG 工具集
│   ├── evaluation/              # 评估模块
│   │   ├── __init__.py
│   │   ├── dataset.py           # 评估数据集
│   │   ├── retrieval.py         # 检索评估
│   │   ├── response.py          # 响应评估
│   │   ├── chunking.py          # 分块质量评估
│   │   └── runner.py            # 批量评估执行器
│   └── storage/
│       ├── __init__.py
│       ├── base.py               # 抽象基类
│       ├── local_store.py        # 本地存储
│       ├── postgres_store.py     # PostgreSQL
│       ├── qdrant_store.py       # Qdrant
│       └── registry.py           # 注册器
├── scripts/
│   ├── chunk_documents.py        # 分块脚本
│   ├── ingest_documents.py       # 摄入脚本
│   ├── evaluate_rag.py           # 评估脚本
│   ├── eval_retrieval_flow.py   # 检索流程评估
│   └── eval_response_flow.py    # 响应生成评估
├── docs/
│   └── DESIGN.md                 # 系统设计文档
├── .env                          # 配置文件
├── pyproject.toml                # 项目配置
└── README.md
```